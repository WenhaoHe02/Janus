# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM foreground gather scheduling, installed before model startup."""

# Standard
from functools import wraps
import json
import os
from typing import Any

# First Party
from lmcache.v1.fixed_boundary_gather import FixedBoundaryGathers


def install_fixed_boundary(
    predictor_class: type,
    sink_class: type,
    io_manager_class: type,
    transport_class: type,
) -> None:
    """Install foreground-only scheduling around existing public interfaces.

    Args:
        predictor_class: GLM manager exposing source and request lifecycle hooks.
        sink_class: Physical sink exposing submit and an owner-mode constructor.
        io_manager_class: KV manager retaining the original gather/consumer gates.
        transport_class: GPU owner transport; the public gather call is counted.
    Raises:
        RuntimeError: Unsupported configuration or repeated installation.
    Notes:
        Disabled by default. It adds no collective and changes no candidate IDs.
        Exact per-request submission counts are printed after request completion.
    """
    if os.getenv("LMCACHE_GLM_DSA_FIXED_BOUNDARY_GATHER") != "1":
        return
    required = {
        "LMCACHE_GLM_DSA_ASYNC_PREDICTION": "0",
        "LMCACHE_GLM_DSA_FULL_LAYER_LOOKAHEAD": "1",
        "LMCACHE_GLM_DSA_ASYNC_SHARED_CORRECTION": "0",
        "LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER": "1",
        "LMCACHE_GLM_DSA_PIPELINE_SHARED_GATHERS": "1",
        "LMCACHE_CSA_OWNER_GPU_METADATA": "1",
        "LMCACHE_GLM_DSA_DEVICE_OWNER_READ": "1",
        "LMCACHE_CSA_OWNER_KV_PACKET": "1",
    }
    if any(os.getenv(key, "0") != value for key, value in required.items()):
        raise RuntimeError("fixed-boundary gather requires synchronous owner pipeline")
    if getattr(predictor_class, "fixed_boundary_installed", False):
        raise RuntimeError("fixed-boundary hooks already installed")
    original_submit = sink_class.submit
    original_after = predictor_class.after_source_layer
    original_end = predictor_class.end_request
    original_cancel = predictor_class.cancel_active_request
    original_gather = transport_class.gather_owner_rows_gpu_into

    @wraps(original_gather)
    def gather(self: Any, **kwargs: Any) -> Any:
        result = original_gather(self, **kwargs)
        self.fixed_boundary_gather_count = (
            getattr(self, "fixed_boundary_gather_count", 0) + 1
        )
        return result

    def gather_count(self: Any) -> int:
        """Return successful public owner-gather submissions, without GPU waits."""
        return int(getattr(self._shard_transport, "fixed_boundary_gather_count", 0))

    def gather_rank(self: Any) -> int:
        """Return the rank used by this owner transport without communication."""
        return int(getattr(self._shard_transport, "rank", -1))

    @wraps(original_submit)
    def submit(self: Any, event: Any) -> Any:
        scheduler = getattr(self, "fixed_boundary_scheduler", None)
        if event.correction and scheduler is not None:
            scheduler.discard(event.schedule.target_layer)
        result = original_submit(self, event)
        if event.correction or not self._enable_prediction:
            return result
        manager = self._attention_kv_manager
        token = manager.active_request_token
        if not token or manager.active_request_id != event.request_id:
            return result
        if not self._preserve_owner_partition or self._shared_prediction_mode not in {
            "all",
            "staged",
        }:
            raise RuntimeError("fixed-boundary gather requires every owner consumer")
        if scheduler is None:
            scheduler = self.fixed_boundary_scheduler = FixedBoundaryGathers()
        request = (event.request_id, token)
        if scheduler.request != request:
            self.fixed_boundary_start_count = manager.owner_gather_submission_count()
        scheduler.add(
            request,
            event.schedule.source_layer,
            event.schedule.target_layer,
            event.schedule.consumer_layers,
            staged=self._shared_prediction_mode == "staged",
        )
        return result

    def advance(self: Any, layer: int) -> tuple[int, ...]:
        """Enqueue consumers whose original staging boundary has been reached."""
        scheduler = getattr(self, "fixed_boundary_scheduler", None)
        if scheduler is None or not scheduler.groups:
            return ()
        manager = self._attention_kv_manager
        request = (manager.active_request_id, manager.active_request_token)
        launched = scheduler.advance(
            request,
            layer,
            lambda target: manager.launch_layer_gather(
                target, timeout_s=self._gate_timeout_s()
            ),
        )
        if launched and not getattr(self, "fixed_boundary_ack", False):
            self.fixed_boundary_ack = True
            print("GLM_FIXED_BOUNDARY_V1011_ACTIVE", layer, launched, flush=True)
        return launched

    def report(self: Any, request_id: str | None) -> None:
        """Report completed scheduling counters and discard request metadata."""
        scheduler = getattr(self, "fixed_boundary_scheduler", None)
        if scheduler is None:
            return
        if scheduler.request and request_id == scheduler.request[0]:
            count = self._attention_kv_manager.owner_gather_submission_count()
            print(
                "GLM_FIXED_BOUNDARY_V1011_REQUEST",
                json.dumps(
                    dict(
                        request_id=request_id,
                        rank=self._attention_kv_manager.owner_gather_rank(),
                        moved_layers=scheduler.launched_layers,
                        pending_groups=len(scheduler.groups),
                        owner_gather_calls=count - self.fixed_boundary_start_count,
                    )
                ),
                flush=True,
            )
        scheduler.reset()

    @wraps(original_after)
    def after(self: Any, source_layer: int, *args: Any, **kwargs: Any) -> Any:
        result = original_after(self, source_layer, *args, **kwargs)
        sink = getattr(self._submit_prefetch, "__self__", None)
        if isinstance(sink, sink_class):
            sink.advance_fixed_boundary(source_layer)
        return result

    @wraps(original_end)
    def end(self: Any, request_id: str) -> Any:
        result = original_end(self, request_id)
        sink = getattr(self._submit_prefetch, "__self__", None)
        if isinstance(sink, sink_class):
            sink.finish_fixed_boundary(request_id)
        return result

    @wraps(original_cancel)
    def cancel(self: Any) -> Any:
        # vLLM disables prefill prediction on its first decode forward, before
        # end_request. Capture the public scheduler's request before resetting;
        # reporting only from end_request would lose every completed prefill.
        sink = getattr(self._submit_prefetch, "__self__", None)
        request_id = None
        if isinstance(sink, sink_class):
            scheduler = getattr(sink, "fixed_boundary_scheduler", None)
            if scheduler is not None and scheduler.request is not None:
                request_id = scheduler.request[0]
        result = original_cancel(self)
        if isinstance(sink, sink_class):
            sink.finish_fixed_boundary(request_id)
        return result

    transport_class.gather_owner_rows_gpu_into = gather
    io_manager_class.owner_gather_submission_count = gather_count
    io_manager_class.owner_gather_rank = gather_rank
    sink_class.submit = submit
    sink_class.advance_fixed_boundary = advance
    sink_class.finish_fixed_boundary = report
    predictor_class.after_source_layer = after
    predictor_class.end_request = end
    predictor_class.cancel_active_request = cancel
    predictor_class.fixed_boundary_installed = True
