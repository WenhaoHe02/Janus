# SPDX-License-Identifier: Apache-2.0
"""Task-scoped CUDA dependencies for prediction-driven background reads."""

from __future__ import annotations

# Standard
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import os
import threading
import time
from typing import Any, Callable, TypeVar

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)
_T = TypeVar("_T")
_active_dependency: ContextVar[PredictionReadDependency | None] = ContextVar(
    "lmcache_prediction_read_dependency", default=None
)


def current_prediction_read_dependency() -> PredictionReadDependency | None:
    """Return this task's bound dependency, or None for ordinary demand reads."""
    return _active_dependency.get()


def bind_prediction_read_dependency(
    callback: Callable[..., _T],
    *,
    target_layer: int,
    source_layer: int | None = None,
    dependency: PredictionReadDependency | None = None,
) -> Callable[..., _T]:
    """Capture source readiness before queuing a prediction-read task.

    Args:
        callback: Original synchronous background-read callable.
        target_layer: Physical consumer receiving the predicted KV.
        source_layer: Prediction source label, when known. For staged consumers
            the event is recorded at task release, not at original prediction.
        dependency: Optional explicit event record, also useful for CPU tests.
    Returns:
        A callable preserving arguments, results and exceptions, with a scoped
        dependency. Legacy mode returns the original callable unchanged.
    Raises:
        ValueError: Unknown mode or unsafe non-FIFO/asynchronous producer setup.
    Notes:
        Opt-in modes are capture (observe legacy waiting) and source (wait on
        the captured event). Context is reset on every return/exception, and
        never implicitly inherited by unrelated executor tasks.
    """
    mode = os.getenv("LMCACHE_GLM_PREDICTION_IO_DEPENDENCY", "legacy")
    if mode == "legacy":
        return callback
    if mode not in ("capture", "source"):
        raise ValueError("Unknown prediction I/O dependency mode")
    if (
        int(os.getenv("LMCACHE_GLM_DSA_OWNER_READ_WORKERS", "1")) != 1
        or os.getenv("LMCACHE_GLM_DSA_ASYNC_PREDICTION", "0") != "0"
    ):
        raise ValueError(
            "Source dependency requires FIFO owner reads and synchronous prediction"
        )
    if dependency is None:
        # Third Party
        import torch

        if not torch.cuda.is_available():
            return callback
        stream = torch.cuda.current_stream()
        timed = os.getenv("LMCACHE_TUTTI_PREDICTION_DEPENDENCY_PROFILE", "0") == "1"
        ready = torch.cuda.Event(enable_timing=timed)
        ready.record(stream)
        dependency = PredictionReadDependency(
            event=ready,
            device=stream.device,
            source_stream_ptr=int(stream.cuda_stream),
            submitted_ns=time.monotonic_ns(),
            target_layer=int(target_layer),
            source_layer=source_layer,
            timed=timed,
        )

    @wraps(callback)
    def invoke(*args: Any, **kwargs: Any) -> _T:
        token = _active_dependency.set(dependency)
        try:
            return callback(*args, **kwargs)
        finally:
            _active_dependency.reset(token)

    return invoke


def wait_for_prediction_read_dependency(
    io_stream: Any, default_stream: Any
) -> PredictionReadWaitProbe | None:
    """Order a read on its source event, or retain the original stream wait.

    Args:
        io_stream: CUDA stream that will copy descriptors and submit the read.
        default_stream: Current stream used by the original dependency path.
    Returns:
        Optional diagnostic event pair. Emit it only after the existing I/O
        completion synchronization; this function adds no CPU wait.
    Raises:
        ValueError: The mode is unsupported or the source belongs to another GPU.
    """
    dependency = current_prediction_read_dependency()
    mode = os.getenv("LMCACHE_GLM_PREDICTION_IO_DEPENDENCY", "legacy")
    if mode not in ("legacy", "capture", "source"):
        raise ValueError("Unknown prediction I/O dependency mode")
    if dependency is None or mode == "legacy":
        io_stream.wait_stream(default_stream)
        return None
    if dependency.device != io_stream.device:
        raise ValueError("Prediction read dependency belongs to another CUDA device")
    probe = None
    if dependency.timed:
        # Third Party
        import torch

        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record(io_stream)
        probe = PredictionReadWaitProbe(
            dependency,
            before,
            after,
            mode,
            int(default_stream.cuda_stream),
            int(io_stream.cuda_stream),
            time.monotonic_ns(),
        )
    if mode == "source":
        io_stream.wait_event(dependency.event)
    else:
        io_stream.wait_stream(default_stream)
    if probe is not None:
        probe.after.record(io_stream)
    return probe


def finish_prediction_read_wait_probe(
    probe: PredictionReadWaitProbe | None,
) -> dict[str, Any] | None:
    """Log completed diagnostic timestamps without synchronizing any stream.

    Args:
        probe: Result of wait_for_prediction_read_dependency for this batch.
    Returns:
        Recorded metrics, or None when instrumentation is disabled. GPU times
        are None if their events are not complete; no wait is added to obtain them.
    """
    if probe is None:
        return None
    dependency = probe.dependency
    pair_ready = probe.before.query() and probe.after.query()
    source_ready = dependency.event.query()
    record = dict(
        mode=probe.mode,
        pid=os.getpid(),
        thread_id=threading.get_ident(),
        target_layer=dependency.target_layer,
        source_layer=dependency.source_layer,
        source_stream_ptr=dependency.source_stream_ptr,
        current_stream_ptr=probe.current_stream_ptr,
        io_stream_ptr=probe.io_stream_ptr,
        task_to_io_host_ms=(probe.io_enqueue_ns - dependency.submitted_ns) / 1e6,
        io_dependency_gpu_ms=probe.before.elapsed_time(probe.after)
        if pair_ready
        else None,
        release_ready_to_io_gpu_ms=dependency.event.elapsed_time(probe.after)
        if pair_ready and source_ready
        else None,
        source_complete=bool(source_ready),
        diagnostic_only=True,
    )
    logger.info("TUTTI_PREDICTION_DEPENDENCY %s", json.dumps(record, sort_keys=True))
    return record


@dataclass(frozen=True)
class PredictionReadDependency:
    """Immutable readiness captured at foreground task submission/release."""

    event: Any
    device: Any
    source_stream_ptr: int
    submitted_ns: int
    target_layer: int
    source_layer: int | None = None
    timed: bool = False


@dataclass(frozen=True)
class PredictionReadWaitProbe:
    """Optional event pair around the existing upstream read dependency."""

    dependency: PredictionReadDependency
    before: Any
    after: Any
    mode: str
    current_stream_ptr: int
    io_stream_ptr: int
    io_enqueue_ns: int
