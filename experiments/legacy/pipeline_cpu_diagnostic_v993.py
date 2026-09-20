# SPDX-License-Identifier: Apache-2.0
"""Request-scoped CPU call trees at public GLM prediction and consumer gates."""

# Standard
import cProfile
from dataclasses import dataclass, field
from functools import wraps
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

active_pipeline_recorder: Any = None


def wrap_active_pipeline_method(method: Callable[..., Any], category: str) -> Callable[..., Any]:
    """Wrap a public loader method without eagerly importing GLM or its manager."""
    cached_recorder = None
    cached_wrapper = None

    @wraps(method)
    def dispatch(*args: Any, **kwargs: Any) -> Any:
        nonlocal cached_recorder, cached_wrapper
        recorder = active_pipeline_recorder
        if recorder is None:
            return method(*args, **kwargs)
        if recorder is not cached_recorder:
            cached_recorder = recorder
            cached_wrapper = recorder.wrap(method, category)
        return cached_wrapper(*args, **kwargs)

    return dispatch


@dataclass
class ProfileRecord:
    """One request/category/thread profile, retained only until request teardown."""

    profiler: Any = field(default_factory=cProfile.Profile)
    calls: list[dict[str, Any]] = field(default_factory=list)


class PipelineRecorder:
    """Record top-level public gate calls without nesting Python profilers.

    Args:
        directory: Explicit diagnostic-only output directory.
    Notes:
        CPU wall scopes include waiting and enqueue work, not GPU kernel time.
        The serving experiment is single-sequence. A new request flushes any
        unfinished predecessor so retained state cannot grow with the workload.
    """

    def __init__(self, directory: Path, *, python_profile: bool = True) -> None:
        self.directory = directory
        self.python_profile = python_profile
        self.request_id: str | None = None
        self.capture_prefill = False
        self.records: dict[tuple[int, str], ProfileRecord] = {}
        self.lock = threading.RLock()
        self.thread_state = threading.local()

    def begin(self, request_id: str) -> None:
        """Begin a request, flushing a superseded request with an explicit label."""
        with self.lock:
            if self.request_id is not None:
                self.flush(self.request_id, 'next_begin')
            self.request_id = str(request_id)
            self.capture_prefill = True

    def flush(self, request_id: str, reason: str) -> None:
        """Write profiles after request work; stale teardown cannot flush a new ID."""
        with self.lock:
            if self.request_id != str(request_id):
                return
            safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', str(request_id))[:120]
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                for (tid, category), record in self.records.items():
                    stem = f'pid{os.getpid()}_tid{tid}_{safe_id}_{time.time_ns()}_{category}'
                    profile_path = self.directory / f'{stem}.pstats'
                    if self.python_profile:
                        record.profiler.dump_stats(str(profile_path))
                    result = dict(request_id=request_id, pid=os.getpid(), thread_id=tid,
                                  category=category, flush_reason=reason,
                                  diagnostic_only=True, calls=record.calls,
                                  wall_total_ms=sum(row['wall_ms'] for row in record.calls),
                                  profile=profile_path.name if self.python_profile else None,
                                  python_profile=self.python_profile)
                    (self.directory / f'{stem}.json').write_text(json.dumps(result), encoding='utf-8')
            except OSError as error:
                print(f'PIPELINE_CPU_DIAGNOSTIC_WRITE_FAILED {error}', flush=True)
            self.records.clear()
            self.request_id = None
            self.capture_prefill = False

    def wrap(self, method: Callable[..., Any], category: str,
             *, tensor_position: int | None = None,
             tensor_keyword: str | None = None) -> Callable[..., Any]:
        """Wrap a gate without changing its return/exception or tensor arguments.

        Args:
            method: Original unbound public class method.
            category: CPU scope label in the diagnostic record.
            tensor_position: Optional positional index of query-shaped data;
                a one-row invocation ends prefill capture for this request.
            tensor_keyword: Keyword spelling of that same optional tensor.
        Returns:
            Argument-preserving wrapper. Nested wrappers use the outer call tree.
        """
        @wraps(method)
        def measured(*args: Any, **kwargs: Any) -> Any:
            if tensor_position is not None:
                tensor = args[tensor_position] if len(args) > tensor_position else kwargs.get(tensor_keyword)
                shape = getattr(tensor, 'shape', ())
                if shape and int(shape[0]) <= 1:
                    self.capture_prefill = False
            if (not self.capture_prefill or self.request_id is None
                    or self.python_profile and getattr(self.thread_state, 'active', False)):
                return method(*args, **kwargs)
            key = (threading.get_ident(), category)
            with self.lock:
                record = self.records.get(key)
                if record is None:
                    record = self.records[key] = ProfileRecord()
            previous_active = getattr(self.thread_state, 'active', False)
            previous_category = getattr(self.thread_state, 'category', None)
            previous_layer = getattr(self.thread_state, 'layer_id', None)
            layer = args[1] if len(args) > 1 and isinstance(args[1], int) else kwargs.get('layer_id', previous_layer)
            self.thread_state.active = True
            self.thread_state.category = category
            self.thread_state.layer_id = layer
            started = time.perf_counter()
            wall_ns = time.time_ns()
            cpu_ns = time.thread_time_ns()
            succeeded = False
            nvtx_handle = None
            if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_NVTX_PHASES') == '1':
                import torch
                nvtx_handle = torch.cuda.nvtx.range_start(
                    f'GLM_HOST/{category}/L{layer}/{self.request_id}')
            if self.python_profile:
                record.profiler.enable()
            try:
                result = method(*args, **kwargs)
                succeeded = True
                return result
            finally:
                if self.python_profile:
                    record.profiler.disable()
                record.calls.append(dict(wall_start_ns=wall_ns,
                    wall_ms=(time.perf_counter() - started) * 1000.0,
                    thread_cpu_ms=(time.thread_time_ns() - cpu_ns) / 1e6,
                    item_count=len(args[1]) if len(args)>1 and isinstance(args[1], (dict,list,tuple)) else None,
                    mapping_identity=id(args[1]) if len(args)>1 and isinstance(args[1],dict) else None,
                    layer_id=layer, parent_category=previous_category,
                    succeeded=succeeded))
                self.thread_state.active = previous_active
                self.thread_state.category = previous_category
                self.thread_state.layer_id = previous_layer
                if nvtx_handle is not None:
                    torch.cuda.nvtx.range_end(nvtx_handle)
        return measured

    def wrap_forward_phase(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Emit handle-based forward NVTX ranges without changing phase behavior.

        Ranges are per thread/manager, so multiple cache managers need not close
        in stack order. Enabled only by the explicit diagnostic environment flag.
        Original arguments, returns and exceptions are preserved.
        """
        @wraps(method)
        def phase(owner: Any, value: str, *args: Any, **kwargs: Any) -> Any:
            if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_NVTX_PHASES') != '1':
                return method(owner, value, *args, **kwargs)
            import torch
            ranges = getattr(self.thread_state, 'forward_ranges', None)
            if ranges is None:
                ranges = self.thread_state.forward_ranges = {}
            identity = id(owner)
            if value != 'inactive':
                previous = ranges.pop(identity, None)
                if previous is not None:
                    torch.cuda.nvtx.range_end(previous)
                ranges[identity] = torch.cuda.nvtx.range_start(
                    f'GLM_FORWARD/{value}/{self.request_id}')
            succeeded = False
            try:
                result = method(owner, value, *args, **kwargs)
                succeeded = True
                return result
            finally:
                if value == 'inactive' or not succeeded:
                    handle = ranges.pop(identity, None)
                    if handle is not None:
                        torch.cuda.nvtx.range_end(handle)
        return phase


def install_pipeline_profiling(manager_class: type, sink_class: type, directory: Path,
                              *, python_profile: bool = True) -> PipelineRecorder:
    """Install diagnostic wrappers on public GLM lifecycle and gate interfaces.

    Args:
        manager_class: GLM predictive manager class exposing begin/end and gates.
        sink_class: Physical sink class exposing consumer launch/wait gates.
        directory: Destination for per-request CPU profiles, not serving results.
        python_profile: If False, collect wall timestamps, including nested scopes,
            without activating cProfile. Nested durations must not be added.
    Returns:
        Recorder supporting additional public-method diagnostic wrappers.
    Raises:
        RuntimeError: Profiling already installed on this class.
        AttributeError: A required public interface is absent.
    """
    if getattr(manager_class, 'pipeline_cpu_diagnostic_v993', False):
        raise RuntimeError('Pipeline diagnostic already installed')
    global active_pipeline_recorder
    recorder = PipelineRecorder(directory, python_profile=python_profile)
    active_pipeline_recorder = recorder
    original_begin = manager_class.begin_request
    original_end = manager_class.end_request

    @wraps(original_begin)
    def begin(owner: Any, request_id: str) -> Any:
        result = original_begin(owner, request_id)
        recorder.begin(request_id)
        return result

    @wraps(original_end)
    def end(owner: Any, request_id: str) -> Any:
        try:
            return original_end(owner, request_id)
        finally:
            recorder.flush(request_id, 'end_request')

    manager_class.begin_request = begin
    manager_class.end_request = end
    manager_class.after_source_layer = recorder.wrap(
        manager_class.after_source_layer, 'prediction_dispatch', tensor_position=2,
        tensor_keyword='hidden_states')
    manager_class.observe_true_topk = recorder.wrap(
        manager_class.observe_true_topk, 'full_correction', tensor_position=2,
        tensor_keyword='true_topk')
    sink_class.start_consumer_prefetch = recorder.wrap(
        sink_class.start_consumer_prefetch, 'consumer_launch')
    sink_class.wait_for_consumer = recorder.wrap(
        sink_class.wait_for_consumer, 'consumer_wait')
    manager_class.pipeline_cpu_diagnostic_v993 = True
    return recorder
