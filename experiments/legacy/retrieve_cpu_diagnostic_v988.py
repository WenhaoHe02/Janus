# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU call profiling of one public retrieve call; never production timing."""

# Standard
import cProfile
from functools import wraps
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import uuid


def wrap_retrieve_for_diagnostics(
    retrieve: Callable[..., Any], directory: Path
) -> Callable[..., Any]:
    """Wrap a public retrieve callable without changing its arguments or result.

    Args:
        retrieve: Original callable, including its public self argument.
        directory: Explicit experiment-only destination for trusted profile files.
    Returns:
        Callable preserving the original return value and exception behavior.
    Notes:
        Profiling perturbs CPU execution. These times are diagnostic only.
        Profile-file failures are reported separately, not substituted for an
        original retrieval exception or return value.
    """

    @wraps(retrieve)
    def profiled(*args: Any, **kwargs: Any) -> Any:
        request_id = str(kwargs.get("req_id", "unknown"))
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", request_id)[:120]
        stem = f"pid{os.getpid()}_{safe_id}_{uuid.uuid4().hex}"
        profiler = cProfile.Profile()
        started = time.perf_counter()
        success = False
        profiler.enable()
        try:
            result = retrieve(*args, **kwargs)
            success = True
            return result
        finally:
            profiler.disable()
            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                directory.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(str(directory / f"{stem}.pstats"))
                (directory / f"{stem}.json").write_text(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "pid": os.getpid(),
                            "elapsed_ms_instrumented": elapsed_ms,
                            "retrieve_succeeded": success,
                            "diagnostic_only": True,
                            "profile": f"{stem}.pstats",
                        }
                    ),
                    encoding="utf-8",
                )
            except OSError as error:
                print(f"RETRIEVE_CPU_DIAGNOSTIC_WRITE_FAILED {error}", flush=True)

    return profiled
