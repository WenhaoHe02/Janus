# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only matched GLM decoder discovery bypass with CPU timings."""

# Standard
from functools import wraps
import json
import os
from pathlib import Path
import time
from typing import Any, Callable


def wrap_discovery(
    original: Callable[[], list[Any]],
    model_type: Callable[[], str | None],
    directory: Path,
) -> Callable[[], list[Any]]:
    """Time original discovery, or bypass it for GLM in explicit fixed mode.

    Args:
        original: Unmodified DS layer-discovery function.
        model_type: Loaded model type captured from connector configuration.
        directory: Diagnostic output; mode file must contain baseline or fixed.
    Returns:
        A function preserving DS and unknown-model discovery in both modes.
    Raises:
        ValueError: Invalid mode; never silently change a diagnostic treatment.
    """
    @wraps(original)
    def measured() -> list[Any]:
        mode = (directory / "mode").read_text().strip()
        if mode not in ("baseline", "fixed"):
            raise ValueError("Expected baseline/fixed discovery mode")
        wall = time.time_ns()
        start = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        bypass = mode == "fixed" and model_type() == "glm_moe_dsa"
        result = [] if bypass else original()
        row = dict(wall_ns=wall, pid=os.getpid(), mode=mode, bypass=bypass,
                   model_type=model_type(), layers=len(result),
                   elapsed_ms=(time.perf_counter_ns() - start) / 1e6,
                   cpu_ms=(time.thread_time_ns() - cpu) / 1e6)
        with (directory / f"discovery_{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return result
    return measured
