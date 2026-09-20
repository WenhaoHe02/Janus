# SPDX-License-Identifier: Apache-2.0
"""CPU launch scopes independent of switches that disable device-submit paths."""

# Standard
from functools import wraps
import inspect
import os
from typing import Any, Callable

# Third Party
import torch

ENABLED = os.getenv("LMCACHE_EXPERIMENT_NVTX", "0") == "1"


def trace_call(label: str) -> Callable:
    """Annotate CPU execution/enqueue, never claim it is GPU or SSD duration.

    Args:
        label: Stable component name used to join CUDA launch correlations.

    Returns:
        A decorator that returns the original callable when instrumentation is off.
    """

    def decorate(function: Callable) -> Callable:
        if not ENABLED:
            return function
        parameters = list(inspect.signature(function).parameters)
        fields = {
            name: parameters.index(name)
            for name in (
                "layer_id",
                "target_layer_id",
                "source_layer_id",
                "target_layer",
                "source_layer",
            )
            if name in parameters
        }

        @wraps(function)
        def scoped(*args: Any, **kwargs: Any) -> Any:
            parts = [f"experiment={label}"]
            for name, position in fields.items():
                value = kwargs.get(
                    name, args[position] if position < len(args) else None
                )
                if isinstance(value, int):
                    parts.append(f"{name}={value}")
            # Read CPU metadata only; never call item()/cpu() on tensor inputs.
            for name in ("state", "_state", "event", "self"):
                if name not in parameters:
                    continue
                position = parameters.index(name)
                value = kwargs.get(
                    name, args[position] if position < len(args) else None
                )
                if name == "event":
                    schedule = getattr(value, "schedule", None)
                    for field in ("source_layer", "target_layer"):
                        layer = getattr(schedule, field, None)
                        if isinstance(layer, int):
                            parts.append(f"{field}={layer}")
                else:
                    layer = getattr(value, "layer_id", None)
                    if not isinstance(layer, int):
                        layer = getattr(value, "layer_idx", None)
                    if isinstance(layer, int):
                        parts.append(f"physical_layer={layer}")
            torch.cuda.nvtx.range_push("|".join(parts))
            try:
                return function(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        return scoped

    return decorate
