# SPDX-License-Identifier: Apache-2.0
"""Experiment-only request phase from scheduler state, not attention shape."""

# Standard
from typing import Any

_PHASES: dict[str, str] = {}


def scheduler_phases(output: Any) -> dict[str, str]:
    """Return scheduled request phases using vLLM's public context-phase API.

    Args:
        output: vLLM 0.26 SchedulerOutput with new/cached request data.
    Returns:
        Request IDs mapped to prefill/decode, including save-skipped requests.
    Raises:
        RuntimeError: If a scheduled request is absent or duplicated.
    """
    phases: dict[str, str] = {}
    for request in output.scheduled_new_reqs:
        phases[request.req_id] = (
            "prefill"
            if request.num_computed_tokens < len(request.prompt_token_ids)
            else "decode"
        )
    cached = output.scheduled_cached_reqs
    for request_id in cached.req_ids:
        if request_id in phases:
            raise RuntimeError("duplicate scheduled request phase")
        phases[request_id] = (
            "prefill" if cached.is_context_phase(request_id) else "decode"
        )
    if set(phases) != set(output.num_scheduled_tokens):
        raise RuntimeError("scheduled request phase coverage mismatch")
    return phases


def observe(phases: dict[str, str]) -> None:
    """Replace one forward's phase snapshot without retaining past requests."""
    global _PHASES
    if any(value not in ("prefill", "decode") for value in phases.values()):
        raise ValueError("unknown experiment request phase")
    _PHASES = dict(phases)


def is_prefill() -> bool:
    """Return true only for a single-request prefill (including later chunks)."""
    return len(_PHASES) == 1 and next(iter(_PHASES.values())) == "prefill"


def is_decode() -> bool:
    """Return true only for a single-request, scheduler-confirmed decode."""
    return len(_PHASES) == 1 and next(iter(_PHASES.values())) == "decode"


def single_request_id() -> str:
    """Return the exact scheduler identity, or empty for idle/mixed batches."""
    return next(iter(_PHASES)) if len(_PHASES) == 1 else ""
