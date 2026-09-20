# SPDX-License-Identifier: Apache-2.0
"""Scheduler-derived phases for continuing an existing external-KV request."""

# Standard
from typing import Any


def scheduled_request_phases(output: Any) -> dict[str, str]:
    """Get request phases without confusing short prefill with decode kernels.

    Args:
        output: vLLM scheduler output with new and cached request data.
    Returns:
        Scheduled request IDs mapped to ``prefill`` or ``decode``.
    Raises:
        RuntimeError: Duplicated identities or incomplete phase metadata.
    """
    phases: dict[str, str] = {}
    for request in output.scheduled_new_reqs:
        if request.req_id in phases:
            raise RuntimeError("duplicate scheduled request phase")
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
        raise RuntimeError("incomplete scheduled request phase coverage")
    return phases


def resume_external_kv_request(
    manager: Any, phases: dict[str, str], *, has_external_load: bool
) -> bool:
    """Resume demand correction only for the active request's later forward.

    Args:
        manager: CSA manager, or None if this model does not use CSA.
        phases: Current scheduler phases, including save-skipped requests.
        has_external_load: True when this forward already enters normal retrieve.
    Returns:
        True if the retained single-request SSD plan was resumed.
    Raises:
        RuntimeError: An active external-KV request is in a mixed batch; the
            current manager owns only one request's slot mapping.
    Notes:
        No prediction is scheduled and no GPU operation is issued here. Cold
        requests and different request identities never inherit a prior plan.
    """
    if manager is None or has_external_load:
        return False
    request = manager.active_request_id
    if not request or request not in phases:
        return False
    if len(phases) != 1:
        raise RuntimeError("external-KV continuation requires a single-request batch")
    manager.set_kv_forward_phase(
        "continuation" if phases[request] == "prefill" else phases[request],
        request_id=request,
    )
    return True
