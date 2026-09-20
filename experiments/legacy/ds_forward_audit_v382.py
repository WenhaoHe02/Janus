# SPDX-License-Identifier: Apache-2.0
"""Opt-in DS continuation audit; never enables prediction or submits reads."""

# Standard
import json
from pathlib import Path
from typing import Any

_CURRENT: dict[str, Any] = {}
_ORDINAL = 0


def scheduler_snapshot(output: Any) -> dict[str, dict[str, Any]]:
    """Describe each scheduled request using vLLM's context-phase API.

    Args:
        output: SchedulerOutput with new/cached requests and scheduled counts.
    Returns:
        Request IDs mapped to exact phase and scheduled query rows.
    Raises:
        RuntimeError: Duplicate identities or incomplete scheduler coverage.
    """
    phases = {}
    for request in output.scheduled_new_reqs:
        phases[request.req_id] = (
            "prefill"
            if request.num_computed_tokens < len(request.prompt_token_ids)
            else "decode"
        )
    cached = output.scheduled_cached_reqs
    for request_id in cached.req_ids:
        if request_id in phases:
            raise RuntimeError("duplicate scheduled request")
        phases[request_id] = (
            "prefill" if cached.is_context_phase(request_id) else "decode"
        )
    if set(phases) != set(output.num_scheduled_tokens):
        raise RuntimeError("incomplete scheduler phase coverage")
    return {
        key: {"phase": phase, "query_rows": int(output.num_scheduled_tokens[key])}
        for key, phase in phases.items()
    }


def observe(snapshot: dict[str, dict[str, Any]]) -> None:
    """Replace worker-local phase state; retain only the current request.

    Args:
        snapshot: Exact scheduler metadata for this forward.
    Raises:
        ValueError: Unsupported phase or nonpositive scheduled query rows.
    """
    global _CURRENT, _ORDINAL
    _ORDINAL += 1
    _CURRENT = {}
    if len(snapshot) != 1:
        return
    request, values = next(iter(snapshot.items()))
    if values["phase"] not in ("prefill", "decode") or values["query_rows"] <= 0:
        raise ValueError("invalid scheduler phase/count")
    _CURRENT = dict(values, request_id=request, forward_ordinal=_ORDINAL)


def record_topk(manager: Any, layer: int, topk: Any, *, stage: str) -> None:
    """Record true history selection and missing residency at attention entry.

    Args:
        manager: CSA manager exposing active_request_id/profile_topk_residency.
        layer: Physical transformer layer.
        topk: Official indexer output, including any unused workspace tail.
        stage: ``before_correction``, ``after_correction`` or ``inactive_bypass``.
    Raises:
        RuntimeError: Insufficient output rows or a changed active request.
    Notes:
        Synchronizes for diagnostics only. It does not change gates, cache
        contents, prediction sets or pending reads. Missing means absent from
        the manager's GPU residency bitmap, not physical SSD bytes measured.
        Ignore old plans and cold requests by exact scheduler identity.
    """
    if not _CURRENT or manager.active_request_id != _CURRENT["request_id"]:
        return
    if stage not in ("before_correction", "after_correction", "inactive_bypass"):
        raise ValueError("unknown audit stage")
    count = _CURRENT["query_rows"]
    if topk.ndim == 1:
        if count != 1:
            raise RuntimeError("one-row top-K for a multi-row scheduler forward")
        active = topk
    else:
        if topk.shape[0] < count:
            raise RuntimeError("official top-K smaller than scheduled query count")
        active = topk[:count]
    selected, missing = manager.profile_topk_residency(layer, active)
    if manager.active_request_id != _CURRENT["request_id"]:
        raise RuntimeError("request changed during residency audit")
    # Third Party
    import torch

    request = _CURRENT["request_id"]
    if not request or any(
        not (c.isascii() and (c.isalnum() or c in "-_")) for c in request
    ):
        raise ValueError("unsafe request identifier")
    rank = int(torch.distributed.get_rank())
    row = dict(
        _CURRENT,
        layer=int(layer),
        rank=rank,
        stage=stage,
        selected=selected.tolist(),
        missing=missing.tolist(),
        authoritative_columns=int(topk.shape[-1]),
        physical_read_measurement=False,
    )
    directory = Path("/tmp/dsv4_forward_audit_v382") / request
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank{rank}.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, separators=(",", ":")) + "\n")
