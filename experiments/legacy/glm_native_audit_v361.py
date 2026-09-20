# SPDX-License-Identifier: Apache-2.0
"""Diagnostic samples from the actual GLM attention consumer, not loader choices.

GPU snapshots are queued on the model stream; CPU transfer happens only after
the forward. This is instrumentation, not a correctness fix or timing run.
"""

# Standard
import os
from pathlib import Path
from typing import Any

# Third Party
import torch

_context: dict[str, Any] | None = None


def begin(requests: list[Any]) -> None:
    """Record the adapter's mapping independently of the native attention table."""
    global _context
    _context = None
    if os.getenv("LMCACHE_GLM_NATIVE_AUDIT", "0") != "1":
        return
    from lmcache.integration.vllm.glm_forward_phase import is_prefill
    if not is_prefill() or len(requests) != 1:
        return
    request = requests[0]
    if not request.is_last_prefill:
        return
    history = int(os.environ.get("LMCACHE_GLM_NATIVE_AUDIT_HISTORY", "76288"))
    if len(request.token_ids) < history:
        return
    _context = {
        "request_id": str(request.req_id),
        "history": history,
        "prompt_len": request.prompt_len,
        "cached_tokens": getattr(request.load_spec, "lmcache_cached_tokens", 0),
        "adapter_rows": request.slot_mapping[:history:64].cpu().clone() // 64,
        "layers": [],
    }


def capture(
    layer: Any, q: Any, cache: torch.Tensor,
    indices: torch.Tensor, metadata: Any,
) -> dict[str, Any] | None:
    """Snapshot Q, true indices, native destinations and sampled KV before MLA.

    Args:
        layer: Actual vLLM attention layer, used only for its public name.
        q: Query tensor or its tuple of components.
        cache: Actual KV allocation passed to the attention kernel.
        indices: Authoritative logical top-K buffer consumed by this layer.
        metadata: Actual native attention metadata, including its block table.
    Returns:
        A ticket to attach output samples, or None when not auditing prefill.
    Raises:
        ValueError: The single-request GLM geometry is not the expected layout.
    """
    if _context is None:
        return None
    if cache.ndim != 3 or cache.shape[1] != 64 or metadata.block_table.shape[0] != 1:
        raise ValueError("native audit requires one GLM request with block size 64")
    components = q if isinstance(q, tuple) else (q,)
    count = int(components[0].shape[0])
    picks = sorted({0, max(0, count // 2), max(0, count - 1)})
    q_rows = torch.tensor(picks, device=cache.device, dtype=torch.long)
    table = metadata.block_table[0, :_context["history"] // 64].long().clone()
    raw = cache.view(torch.uint8).reshape(cache.shape[0], -1)
    valid_rows = (table >= 0) & (table < raw.shape[0])
    safe_rows = table.clamp(0, raw.shape[0] - 1)
    offsets = torch.linspace(0, raw.shape[1] - 1, 64, device=cache.device).round().long()
    ticket = {
        "layer_name": str(getattr(layer, "layer_name", getattr(layer, "prefix", "unknown"))),
        "query_rows": count, "query_picks": picks,
        "native_rows": table,
        "native_rows_valid": valid_rows,
        "native_slot_mapping": metadata.slot_mapping.clone() if hasattr(metadata, "slot_mapping") else None,
        "cache_shape": list(cache.shape), "cache_stride": list(cache.stride()),
        "cache_ptr": cache.data_ptr(), "byte_offsets": offsets,
        "kv_samples": raw[safe_rows[:, None], offsets[None, :]],
        "q_samples": [part.index_select(0, q_rows).clone() for part in components],
        "topk_samples": indices[:count].index_select(0, q_rows).clone(),
        "mixed_fp8": bool(metadata.fp8_use_mixed_batch),
        "num_prefills": int(metadata.num_prefills),
        "num_decodes": int(metadata.num_decodes),
    }
    _context["layers"].append(ticket)
    return ticket


def finish(ticket: dict[str, Any] | None, output: torch.Tensor) -> None:
    """Attach model-stream output samples without blocking the CPU."""
    if ticket is not None:
        picks = torch.tensor(ticket["query_picks"], device=output.device, dtype=torch.long)
        ticket["output_samples"] = output.index_select(0, picks).clone()


def to_cpu(value: Any) -> Any:
    """Materialize only retained diagnostic tensors, preserving their structure."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    return value


def flush() -> None:
    """Write the completed native-consumer snapshots after forward consumption."""
    global _context
    record = _context
    _context = None
    if record is None:
        return
    request_id = record["request_id"]
    if not request_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in request_id):
        raise ValueError("invalid native-audit request id")
    from lmcache.v1.csa_attention_kv_prefetch_manager import get_csa_attention_kv_prefetch_manager
    manager = get_csa_attention_kv_prefetch_manager()
    if manager is not None and manager.active_request_id == request_id:
        record["loader_plan"] = manager.snapshot_request_read_plan()
    rank = torch.distributed.get_rank()
    destination = Path("/tmp/glm_native_audit_v361") / request_id / f"rank{rank}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record["rank"] = rank
    record["sampled_not_full_bytes"] = True
    torch.save(to_cpu(record), destination)
    print(f"GLM_NATIVE_AUDIT_V361 request={request_id} rank={rank} layers={len(record['layers'])} path={destination}", flush=True)
