# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact, score-synchronized K-partitioned prediction benchmark.

This is not the synchronization-free production predictor. Every rank exchanges
its full local Top-K, merges the global Top-K, then retains its owned IDs for
the existing SSD/KV transport. Empty shards still participate in collectives.
"""

# Standard
from typing import Any, Callable

# Third Party
import torch

_ANNOUNCED = False


def pack_candidates(
    logits: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    width: int,
    rank: int,
    world_size: int,
    interleave_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Pack exact local candidate scores and global IDs into one int64 tensor.

    Args:
        logits: Finite valid scores in a two-dimensional local K score matrix.
        indices: Local Top-K indices; invalid entries may be -1.
        starts: Inclusive per-query local causal start.
        ends: Exclusive per-query local causal end.
        width: Fixed global K and communication width, not K/world_size.
        rank: Rank owning this block-cyclic K shard.
        world_size: Number of K shards.
        interleave_blocks: Consecutive physical K blocks per owner.
        block_size: Indexer K entries per block.

    Returns:
        Contiguous [queries, width] packets containing int32 IDs and exact
        float32 score bits. Missing candidates have ID -1 and score -inf.

    Raises:
        ValueError: Invalid shapes, partition geometry, or non-float32 scores.
    """
    if (
        logits.ndim != 2
        or indices.ndim != 2
        or logits.dtype != torch.float32
        or logits.shape[0] != indices.shape[0]
        or indices.shape[1] > width
        or starts.numel() != indices.shape[0]
        or ends.numel() != indices.shape[0]
    ):
        raise ValueError("invalid local score/candidate shapes")
    if width <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid K or rank")
    if block_size <= 0 or interleave_blocks <= 0:
        raise ValueError("invalid block partition")
    # ((int64(-1) << 32) | uint32(float32(-inf))) == -8388608.
    packet = torch.full(
        (indices.shape[0], width), -8388608, dtype=torch.int64, device=logits.device
    )
    if indices.shape[1] == 0 or logits.shape[1] == 0:
        return packet
    local = indices.to(torch.int64)
    valid = (
        (local >= starts.reshape(-1, 1))
        & (local < ends.reshape(-1, 1))
        & (local >= 0)
        & (local < logits.shape[1])
    )
    scores = logits.gather(1, local.clamp(0, logits.shape[1] - 1))
    valid &= torch.isfinite(scores)
    block = local // block_size
    global_block = (
        block // interleave_blocks * world_size + rank
    ) * interleave_blocks + block % interleave_blocks
    global_ids = global_block * block_size + local % block_size
    global_ids = torch.where(valid, global_ids, -1)
    scores = torch.where(valid, scores, -float("inf")).contiguous()
    bits = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    packet[:, : indices.shape[1]] = (global_ids << 32) | bits
    return packet


def merge_owned_candidates(
    packets: torch.Tensor,
    *,
    rank: int,
    interleave_blocks: int,
    block_size: int,
    fused_topk: bool = False,
) -> torch.Tensor:
    """Select exact global Top-K and return only IDs owned by this rank.

    Args:
        packets: [world_size, queries, K] int64 packets from pack_candidates.
        rank: Rank retaining its disjoint share of the global selection.
        interleave_blocks: Block-cyclic ownership granularity.
        block_size: Number of indexer K entries per physical block.
        fused_topk: Use the existing vLLM prefill Top-K kernel on CUDA. The
            default torch.topk implementation also permits CPU contract tests.

    Returns:
        [queries, K] int32 global IDs with non-owned/invalid positions set to -1.
        Equal-score boundary ties may choose any valid Top-K, like torch.topk.

    Raises:
        ValueError: Invalid packet dimensions or ownership geometry.
    """
    if packets.ndim != 3 or packets.dtype != torch.int64:
        raise ValueError("expected [world, queries, K] int64 packets")
    world, rows, width = packets.shape
    if not 0 <= rank < world or width <= 0 or min(block_size, interleave_blocks) <= 0:
        raise ValueError("invalid packet geometry")
    flat = packets.permute(1, 0, 2).reshape(rows, world * width)
    scores = (flat & 0xFFFFFFFF).to(torch.int32).contiguous().view(torch.float32)
    ids = flat >> 32
    if fused_topk:
        if not scores.is_cuda:
            raise ValueError("fused Top-K requires CUDA")
        positions = torch.full(
            (rows, width), -1, dtype=torch.int32, device=scores.device
        )
        starts = torch.zeros(rows, dtype=torch.int32, device=scores.device)
        ends = torch.full_like(starts, world * width)
        torch.ops._C.top_k_per_row_prefill(
            scores,
            starts,
            ends,
            positions,
            rows,
            scores.stride(0),
            scores.stride(1),
            width,
        )
        position_ids = positions.to(torch.int64).clamp(0, world * width - 1)
        position_valid = (positions >= 0) & (positions < world * width)
        chosen_values = scores.gather(1, position_ids)
    else:
        chosen = torch.topk(scores, width, dim=1, sorted=False)
        position_ids, chosen_values = chosen.indices, chosen.values
        position_valid = torch.ones_like(position_ids, dtype=torch.bool)
    selected = ids.gather(1, position_ids)
    owner = selected // block_size // interleave_blocks % world
    valid = (
        (selected >= 0)
        & torch.isfinite(chosen_values)
        & (owner == rank)
        & position_valid
    )
    return torch.where(valid, selected, -1).to(torch.int32)


def score_owner_exact_sync(
    indexer_op: Any,
    q_values: torch.Tensor,
    q_scale: torch.Tensor | None,
    weights: torch.Tensor,
    output: torch.Tensor,
    gather_chunk: Any,
    global_ks: torch.Tensor,
    global_ke: torch.Tensor,
    *,
    total_seq_lens: int,
    query_rows: int,
    query_offset: int,
    rank: int,
    world_size: int,
    interleave_size: int,
    topk_tokens: int,
    workspace_slot: int,
    use_fp4: bool,
    fp8_dtype: torch.dtype,
    runtime_info: dict[str, int] | None,
    workspace: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    key_launch_rows: Callable[[int], int],
    query_launch_rows: Callable[[int, int], int],
    query_tile_rows: Callable[[], int],
    max_logits_bytes: Callable[[int], int],
    pad_queries: Callable[[torch.Tensor, int], torch.Tensor],
    pad_bounds: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Score K shards and synchronize exact global Top-K on the model hook.

    Args:
        indexer_op: Read-only target indexer with public K cache and geometry.
        q_values, q_scale: Proxy query values and optional FP4 query scales.
        weights: Indexer head weights for these queries.
        output: Private [query_rows, at least K] output owned by the caller.
        gather_chunk: Official prefill block-table metadata.
        global_ks, global_ke: Official per-query causal bounds.
        total_seq_lens: Total indexer K entries, at most signed int32 maximum.
        query_rows, query_offset: Number and metadata offset of query rows.
        rank, world_size: Exact speculative TP partition identity.
        interleave_size: Block-cyclic K ownership in blocks.
        topk_tokens: Exact global K; every shard contributes up to K candidates.
        workspace_slot: Existing stream-private K workspace identity.
        use_fp4, fp8_dtype: Existing quantized K cache representation.
        runtime_info: Optional auditable call/shape counters.
        workspace: Caller-owned K workspace provider.
        key_launch_rows, query_launch_rows: Existing launch bucket policies.
        query_tile_rows, max_logits_bytes: Existing memory/tile limits.
        pad_queries, pad_bounds: Existing padded launch helpers.

    Returns:
        The supplied output filled with each rank's exact global-Top-K share.

    Raises:
        ValueError: Unsupported geometry or output capacity.
        RuntimeError: TP identity mismatch or insufficient logits memory budget.

    Notes:
        Must be called in identical model-forward order on every TP rank, never
        from asynchronous owner threads. No authoritative indexer state changes.
        The rank-independent tile bound ensures equal collective counts even
        when local shard sizes differ or some shards contain no keys.
    """
    global _ANNOUNCED
    import torch.distributed as dist
    from vllm import _custom_ops as ops, envs
    from vllm.distributed import get_tp_group
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    from lmcache.v1.owner_prefill_metadata import (
        owner_partition_geometry,
        prepare_owner_metadata,
    )

    if not 0 <= total_seq_lens < 2**31 or output.shape[1] < topk_tokens:
        raise ValueError("K-sync ID/output range invalid")
    group = get_tp_group()
    if group.world_size != world_size or group.rank_in_group != rank:
        raise RuntimeError("K-sync TP identity differs")
    if query_rows == 0:
        return output
    block_size = int(indexer_op.k_cache.kv_cache.shape[1])
    table, cu, starts, ends, local_rows = prepare_owner_metadata(
        gather_chunk.block_table,
        global_ks,
        global_ke,
        total_tokens=total_seq_lens,
        rank=rank,
        world_size=world_size,
        interleave_blocks=interleave_size,
        block_size=block_size,
        query_offset=query_offset,
        query_rows=query_rows,
    )
    local_k = min(local_rows, topk_tokens)
    geometries = [
        owner_partition_geometry(
            total_seq_lens, peer, world_size, interleave_size, block_size
        )
        for peer in range(world_size)
    ]
    maximum_rows = max(item[1] for item in geometries)
    maximum_launch = max(topk_tokens, key_launch_rows(maximum_rows))
    tile_cap = query_tile_rows()
    if tile_cap <= 0:
        raise ValueError("exact K-sync requires an explicit uniform query tile")
    tile = min(
        tile_cap,
        max_logits_bytes(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB)
        // (maximum_launch * 4),
    )
    # Launch padding must also fit the budget, not just the logical row count.
    row_budget = max_logits_bytes(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB) // (
        maximum_launch * 4
    )
    while tile > 0 and query_launch_rows(tile, tile_cap) > row_budget:
        tile //= 2
    if tile <= 0:
        raise RuntimeError("K-sync logits budget cannot hold one query row")
    if runtime_info is not None:
        runtime_info.update(
            exact_k_sync=1,
            exact_global_topk_tokens=topk_tokens,
            local_topk_tokens=topk_tokens,
            exact_local_score_candidates=local_k,
            local_key_rows=local_rows,
            local_query_rows=query_rows,
            query_sample_stride=1,
            fused_owner_metadata=1,
            exact_k_sync_collectives=0,
        )
        runtime_info["owner_block_bound"] = max(1, max(item[0] for item in geometries))
    k_cast = scales = None
    if local_rows:
        launch_k = max(topk_tokens, key_launch_rows(local_rows))
        capacity = (
            int(indexer_op.max_total_seq_len) + world_size - 1
        ) // world_size + block_size
        quant, scale = workspace(
            device=output.device,
            total_capacity=max(capacity, launch_k),
            head_dim=int(indexer_op.head_dim),
            use_fp4=use_fp4,
            fp8_dtype=fp8_dtype,
            slot=workspace_slot,
        )
        quant, scale = quant[:launch_k], scale[:launch_k]
        if launch_k > local_rows:
            quant[local_rows:].zero_()
            scale[local_rows:].zero_()
        ops.cp_gather_indexer_k_quant_cache(
            indexer_op.k_cache.kv_cache,
            quant[:local_rows],
            scale[:local_rows],
            table,
            cu,
        )
        k_cast = quant.view(torch.int8) if use_fp4 else quant
        scales = scale.view(torch.int32 if use_fp4 else torch.float32).squeeze(-1)
    for begin in range(0, query_rows, tile):
        end = min(query_rows, begin + tile)
        rows = end - begin
        if local_rows:
            launch_rows = query_launch_rows(rows, tile_cap)
            q = pad_queries(q_values[begin:end], launch_rows)
            qs = (
                pad_queries(q_scale[begin:end], launch_rows)
                if q_scale is not None
                else None
            )
            ks, ke = pad_bounds(
                starts[begin:end], ends[begin:end], launch_rows, local_rows
            )
            logits = fp8_fp4_mqa_logits(
                (q.view(torch.int8) if use_fp4 else q, qs),
                (k_cast, scales),
                pad_queries(weights[begin:end], launch_rows),
                ks,
                ke,
                clean_logits=False,
            )
            selected = torch.full(
                (launch_rows, topk_tokens), -1, dtype=output.dtype, device=output.device
            )
            torch.ops._C.top_k_per_row_prefill(
                logits,
                ks,
                ke,
                selected,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            packet = pack_candidates(
                logits[:rows],
                selected[:rows],
                ks[:rows],
                ke[:rows],
                width=topk_tokens,
                rank=rank,
                world_size=world_size,
                interleave_blocks=interleave_size,
                block_size=block_size,
            )
        else:
            packet = torch.full(
                (rows, topk_tokens), -8388608, dtype=torch.int64, device=output.device
            )
        received = torch.empty(
            (world_size * rows, topk_tokens), dtype=packet.dtype, device=packet.device
        )
        dist.all_gather_into_tensor(received, packet, group=group.device_group)
        merged = merge_owned_candidates(
            received.view(world_size, rows, topk_tokens),
            rank=rank,
            interleave_blocks=interleave_size,
            block_size=block_size,
            fused_topk=True,
        )
        output[begin:end, :topk_tokens].copy_(merged)
        if runtime_info is not None:
            runtime_info["exact_k_sync_collectives"] += 1
    if not _ANNOUNCED:
        from lmcache.logging import init_logger

        init_logger(__name__).info(
            "EXACT_K_SYNC_ACTIVE rank=%d world=%d global_k=%d "
            "local_candidate_cap=%d collectives=%d",
            rank,
            world_size,
            topk_tokens,
            local_k,
            (query_rows + tile - 1) // tile,
        )
        _ANNOUNCED = True
    return output
