# SPDX-License-Identifier: Apache-2.0
"""Exact cyclic/contiguous K metadata and top-K maps without host readbacks."""

from __future__ import annotations

# Third Party
import torch
import triton
import triton.language as tl


def owner_partition_geometry(
    total_tokens: int,
    rank: int,
    world_size: int,
    interleave_blocks: int,
    block_size: int,
) -> tuple[int, int]:
    """Return local block/valid-row counts for block-cyclic K ownership.

    Args:
        total_tokens: Valid K entries, not original uncompressed model tokens.
        rank: Rank within the partition.
        world_size: Number of partition ranks.
        interleave_blocks: Consecutive blocks owned before advancing rank.
        block_size: K entries in a physical cache block.

    Returns:
        Local block count and local valid-entry count, including a partial tail.

    Raises:
        ValueError: If lengths or partition geometry are invalid.
    """
    if total_tokens < 0 or block_size <= 0 or interleave_blocks <= 0:
        raise ValueError("invalid owner K length or block geometry")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid owner rank")
    blocks = (total_tokens + block_size - 1) // block_size
    cycles, tail = divmod(blocks, world_size * interleave_blocks)
    local_blocks = cycles * interleave_blocks + max(
        0, min(interleave_blocks, tail - rank * interleave_blocks)
    )
    rows = local_blocks * block_size
    if blocks and (blocks - 1) // interleave_blocks % world_size == rank:
        rows -= (block_size - total_tokens % block_size) % block_size
    return local_blocks, rows


def contiguous_owner_geometry(
    total_tokens: int,
    rank: int,
    world_size: int,
    block_size: int,
) -> tuple[int, int, int]:
    """Return first block, block count and valid rows of a balanced K shard.

    Args:
        total_tokens: Valid logical K entries (not uncompressed query tokens).
        rank: Rank receiving one contiguous, non-overlapping block range.
        world_size: Number of participating ranks.
        block_size: K entries stored in a physical block.

    Returns:
        First logical block, number of blocks and valid K rows. Lowest ranks
        receive remainder blocks; all ranks differ by at most one block.

    Raises:
        ValueError: Invalid lengths, block size, world size or rank.
    """
    if total_tokens < 0 or block_size <= 0:
        raise ValueError("invalid contiguous owner geometry")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid contiguous owner rank")
    blocks = (total_tokens + block_size - 1) // block_size
    width, remainder = divmod(blocks, world_size)
    first = rank * width + min(rank, remainder)
    count = width + int(rank < remainder)
    rows = max(0, min(count * block_size, total_tokens - first * block_size))
    return first, count, rows


@triton.jit
def _prepare_metadata(
    block_table: tl.tensor,
    global_start: tl.tensor,
    global_end: tl.tensor,
    local_table: tl.tensor,
    cu_seqlens: tl.tensor,
    local_start: tl.tensor,
    local_end: tl.tensor,
    query_offset: tl.tensor,
    query_rows: tl.tensor,
    local_blocks: tl.tensor,
    local_rows: tl.tensor,
    table_stride: tl.tensor,
    start_stride: tl.tensor,
    end_stride: tl.tensor,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    FIRST_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
) -> None:
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    if CONTIGUOUS:
        logical = FIRST_BLOCK + i
    else:
        logical = (i // INTERLEAVE * WORLD + RANK) * INTERLEAVE + i % INTERLEAVE
    physical = tl.load(block_table + logical * table_stride, i < local_blocks, other=0)
    tl.store(local_table + i, physical, i < local_blocks)
    tl.store(cu_seqlens + i, tl.where(i == 0, 0, local_rows), i < 2)
    start = tl.maximum(
        tl.load(
            global_start + (query_offset + i) * start_stride, i < query_rows, other=0
        ),
        0,
    )
    end = tl.maximum(
        tl.load(global_end + (query_offset + i) * end_stride, i < query_rows, other=0),
        0,
    )
    if CONTIGUOUS:
        ks = tl.maximum(start - FIRST_BLOCK * BLOCK_SIZE, 0)
        ke = tl.maximum(end - FIRST_BLOCK * BLOCK_SIZE, 0)
    else:
        stripe = INTERLEAVE * BLOCK_SIZE
        period = WORLD * stripe
        ks = start // period * stripe + tl.minimum(
            tl.maximum(start % period - RANK * stripe, 0), stripe
        )
        ke = end // period * stripe + tl.minimum(
            tl.maximum(end % period - RANK * stripe, 0), stripe
        )
    tl.store(local_start + i, tl.minimum(ks, local_rows), i < query_rows)
    tl.store(local_end + i, tl.minimum(ke, local_rows), i < query_rows)


@triton.jit
def _globalize_topk(
    local_topk: tl.tensor,
    output: tl.tensor,
    local_rows: tl.tensor,
    n_entries: tl.tensor,
    output_row_offset: tl.tensor,
    output_stride: tl.tensor,
    TOPK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    STRIPE: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    FIRST_KEY: tl.constexpr,
    TILE: tl.constexpr,
) -> None:
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    entry = tl.load(local_topk + i, i < n_entries, other=-1).to(tl.int64)
    valid = (entry >= 0) & (entry < local_rows)
    if CONTIGUOUS:
        mapped = FIRST_KEY + entry
    else:
        mapped = (entry // STRIPE * WORLD + RANK) * STRIPE + entry % STRIPE
    destination = (output_row_offset + i // TOPK) * output_stride + i % TOPK
    tl.store(output + destination, tl.where(valid, mapped, -1), i < n_entries)


def prepare_owner_metadata(
    block_table: torch.Tensor,
    global_start: torch.Tensor,
    global_end: torch.Tensor,
    *,
    total_tokens: int,
    rank: int,
    world_size: int,
    interleave_blocks: int,
    block_size: int,
    query_offset: int,
    query_rows: int,
    contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build local block table, sequence length and causal bounds in one kernel.

    Args:
        block_table: CUDA int32/int64 [1, blocks] logical-to-physical K map.
        global_start: CUDA int32/int64 one-dimensional inclusive K starts.
        global_end: CUDA int32/int64 one-dimensional exclusive K ends.
        total_tokens: Valid K entries across all ranks.
        rank: Rank within the partition.
        world_size: Partition world size.
        interleave_blocks: Consecutive cache blocks owned by a rank.
        block_size: Entries per block.
        query_offset: First query index in the global bound vectors.
        query_rows: Number of consecutive queries to prepare.
        contiguous: Use the balanced contiguous block partition instead of
            block-cyclic ownership; preserves the existing contiguous mode.

    Returns:
        Local block table, [0, local_rows], local K starts/ends, local row count.
        Bounds exactly equal searchsorted over the rank's sorted global K IDs.

    Raises:
        ValueError: If tensor geometry or partition/query parameters are invalid.

    Notes:
        No CPU tensor-value read, collective, prediction skip or CUDA Graph.
        Reads current block-map values on every call, so no stale map cache.
    """
    n_blocks, n_rows = owner_partition_geometry(
        total_tokens, rank, world_size, interleave_blocks, block_size
    )
    first_block = 0
    if contiguous:
        first_block, n_blocks, n_rows = contiguous_owner_geometry(
            total_tokens, rank, world_size, block_size
        )
    if (
        block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] < (total_tokens + block_size - 1) // block_size
        or global_start.ndim != 1
        or global_end.ndim != 1
        or query_offset < 0
        or query_rows < 0
        or query_offset + query_rows > min(global_start.numel(), global_end.numel())
        or any(
            t.device != block_table.device
            or not t.is_cuda
            or t.dtype not in (torch.int32, torch.int64)
            for t in (block_table, global_start, global_end)
        )
    ):
        raise ValueError("invalid owner metadata tensors/query range")
    table = torch.empty(
        (1, n_blocks), dtype=block_table.dtype, device=block_table.device
    )
    cu = torch.empty(2, dtype=torch.int32, device=block_table.device)
    starts = torch.empty(
        query_rows, dtype=global_start.dtype, device=block_table.device
    )
    ends = torch.empty(query_rows, dtype=global_end.dtype, device=block_table.device)
    _prepare_metadata[(triton.cdiv(max(n_blocks, query_rows, 2), 256),)](
        block_table,
        global_start,
        global_end,
        table,
        cu,
        starts,
        ends,
        query_offset,
        query_rows,
        n_blocks,
        n_rows,
        block_table.stride(1),
        global_start.stride(0),
        global_end.stride(0),
        RANK=rank,
        WORLD=world_size,
        INTERLEAVE=interleave_blocks,
        BLOCK_SIZE=block_size,
        CONTIGUOUS=contiguous,
        FIRST_BLOCK=first_block,
        TILE=256,
    )
    return table, cu, starts, ends, n_rows


def globalize_owner_topk_into(
    local_topk: torch.Tensor,
    output: torch.Tensor,
    *,
    local_rows: int,
    rank: int,
    world_size: int,
    interleave_blocks: int,
    block_size: int,
    output_row_offset: int = 0,
    first_key: int | None = None,
) -> None:
    """Write local top-K as global IDs, preserving order and -1 invalid entries.

    Args:
        local_topk: Contiguous CUDA int32/int64 [queries, k] native top-K result.
        output: CUDA output [queries, width], possibly wider than k.
        local_rows: Number of valid local K entries.
        rank: Partition rank.
        world_size: Partition world size.
        interleave_blocks: Consecutive owned blocks per rank.
        block_size: Entries per block.
        output_row_offset: Output query row where this slice begins.
        first_key: First global K row for a contiguous shard. None retains
            the original block-cyclic mapping.

    Returns:
        None; writes are ordered on the caller's current stream.

    Raises:
        ValueError: If geometry, types, devices or output bounds are invalid.
    """
    owner_partition_geometry(0, rank, world_size, interleave_blocks, block_size)
    if (
        local_rows < 0
        or (first_key is not None and first_key < 0)
        or local_topk.ndim != 2
        or output.ndim != 2
        or not local_topk.is_contiguous()
        or output.stride(1) != 1
        or local_topk.dtype != output.dtype
        or local_topk.dtype not in (torch.int32, torch.int64)
        or not local_topk.is_cuda
        or output.device != local_topk.device
        or output_row_offset < 0
        or output_row_offset + local_topk.shape[0] > output.shape[0]
        or local_topk.shape[1] > output.shape[1]
    ):
        raise ValueError("invalid owner top-K mapping tensors")
    if local_topk.numel() == 0:
        return
    _globalize_topk[(triton.cdiv(local_topk.numel(), 256),)](
        local_topk,
        output,
        local_rows,
        local_topk.numel(),
        output_row_offset,
        output.stride(0),
        TOPK=local_topk.shape[1],
        RANK=rank,
        WORLD=world_size,
        STRIPE=interleave_blocks * block_size,
        CONTIGUOUS=first_key is not None,
        FIRST_KEY=0 if first_key is None else first_key,
        TILE=256,
    )
