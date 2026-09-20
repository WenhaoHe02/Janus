# SPDX-License-Identifier: Apache-2.0
"""Fused routing for separate owner-ID and KV AllGather buffers."""

from __future__ import annotations

# Third Party
import torch
import triton
import triton.language as tl


@triton.jit
def _claim_rows(
    ids: tl.tensor,
    claims: tl.tensor,
    ready: tl.tensor,
    logical_limit: tl.tensor,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = pos // WIDTH
    offset = pos % WIDTH
    count = tl.load(ids + rank * (WIDTH + 1), pos < ROWS, other=0)
    block = tl.load(ids + rank * (WIDTH + 1) + 1 + offset, pos < ROWS, other=-1)
    advertised = (pos < ROWS) & (offset < count) & (block >= 0)
    valid = advertised & (block < logical_limit)
    tl.atomic_min(claims + block, pos.to(tl.int32), mask=valid, sem="relaxed")
    invalid = ((pos < ROWS) & (count < 0)) | (advertised & ~valid)
    tl.atomic_and(ready, tl.min((~invalid).to(tl.int32), 0), sem="relaxed")


@triton.jit
def _scatter_rows(
    ids: tl.tensor,
    receive: tl.tensor,
    claims: tl.tensor,
    row_map: tl.tensor,
    destination: tl.tensor,
    resident: tl.tensor,
    selected: tl.tensor,
    ready: tl.tensor,
    logical_limit: tl.tensor,
    destination_limit: tl.tensor,
    receive_stride: tl.tensor,
    destination_stride: tl.tensor,
    WIDTH: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pos = tl.program_id(0)
    tile = tl.program_id(1)
    rank = pos // WIDTH
    offset = pos % WIDTH
    count = tl.load(ids + rank * (WIDTH + 1))
    block = tl.load(ids + rank * (WIDTH + 1) + 1 + offset)
    valid = (offset < count) & (block >= 0) & (block < logical_limit)
    owner = tl.load(claims + block, mask=valid, other=-1)
    valid = valid & (owner == pos)
    physical = tl.load(row_map + block, mask=valid, other=-1)
    mapped = valid & (physical >= 0) & (physical < destination_limit)
    if tile == 0:
        tl.store(selected + pos, tl.where(mapped, block, -1))
        tl.atomic_and(ready, (~(valid & ~mapped)).to(tl.int32), sem="relaxed")
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        receive + pos.to(tl.int64) * receive_stride + offsets,
        mask=mapped & (offsets < ROW_BYTES),
        other=0,
    )
    tl.store(
        destination + physical.to(tl.int64) * destination_stride + offsets,
        values,
        mask=mapped & (offsets < ROW_BYTES),
    )
    # Consumers only see this bitmap after the entire scatter kernel's event.
    if tile == 0:
        tl.store(resident + block, 1, mask=mapped)


def scatter_owner_sidecars(
    gathered_ids: torch.Tensor,
    receive: torch.Tensor,
    logical_rows: torch.Tensor,
    destination: torch.Tensor,
    resident: torch.Tensor,
    *,
    world_size: int,
    padded_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route byte-exact gathered KV without sorting or host scalar reads.

    Args:
        gathered_ids: Contiguous int64 rank-major ``[count, IDs...]`` sidecars.
        receive: uint8 gathered rows, including each rank's padding.
        logical_rows: Contiguous int64 logical-to-physical destination map.
        destination: uint8 KV rows, allowing a strided outer dimension.
        resident: Contiguous bool logical residency bitmap, updated in place.
        world_size: Number of rank payloads in the inputs.
        padded_blocks: Fixed per-rank payload capacity.

    Returns:
        Fixed-size rank-major selected IDs (invalid/duplicate slots are -1),
        and scalar int32 readiness. First rank-major occurrence wins, including
        duplicate IDs with different bytes. Failed ranks publish no blocks.
        Readiness is zero for failed ranks or invalid advertised destinations.

    Raises:
        ValueError: If input devices, dtypes, sizes or strides are unsupported.

    Notes:
        Caller must order the current stream after BOTH collectives and retain
        their buffers until this call's work finishes. Consumers must wait on
        an event recorded AFTER this call before reading KV or residency.
        Temporary claims are stream-local allocator storage, not a shared cache.
    """
    rows = world_size * padded_blocks
    tensors = (gathered_ids, receive, logical_rows, destination, resident)
    if (
        world_size <= 0
        or padded_blocks <= 0
        or any(t.device != destination.device or not t.is_cuda for t in tensors)
        or gathered_ids.dtype != torch.int64
        or logical_rows.dtype != torch.int64
        or resident.dtype != torch.bool
        or receive.dtype != torch.uint8
        or destination.dtype != torch.uint8
        or receive.ndim != 2
        or destination.ndim != 2
        or gathered_ids.numel() != world_size * (padded_blocks + 1)
        or receive.shape[0] != rows
        or receive.shape[1] != destination.shape[1]
        or receive.stride(1) != 1
        or destination.stride(1) != 1
        or not all(t.is_contiguous() for t in (gathered_ids, logical_rows, resident))
        or min(logical_rows.numel(), resident.numel()) <= 0
    ):
        raise ValueError("unsupported owner-sidecar routing geometry")
    limit = min(logical_rows.numel(), resident.numel())
    claims = torch.full((limit,), rows, dtype=torch.int32, device=destination.device)
    ready = torch.ones((), dtype=torch.int32, device=destination.device)
    selected = torch.empty(rows, dtype=torch.int64, device=destination.device)
    _claim_rows[(triton.cdiv(rows, 256),)](
        gathered_ids, claims, ready, limit, ROWS=rows, WIDTH=padded_blocks, BLOCK=256
    )
    _scatter_rows[(rows, triton.cdiv(destination.shape[1], 2048))](
        gathered_ids,
        receive,
        claims,
        logical_rows,
        destination,
        resident,
        selected,
        ready,
        limit,
        destination.shape[0],
        receive.stride(0),
        destination.stride(0),
        WIDTH=padded_blocks,
        ROW_BYTES=destination.shape[1],
        BLOCK=2048,
    )
    return selected, ready
