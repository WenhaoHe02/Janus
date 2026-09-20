# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape owner ID+KV packets for a single ordered AllGather."""

from __future__ import annotations

# Third Party
import torch
import triton
import triton.language as tl


_OWNER_PACKET_ALIGNMENT = 256


def owner_packet_layout(padded_blocks: int, row_bytes: int) -> tuple[int, int]:
    """Return the aligned sidecar offset and total bytes for one rank packet.

    Args:
        padded_blocks: Fixed number of rows advertised by one rank.
        row_bytes: Number of bytes in one KV row.

    Returns:
        A pair ``(kv_offset, packet_bytes)``. IDs occupy the prefix and KV rows
        begin at the 256-byte-aligned ``kv_offset``.

    Raises:
        ValueError: If either dimension is not positive.

    Notes:
        Keeping IDs in a sidecar prefix preserves aligned, contiguous KV rows.
        Interleaving an eight-byte ID before every row makes every large KV
        load and store unaligned and severely slows the fused packet path.
    """
    if padded_blocks <= 0 or row_bytes <= 0:
        raise ValueError("owner packet dimensions must be positive")
    id_bytes = padded_blocks * 8
    kv_offset = (
        (id_bytes + _OWNER_PACKET_ALIGNMENT - 1) // _OWNER_PACKET_ALIGNMENT
    ) * _OWNER_PACKET_ALIGNMENT
    return kv_offset, kv_offset + padded_blocks * row_bytes


@triton.jit
def _pack_owner_rows(
    source: tl.tensor,
    row_map: tl.tensor,
    ids: tl.tensor,
    packet: tl.tensor,
    n_local: tl.tensor,
    local_ready: tl.tensor,
    row_limit: tl.tensor,
    source_limit: tl.tensor,
    source_stride: tl.tensor,
    ROW_BYTES: tl.constexpr,
    KV_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    block = tl.load(ids + row, mask=row < n_local, other=-1)
    valid = local_ready & (row < n_local) & (block >= 0) & (block < row_limit)
    physical = tl.load(row_map + block, mask=valid, other=0)
    valid = valid & (physical >= 0) & (physical < source_limit)
    if tile == 0:
        header = tl.where(valid, block, tl.where((~local_ready) & (row == 0), -2, -1))
        tl.store((packet + row * 8).to(tl.pointer_type(tl.int64)), header)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        source + physical.to(tl.int64) * source_stride + offsets,
        mask=valid & (offsets < ROW_BYTES),
        other=0,
    )
    tl.store(
        packet + KV_OFFSET + row * ROW_BYTES + offsets,
        values,
        mask=offsets < ROW_BYTES,
    )


@triton.jit
def _claim_owner_rows(
    packet: tl.tensor,
    claims: tl.tensor,
    rows: tl.tensor,
    logical_limit: tl.tensor,
    ROWS_PER_RANK: tl.constexpr,
    PACKET_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    rows_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = rows_idx // ROWS_PER_RANK
    local_row = rows_idx % ROWS_PER_RANK
    block = tl.load(
        (packet + rank * PACKET_BYTES + local_row * 8).to(
            tl.pointer_type(tl.int64)
        ),
        mask=rows_idx < rows,
        other=-1,
    )
    valid = (rows_idx < rows) & (block >= 0) & (block < logical_limit)
    tl.atomic_min(claims + block, rows_idx.to(tl.int32), mask=valid, sem="relaxed")


@triton.jit
def _scatter_owner_rows(
    packet: tl.tensor,
    claims: tl.tensor,
    row_map: tl.tensor,
    destination: tl.tensor,
    resident: tl.tensor,
    selected: tl.tensor,
    logical_limit: tl.tensor,
    destination_limit: tl.tensor,
    destination_stride: tl.tensor,
    ROW_BYTES: tl.constexpr,
    ROWS_PER_RANK: tl.constexpr,
    KV_OFFSET: tl.constexpr,
    PACKET_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    rank = row // ROWS_PER_RANK
    local_row = row % ROWS_PER_RANK
    rank_packet = packet + rank * PACKET_BYTES
    block = tl.load((rank_packet + local_row * 8).to(tl.pointer_type(tl.int64)))
    valid = (block >= 0) & (block < logical_limit)
    owner = tl.load(claims + block, mask=valid, other=-1)
    valid = valid & (owner == row)
    physical = tl.load(row_map + block, mask=valid, other=-1)
    valid = valid & (physical >= 0) & (physical < destination_limit)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        rank_packet + KV_OFFSET + local_row * ROW_BYTES + offsets,
        mask=valid & (offsets < ROW_BYTES),
        other=0,
    )
    tl.store(
        destination + physical.to(tl.int64) * destination_stride + offsets,
        values,
        mask=valid & (offsets < ROW_BYTES),
    )
    if tile == 0:
        tl.store(resident + block, 1, mask=valid)
        tl.store(selected + row, tl.where(valid, block, -1))


def pack_owner_rows(
    source: torch.Tensor,
    logical_rows: torch.Tensor,
    local_ids: torch.Tensor,
    packet: torch.Tensor,
    *,
    local_ready: bool,
    padded_blocks: int,
) -> None:
    """Pack IDs and byte-exact KV rows on the current CUDA stream.

    Args:
        source: uint8 [physical_rows, row_bytes] K cache, optionally strided.
        logical_rows: int64 logical-to-physical row mapping on CUDA.
        local_ids: int64 rank-local candidate IDs on CUDA.
        packet: Flat contiguous uint8 output using :func:`owner_packet_layout`.
        local_ready: False publishes failure (-2) rather than any KV rows.
        padded_blocks: Fixed number of packet rows, including padding.

    Notes:
        Padding is zeroed and marked -1. No device-to-host transfer occurs.
    """
    row_bytes = int(source.shape[1])
    kv_offset, packet_bytes = owner_packet_layout(padded_blocks, row_bytes)
    if packet.dtype != torch.uint8 or packet.numel() != packet_bytes:
        raise ValueError("owner send packet has the wrong byte geometry")
    _pack_owner_rows[(padded_blocks, triton.cdiv(row_bytes, 1024))](
        source,
        logical_rows,
        local_ids,
        packet,
        local_ids.numel(),
        local_ready,
        logical_rows.numel(),
        source.shape[0],
        source.stride(0),
        ROW_BYTES=row_bytes,
        KV_OFFSET=kv_offset,
        BLOCK=1024,
    )


def scatter_owner_packets(
    packet: torch.Tensor,
    claims: torch.Tensor,
    logical_rows: torch.Tensor,
    destination: torch.Tensor,
    resident: torch.Tensor,
    *,
    padded_blocks: int,
    world_size: int,
) -> torch.Tensor:
    """Scatter a gathered ID+KV packet, deduplicating on the GPU.

    Args:
        packet: Flat rank-major uint8 packets using :func:`owner_packet_layout`.
        claims: Reusable int32 [logical_rows] scratch storage.
        logical_rows: int64 logical-to-physical row map on CUDA.
        destination: Final uint8 K cache, optionally strided between rows.
        resident: Bool logical residency bitmap updated only for valid rows.
        padded_blocks: Fixed number of rows contributed by each rank.
        world_size: Number of rank packets in ``packet``.

    Returns:
        Fixed-size int64 IDs with duplicates, padding and failed rows set to
        -1. The first rank-major occurrence wins. Consumers must await CUDA
        completion before using either the KV or its residency bitmap.
    """
    row_bytes = int(destination.shape[1])
    kv_offset, packet_bytes = owner_packet_layout(padded_blocks, row_bytes)
    if (
        world_size <= 0
        or packet.dtype != torch.uint8
        or packet.numel() != world_size * packet_bytes
    ):
        raise ValueError("owner receive packet has the wrong byte geometry")
    count = padded_blocks * world_size
    limit = min(logical_rows.numel(), resident.numel())
    claims.fill_(count)
    selected = torch.empty(count, dtype=torch.int64, device=destination.device)
    _claim_owner_rows[(triton.cdiv(count, 256),)](
        packet,
        claims,
        count,
        limit,
        ROWS_PER_RANK=padded_blocks,
        PACKET_BYTES=packet_bytes,
        BLOCK=256,
    )
    _scatter_owner_rows[(count, triton.cdiv(row_bytes, 1024))](
        packet,
        claims,
        logical_rows,
        destination,
        resident,
        selected,
        limit,
        destination.shape[0],
        destination.stride(0),
        ROW_BYTES=row_bytes,
        ROWS_PER_RANK=padded_blocks,
        KV_OFFSET=kv_offset,
        PACKET_BYTES=packet_bytes,
        BLOCK=1024,
    )
    return selected
