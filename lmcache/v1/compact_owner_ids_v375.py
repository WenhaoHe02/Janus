# SPDX-License-Identifier: Apache-2.0
"""Compact valid owner history IDs on GPU without a count readback or truncation."""

# Third Party
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n", "limit", "capacity"])
def _compact(ids, output, n, limit, capacity, tile: tl.constexpr):
    position = tl.arange(0, tile)
    value = tl.load(ids + position, position < n, other=-1)
    valid = (position < n) & (value >= 0) & (value < limit)
    offset = tl.cumsum(valid.to(tl.int32)) - 1
    total = tl.sum(valid.to(tl.int32))
    tl.device_assert(total <= capacity, "owner history IDs exceed geometric bound")
    # These stores target disjoint prefix/tail regions, so no fill/scatter race.
    tl.store(output + position, -1, (position < capacity) & (position >= total))
    tl.store(output + offset, value.to(tl.int64), valid & (offset < capacity))


def compact_owner_history_ids(
    selected_ids: torch.Tensor, history_blocks: int, capacity: int
) -> torch.Tensor:
    """Preserve every selected historical ID in a fixed-capacity GPU vector.

    Args:
        selected_ids: Contiguous 1-D CUDA int32/int64 IDs, possibly interspersed
            with negative padding or append IDs. The producer emits unique IDs.
        history_blocks: Exclusive upper bound of the registered cached history.
        capacity: Rank-invariant geometric owner bound, not a new selection quota.

    Returns:
        CUDA int64 vector of capacity entries. Valid IDs retain input order,
        followed by -1 padding. No valid history ID is silently dropped.

    Raises:
        ValueError: Invalid metadata. A GPU assertion fails closed on overflow.

    Notes:
        Enqueues on the current CUDA stream. The caller must order that stream
        after the selection event and publish a new event after compaction.
        Runtime n/limit/capacity avoid compiling for every exact history length.
    """
    if (
        not selected_ids.is_cuda
        or selected_ids.ndim != 1
        or selected_ids.dtype not in (torch.int32, torch.int64)
        or not selected_ids.is_contiguous()
        or selected_ids.numel() <= 0
        or history_blocks <= 0
        or capacity <= 0
        or capacity > selected_ids.numel()
    ):
        raise ValueError("invalid owner compaction metadata")
    result = torch.empty(capacity, dtype=torch.int64, device=selected_ids.device)
    _compact[(1,)](
        selected_ids,
        result,
        selected_ids.numel(),
        history_blocks,
        capacity,
        tile=triton.next_power_of_2(selected_ids.numel()),
        debug=True,
    )
    return result
