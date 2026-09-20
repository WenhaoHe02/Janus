# SPDX-License-Identifier: Apache-2.0
"""Rank-invariant capacity for the existing cyclic K-block ownership map."""


def cyclic_owner_block_bound(
    total_entries: int, block_size: int, world_size: int, interleave_blocks: int
) -> int:
    """Return the maximum owned block count across ranks, including a partial block.

    Args:
        total_entries: Existing scorer's total compressed K-entry span.
        block_size: K entries in one physical/logical cache block.
        world_size: Number of ranks owning alternating groups.
        interleave_blocks: Consecutive blocks assigned to each rank per cycle.

    Returns:
        Positive rank-invariant capacity; an empty span reserves one padding slot.

    Raises:
        ValueError: Negative span or non-positive ownership geometry.
    """
    if total_entries < 0 or min(block_size, world_size, interleave_blocks) <= 0:
        raise ValueError("invalid cyclic owner geometry")
    blocks = (total_entries + block_size - 1) // block_size
    cycles, remainder = divmod(blocks, world_size * interleave_blocks)
    return max(1, cycles * interleave_blocks + min(remainder, interleave_blocks))
