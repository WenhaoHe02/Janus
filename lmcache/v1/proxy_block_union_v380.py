# SPDX-License-Identifier: Apache-2.0
"""Exact GPU block union when the existing block budget cannot truncate it."""

# Third Party
import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["rows", "columns", "row_stride", "column_stride", "cursor"]
)
def _mark(
    tokens,
    bitmap,
    rows,
    columns,
    row_stride,
    column_stride,
    cursor,
    tile: tl.constexpr,
):
    flat = tl.program_id(0) * tile + tl.arange(0, tile)
    values = tl.load(
        tokens + (flat // columns) * row_stride + (flat % columns) * column_stride,
        flat < rows * columns,
        other=-1,
    )
    valid = (flat < rows * columns) & (values >= 0) & (values < cursor)
    blocks = tl.where(valid, values // 64, 0).to(tl.int32)
    tl.atomic_or(bitmap + blocks, 1, valid, sem="relaxed")


@triton.jit(do_not_specialize=["n"])
def _compact(bitmap, output, n, tile: tl.constexpr):
    index = tl.arange(0, tile)
    selected = tl.load(bitmap + index, index < n, other=0) != 0
    position = tl.cumsum(selected.to(tl.int32)) - 1
    total = tl.sum(selected.to(tl.int32))
    # Prefix and padding stores have disjoint destinations; no barrier is needed.
    tl.store(output + index, -1, (index < n) & (index >= total))
    tl.store(output + position, index.to(tl.int64), selected)


def unique_proxy_blocks(
    token_ids: torch.Tensor, cursor: int, num_blocks: int
) -> torch.Tensor:
    """Return all selected initialized 64-entry blocks, sorted with -1 padding.

    Args:
        token_ids: One- or two-dimensional CUDA int32/int64 predictions; a
            strided top-k column slice is accepted without materializing it.
        cursor: Exclusive initialized-entry limit, in [0, num_blocks * 64].
        num_blocks: Output/bitmap capacity, in [1, 2048].
    Returns:
        CUDA int64[num_blocks], unique valid block IDs followed by -1 padding.
    Raises:
        ValueError: Unsupported geometry, dtype, or device.
    Notes:
        Caller must ensure its block budget is at least num_blocks. This
        replaces counting followed by selecting every block, not the indexer's
        Top-K scoring or any genuinely truncating budget decision. Kernels run
        on the caller's stream and do not read counts back to the CPU.
    """
    if (
        not token_ids.is_cuda
        or token_ids.ndim not in (1, 2)
        or token_ids.dtype not in (torch.int32, torch.int64)
        or not 0 < num_blocks <= 2048
        or not 0 <= cursor <= num_blocks * 64
    ):
        raise ValueError("invalid complete-budget proxy union geometry")
    rows = token_ids.shape[0]
    columns = token_ids.shape[1] if token_ids.ndim == 2 else 1
    row_stride = token_ids.stride(0)
    column_stride = token_ids.stride(1) if token_ids.ndim == 2 else 0
    bitmap = torch.zeros(num_blocks, dtype=torch.int32, device=token_ids.device)
    output = torch.empty(num_blocks, dtype=torch.int64, device=token_ids.device)
    if rows and columns:
        _mark[(triton.cdiv(rows * columns, 256),)](
            token_ids,
            bitmap,
            rows,
            columns,
            row_stride,
            column_stride,
            cursor,
            256,
        )
    _compact[(1,)](
        bitmap,
        output,
        num_blocks,
        triton.next_power_of_2(num_blocks),
    )
    return output
