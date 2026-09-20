"""Install read-only diagnostics on the indexed CSA read path GLM actually uses.

An earlier attempt logged ``_resolve_range_ios`` and the adapter's coverage
warning. Both were the wrong targets: with
``LMCACHE_CSA_LAYER_MAJOR_INDEXED_SPARSE=1`` GLM reads through
``_issue_indexed_reads``, which submits precompiled ``indexed_slba_table`` /
``indexed_dst_rows_table`` entries via ``load_indexed_chunks_to_hbm`` and never
calls ``_resolve_range_ios``. The adapter's "failed to load N tokens" warning is
a comparison of the adapter's expected mask against the mask ``retrieve()``
returned, so it reports a plan rejection rather than a byte-coverage failure.

This patch instruments the real path at two points, both gated on
``LMCACHE_CSA_INDEXED_READ_DIAG=1`` so a normal run pays nothing:

``CSA_IDX_SUBMIT``
    Before submission, for a small fixed set of layers: the number of selected
    blocks and, for a few sampled block ids, the ``(block_id, slba, dst_row)``
    actually being submitted. This shows whether two reads of one prompt resolve
    the same logical block to the same NVMe address and the same GPU row.

``CSA_IDX_BYTES``
    After the scatter event is synchronized, a checksum of those same
    ``dst_row`` entries in the K cache. Matching ``(slba, dst_row)`` with
    differing checksums separates "wrong mapping" from "right mapping, wrong
    bytes".

Together they split one good/bad pair four ways:

  * same slba, same dst_row, same bytes -> defect is downstream (attention or
    indexer), not in the load.
  * same slba, same dst_row, different bytes -> NVMe content differs, or a
    concurrent writer is touching those rows.
  * different slba -> the source mapping is wrong.
  * different dst_row -> the destination mapping is wrong.

The checksum path adds a stream synchronize that a normal run does not perform,
so it can mask a race. That is why the mapping log is gated separately from the
byte log: run with ``CSA_IDX_BYTES`` off first to keep timing untouched.

Usage:
    python3 install_indexed_read_diag_v350.py --lmcache-root <site-packages>/lmcache
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "CSA_INDEXED_READ_DIAG_V350"

SUBMIT_ANCHOR = """        slba_table = state.indexed_slba_table
        dst_rows_table = state.indexed_dst_rows_table
        if slba_table is None or dst_rows_table is None:
            raise RuntimeError("indexed CSA tables are unavailable")
"""

SUBMIT_PATCH = '''        slba_table = state.indexed_slba_table
        dst_rows_table = state.indexed_dst_rows_table
        if slba_table is None or dst_rows_table is None:
            raise RuntimeError("indexed CSA tables are unavailable")
        # CSA_INDEXED_READ_DIAG_V350: observe-only mapping log.
        _diag_v350 = os.getenv("LMCACHE_CSA_INDEXED_READ_DIAG", "0") == "1"
        _diag_rows_v350 = []
        if _diag_v350 and int(getattr(state, "layer_id", -1)) in (2, 40, 77):
            try:
                _ids = torch.as_tensor(
                    sorted_block_ids, dtype=torch.int64, device="cpu"
                ).reshape(-1)
                _n = int(_ids.numel())
                if _n:
                    _picks = sorted({0, _n // 2, _n - 1})
                    _slba_cpu = slba_table.detach().to("cpu")
                    _rows_cpu = dst_rows_table.detach().to("cpu")
                    for _p in _picks:
                        _bid = int(_ids[_p])
                        if 0 <= _bid < int(_slba_cpu.numel()):
                            _diag_rows_v350.append(
                                (_bid, int(_slba_cpu[_bid]), int(_rows_cpu[_bid]))
                            )
                    logger.info(
                        "CSA_IDX_SUBMIT request=%s layer=%d blocks=%d samples=%s",
                        self.active_request_id,
                        int(state.layer_id),
                        _n,
                        _diag_rows_v350,
                    )
            except Exception:
                _diag_rows_v350 = []
'''

BYTES_ANCHOR = """        with torch.inference_mode(), torch.cuda.stream(io_stream):
            state.in_pool_bitmap.index_fill_(0, selected_ids, True)
            last_scatter_event = torch.cuda.Event()
            last_scatter_event.record(io_stream)
        load_ms = (time.perf_counter() - load_start) * 1000.0
"""

BYTES_PATCH = '''        with torch.inference_mode(), torch.cuda.stream(io_stream):
            state.in_pool_bitmap.index_fill_(0, selected_ids, True)
            last_scatter_event = torch.cuda.Event()
            last_scatter_event.record(io_stream)
        load_ms = (time.perf_counter() - load_start) * 1000.0
        # CSA_INDEXED_READ_DIAG_V350: observe-only byte checksum. Adds a
        # synchronize that a normal run does not do, so keep it off unless the
        # mapping log has already been compared.
        if (
            _diag_rows_v350
            and os.getenv("LMCACHE_CSA_INDEXED_READ_BYTES", "0") == "1"
        ):
            try:
                last_scatter_event.synchronize()
                with torch.inference_mode():
                    _sums = []
                    for _bid, _slba, _row in _diag_rows_v350:
                        if 0 <= _row < int(k_cache_flat.shape[0]):
                            _sums.append(
                                (
                                    _bid,
                                    _slba,
                                    _row,
                                    int(
                                        torch.sum(
                                            k_cache_flat[_row].to(torch.int64)
                                        ).item()
                                    ),
                                )
                            )
                logger.info(
                    "CSA_IDX_BYTES request=%s layer=%d samples=%s",
                    self.active_request_id,
                    int(state.layer_id),
                    _sums,
                )
            except Exception:
                pass
'''


def patch_manager(root: Path) -> bool:
    """Insert both diagnostics into the CSA attention KV prefetch manager.

    Args:
        root: Installed ``lmcache`` package directory.

    Returns:
        True when the file was modified, False when already patched.

    Raises:
        RuntimeError: If an anchor is missing or not unique, meaning the
            installed source differs from what this patch targets.
    """
    path = root / "v1" / "csa_attention_kv_prefetch_manager.py"
    source = path.read_text()
    if MARKER in source:
        return False

    for name, anchor in (("submit", SUBMIT_ANCHOR), ("bytes", BYTES_ANCHOR)):
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"{name} anchor found {count} times, expected 1")

    source = source.replace(SUBMIT_ANCHOR, SUBMIT_PATCH, 1)
    source = source.replace(BYTES_ANCHOR, BYTES_PATCH, 1)
    path.write_text(source)
    return True


def main() -> None:
    """Patch the installed manager and report what changed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()

    target = args.lmcache_root / "v1" / "csa_attention_kv_prefetch_manager.py"
    if not target.exists():
        raise RuntimeError(f"not found: {target}")

    changed = patch_manager(args.lmcache_root)
    print(f"INDEXED_READ_DIAG_V350_INSTALLED changed={int(changed)}")


if __name__ == "__main__":
    main()
