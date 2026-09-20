"""Log vLLM's slot_mapping beside the physical rows LMCache derives from it.

The indexed-read diagnostic proved the *source* side is right: for one prompt,
two consecutive cached reads resolved the same logical block to the same NVMe
address. What it could not check is whether the *destination* row is right,
because it only logged rows LMCache had computed itself. A byte checksum at
those rows is equally blind: it verifies the bytes LMCache just wrote are at the
row LMCache chose, so it reads back clean even when that row is not where vLLM
placed the history.

Self-consistency is not correctness either. Both observed mappings were valid
bijections (one contiguous, one reversed); a consistently wrong permutation
looks exactly the same from inside. The only ground truth is vLLM's
``slot_mapping``.

This patch logs, at the single derivation site in
``_dsv4_build_indexer_cache_chunks``, the raw inputs and outputs:

``CSA_SLOTMAP_DERIVE``
    ``compress_ratio``, ``tokens_per_block``, ``block_size``, the
    ``slot_mapping`` length, ``total_blocks``, and for the first/middle/last
    block the triple ``(block_index, position, raw_slot, derived_physical_row)``.

Two things become checkable that were not before:

  * whether ``tokens_per_block`` matches the real layout — if ``compress_ratio``
    were wrong, ``positions`` would stride incorrectly and every derived row
    would point somewhere unrelated, while still forming a tidy bijection.
  * whether ``raw_slot // tokens_per_block`` reproduces vLLM's own block for
    that token, comparable against the block table for the same request.

Observe-only: reads existing CPU tensors, forwards nothing, mutates nothing.
Gated on ``LMCACHE_CSA_SLOTMAP_DIAG=1``.

Usage:
    python3 install_slotmap_derivation_diag_v356.py --lmcache-root <site-packages>/lmcache
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "CSA_SLOTMAP_DIAG_V356"

ANCHOR = """        physical_blocks = torch.div(
            selected_slots,
            tokens_per_block,
            rounding_mode="floor",
        )
"""

PATCH = '''        physical_blocks = torch.div(
            selected_slots,
            tokens_per_block,
            rounding_mode="floor",
        )
        # CSA_SLOTMAP_DIAG_V356: observe-only. One record per call, not per I/O.
        if os.getenv("LMCACHE_CSA_SLOTMAP_DIAG", "0") == "1":
            try:
                _n = int(physical_blocks.numel())
                _picks = sorted({0, _n // 2, _n - 1}) if _n else []
                _trip = [
                    (
                        int(_i),
                        int(positions[_i]),
                        int(selected_slots[_i]),
                        int(physical_blocks[_i]),
                    )
                    for _i in _picks
                ]
                logger.info(
                    "CSA_SLOTMAP_DERIVE compress_ratio=%d tokens_per_block=%d "
                    "block_size=%d slot_mapping_len=%d total_blocks=%d "
                    "first_start=%d samples=%s",
                    int(compression_ratio),
                    int(tokens_per_block),
                    int(block_size),
                    int(slots_cpu.numel()),
                    int(total_blocks),
                    int(blocks[0][1]),
                    _trip,
                )
            except Exception:
                pass
'''


def patch_engine(root: Path) -> bool:
    """Insert the slot-mapping derivation log into the cache engine.

    Args:
        root: Installed ``lmcache`` package directory.

    Returns:
        True when modified, False when already patched.

    Raises:
        RuntimeError: If the anchor is missing or not unique.
    """
    path = root / "v1" / "cache_engine.py"
    source = path.read_text()
    if MARKER in source:
        return False
    count = source.count(ANCHOR)
    if count != 1:
        raise RuntimeError(f"anchor found {count} times, expected 1")
    path.write_text(source.replace(ANCHOR, PATCH, 1))
    return True


def main() -> None:
    """Patch the installed cache engine and report what changed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / "v1" / "cache_engine.py"
    if not target.exists():
        raise RuntimeError(f"not found: {target}")
    print(f"SLOTMAP_DIAG_V356_INSTALLED changed={int(patch_engine(args.lmcache_root))}")


if __name__ == "__main__":
    main()
