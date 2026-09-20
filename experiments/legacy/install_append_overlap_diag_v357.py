"""Measure whether the append region's K-cache rows overlap the history's rows.

Everything measured so far shows the corruption scales with the number of tokens
computed inside the same forward that reads external KV: in-forward 12 and 128
are byte-exact, 512 and above are reliably broken. No hypothesis tested so far
explains that dependence.

One mechanism would. LMCache only ever sees ``slot_mapping[:lmcache_cached_tokens]``
-- the history slice. The append region's rows never appear in any LMCache-side
log. If vLLM's rows for the append tokens intersect the rows LMCache fills with
history KV, then the model's own append writes clobber history KV (or the reverse),
and the damage is proportional to the number of append blocks. That is exactly the
observed scaling.

The adapter is the only place holding the *complete* ``slot_mapping``, so the
check belongs here, immediately after the assert that ties it to ``tokens``:

``CSA_APPEND_OVERLAP``
    ``total_tokens``, ``lmcache_cached_tokens``, block counts for each region,
    and the size of the intersection between the history block set and the append
    block set. A non-zero intersection is the defect; zero rules the mechanism out
    and, with source addresses already proven correct, points downstream instead.

Rows are derived the same way LMCache derives them (``slot // block_size``) so the
two logs are directly comparable. ``block_size`` is read from the adapter's own
``self._block_size`` rather than assumed.

Observe-only: reads an existing CPU/GPU tensor, mutates nothing, submits no I/O.
Gated on ``LMCACHE_CSA_APPEND_OVERLAP_DIAG=1``.

Usage:
    python3 install_append_overlap_diag_v357.py --lmcache-root <site-packages>/lmcache
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "CSA_APPEND_OVERLAP_DIAG_V357"

ANCHOR = """            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
"""

PATCH = '''            assert len(tokens) == len(slot_mapping)
            # CSA_APPEND_OVERLAP_DIAG_V357: observe-only, once per request.
            if os.getenv("LMCACHE_CSA_APPEND_OVERLAP_DIAG", "0") == "1":
                try:
                    _cached = int(request.load_spec.lmcache_cached_tokens)
                    _bs = int(self._block_size)
                    _sm = slot_mapping.detach().to(
                        device="cpu", dtype=torch.int64
                    ).reshape(-1)
                    _hist = _sm[:_cached]
                    _app = _sm[_cached:]
                    _hist_rows = torch.unique(_hist // _bs) if _hist.numel() else _hist
                    _app_rows = torch.unique(_app // _bs) if _app.numel() else _app
                    _inter = 0
                    if _hist_rows.numel() and _app_rows.numel():
                        _inter = int(
                            torch.isin(_app_rows, _hist_rows).sum().item()
                        )
                    logger.info(
                        "CSA_APPEND_OVERLAP request=%s total_tokens=%d "
                        "cached_tokens=%d block_size=%d hist_rows=%d "
                        "append_rows=%d overlap_rows=%d "
                        "hist_row_min=%s hist_row_max=%s "
                        "append_row_min=%s append_row_max=%s",
                        getattr(request, "req_id", "unknown"),
                        int(_sm.numel()),
                        _cached,
                        _bs,
                        int(_hist_rows.numel()),
                        int(_app_rows.numel()),
                        _inter,
                        int(_hist_rows.min()) if _hist_rows.numel() else None,
                        int(_hist_rows.max()) if _hist_rows.numel() else None,
                        int(_app_rows.min()) if _app_rows.numel() else None,
                        int(_app_rows.max()) if _app_rows.numel() else None,
                    )
                except Exception:
                    pass

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
'''


def patch_adapter(root: Path) -> bool:
    """Insert the append/history row-overlap log into the vLLM adapter.

    Args:
        root: Installed ``lmcache`` package directory.

    Returns:
        True when modified, False when already patched.

    Raises:
        RuntimeError: If the anchor is missing or not unique.
    """
    path = root / "integration" / "vllm" / "vllm_v1_adapter.py"
    source = path.read_text()
    if MARKER in source:
        return False
    count = source.count(ANCHOR)
    if count != 1:
        raise RuntimeError(f"anchor found {count} times, expected 1")
    path.write_text(source.replace(ANCHOR, PATCH, 1))
    return True


def main() -> None:
    """Patch the installed adapter and report what changed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / "integration" / "vllm" / "vllm_v1_adapter.py"
    if not target.exists():
        raise RuntimeError(f"not found: {target}")
    changed = patch_adapter(args.lmcache_root)
    print(f"APPEND_OVERLAP_DIAG_V357_INSTALLED changed={int(changed)}")


if __name__ == "__main__":
    main()
