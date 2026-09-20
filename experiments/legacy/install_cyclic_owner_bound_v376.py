# SPDX-License-Identifier: Apache-2.0
"""Expose cyclic owner geometry to the existing GPU-only prediction submission."""

# Standard
import argparse
from pathlib import Path

# Local
from install_aligned_entry_io_v374 import replace_once


def install(root: Path) -> None:
    """Add missing runtime_info for cyclic K shards, without changing selection.

    Args:
        root: Installed LMCache package in the new experiment container.
    Raises:
        RuntimeError: Expected fused owner scorer anchor is missing or ambiguous.
    """
    path = root / "v1/csa_prefill_cp_scorer.py"
    source = path.read_text()
    if "CYCLIC_OWNER_BOUND_V376" in source:
        return
    source = replace_once(
        source,
        "    if not local_rows or not query_rows:\n",
        """    # CYCLIC_OWNER_BOUND_V376: cyclic partitioning also has a common bound.
    if runtime_info is not None and not contiguous_partition and block_size == 64:
        from lmcache.v1.owner_block_bound_v376 import cyclic_owner_block_bound

        runtime_info["owner_block_bound"] = cyclic_owner_block_bound(
            total_seq_lens, block_size, world_size, interleave_size
        )
    if not local_rows or not query_rows:
""",
    )
    compile(source, str(path), "exec")
    path.write_text(source)
    (root / "v1/owner_block_bound_v376.py").write_bytes(
        Path("/tmp/owner_block_bound_v376.py").read_bytes()
    )
    print("CYCLIC_OWNER_BOUND_V376_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)
