# SPDX-License-Identifier: Apache-2.0
"""Install an exact non-truncating block-union path at normal startup."""

# Standard
import argparse
from pathlib import Path

# Local
from install_aligned_entry_io_v374 import replace_once


def install(root: Path) -> None:
    """Use the GPU union only when it retains exactly the original candidate set.

    Args:
        root: Installed LMCache package in a new experiment container.
    Raises:
        RuntimeError: The expected prediction-only selection anchor differs.
    """
    path = root / "v1/indexer_ssd_manager.py"
    source = path.read_text()
    if "PROXY_BLOCK_UNION_V380" not in source:
        source = replace_once(
            source,
            "    flat = token_ids.reshape(-1).to(dtype=torch.int64)\n",
            """    # PROXY_BLOCK_UNION_V380: no budget truncation, so no count sort.
    if (fixed_metadata and token_ids.is_cuda and num_blocks <= min(block_budget, 2048)
            and os.getenv("LMCACHE_DSV4_FUSED_PROXY_BLOCK_UNION", "0") == "1"):
        from lmcache.v1.proxy_block_union_v380 import unique_proxy_blocks

        return unique_proxy_blocks(token_ids, cursor, num_blocks)
    flat = token_ids.reshape(-1).to(dtype=torch.int64)
""",
        )
        compile(source, str(path), "exec")
        path.write_text(source)
    (root / "v1/proxy_block_union_v380.py").write_bytes(
        Path("/tmp/proxy_block_union_v380.py").read_bytes()
    )
    print("PROXY_BLOCK_UNION_V380_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)
