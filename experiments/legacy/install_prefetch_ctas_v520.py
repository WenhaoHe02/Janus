# SPDX-License-Identifier: Apache-2.0
"""Add only communicator-local options to the deployed stable transport."""

# Standard
import argparse
from pathlib import Path


def install(root: Path, helper: Path) -> None:
    """Patch a fresh copied transport; never replace it with a different branch."""
    path = root / "v1/ssd_tp_sharded_prefetch.py"
    source = path.read_text()
    old = (
        '        process_group = dist.new_group(ranks=ranks, backend="nccl")\n'
        '        metadata_process_group = dist.new_group(ranks=ranks, backend="nccl")'
    )
    if source.count(old) != 1:
        raise RuntimeError("stable prefetch communicator creation anchor differs")
    new = """        from lmcache.v1.prefetch_nccl_config import prefetch_nccl_group_options
        process_group = dist.new_group(
            ranks=ranks, backend="nccl", **prefetch_nccl_group_options(dist)
        )
        metadata_process_group = dist.new_group(
            ranks=ranks, backend="nccl", **prefetch_nccl_group_options(dist)
        )"""
    source = source.replace(old, new, 1)
    compile(source, str(path), "exec")
    module = helper.read_text()
    compile(module, str(helper), "exec")
    path.write_text(source)
    (root / "v1/prefetch_nccl_config.py").write_text(module)
    print("PREFETCH_PER_COMMUNICATOR_CTAS_V520_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument(
        "--helper", type=Path, default=Path("/tmp/prefetch_nccl_config.py")
    )
    args = parser.parse_args()
    install(args.lmcache_root, args.helper)
