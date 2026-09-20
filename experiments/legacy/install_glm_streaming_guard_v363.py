# SPDX-License-Identifier: Apache-2.0
"""Install the compact-main admission fix into a disposable container only."""

# Standard
import argparse
from pathlib import Path


def install(root: Path) -> None:
    """Preserve GLM native indexer K in the historical compact-main payload.

    Args:
        root: Installed lmcache package in the new experiment container.

    Raises:
        RuntimeError: The expected source anchor is missing or ambiguous.
    """
    path = root / "v1/cache_engine.py"
    source = path.read_text()
    marker = "GLM_COMPACT_MAIN_GUARD_V363"
    if marker in source:
        return
    begin = source.index("        compact_candidates = (", source.index(
        "    def prepare_dsv4_streaming_retrieve("))
    end = source.index("        if any(\n", begin)
    old = source[begin:end]
    if "for key, start, end in blocks if end > tail_start" not in old:
        raise RuntimeError("unexpected compact candidate filter")
    new = '''        # GLM_COMPACT_MAIN_GUARD_V363: native GLM indexer K remains in
        # compact main storage; historical rows must never be skipped as tail.
        compact_candidates = (
            ((key, start, end) for key, start, end in blocks if end > tail_start)
            if (
                tail_start is not None
                and role == _DSV4_CSA_HCA_DEFERRED_RETRIEVE_ROLE
                and not _glm_dsa_layer_major_enabled()
            )
            else ((key, start, end) for key, start, end in blocks)
        )
'''
    modified = source[:begin] + new + source[end:]
    compile(modified, str(path), "exec")
    backup = path.with_name("cache_engine.before_guard_v363.py")
    if backup.exists():
        raise RuntimeError("refusing to overwrite previous source backup")
    backup.write_text(source)
    path.write_text(modified)
    print(marker, "INSTALLED", path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)
