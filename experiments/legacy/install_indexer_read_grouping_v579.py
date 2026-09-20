# SPDX-License-Identifier: Apache-2.0
"""Opt-in native Indexer-K DMA grouping, installed only before worker import."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil


def transform(source: str) -> str:
    """Replace only the audited native read grouping while preserving other code.

    Args:
        source: Deployed CSA manager with the exact-tail v374 reader installed.
    Returns:
        Patched source, or the same source when this patch is already present.
    Raises:
        ValueError: The tail reader or exact reviewed grouping block is absent.
    """
    marker = "# INDEXER_READ_GROUPING_V579"
    if marker in source:
        return source
    if "def _enqueue_aligned_group_read(" not in source:
        raise ValueError("install the byte-exact aligned-tail reader first")
    old = (
        "        group_rows = next((size for size in (6, 4, 2) "
        "if rows % size == 0), 0)\n"
        """        if (not group_rows
                and os.getenv("LMCACHE_DSV4_ALIGNED_ENTRY_IO", "0") == "1"):
            group_rows = 2
"""
    )
    if source.count(old) != 1:
        raise ValueError("native grouping differs from reviewed deployed version")
    new = """        # INDEXER_READ_GROUPING_V579
        from lmcache.v1.indexer_read_grouping import native_indexer_group_rows

        group_rows = native_indexer_group_rows(
            rows,
            aligned_tail_supported=(
                os.getenv("LMCACHE_DSV4_ALIGNED_ENTRY_IO", "0") == "1"
            ),
            preferred_rows=int(os.getenv("LMCACHE_DSV4_INDEXER_READ_GROUP_ROWS", "0")),
        )
"""
    updated = source.replace(old, new)
    ast.parse(updated)
    return updated


def install(root: Path, helper: Path) -> None:
    """Install reviewed source and retain a single immutable previous-code copy.

    Args:
        root: Fresh container lmcache package directory, before worker import.
        helper: Pure grouping helper to copy into the same package.
    Raises:
        ValueError: Source checks fail or an unrelated previous backup exists.
    """
    target = root / "v1/csa_attention_kv_prefetch_manager.py"
    source = target.read_text(encoding="utf-8")
    updated = transform(source)
    backup = target.with_suffix(".py.pre_indexer_group_v579")
    if updated != source:
        if backup.exists() and backup.read_text(encoding="utf-8") != source:
            raise ValueError("previous grouping backup differs")
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(helper, root / "v1/indexer_read_grouping.py")
        target.write_text(updated, encoding="utf-8")
    print("INDEXER_READ_GROUPING_V579_INSTALLED", flush=True)


def wire(startup: Path) -> None:
    """Add an opt-in pre-import installer, preserving the original startup.

    Args:
        startup: Existing terminal startup script with one vLLM exec line.
    Raises:
        ValueError: The exec anchor or backup does not match the expected source.
    """
    source = startup.read_text(encoding="utf-8")
    marker = "# INDEXER_GROUP_STARTUP_V579"
    if marker in source:
        return
    anchor = (
        "exec ${LMCACHE_EXEC_PREFIX:-} python3 -m vllm.entrypoints.openai.api_server"
    )
    if source.count(anchor) != 1:
        raise ValueError("unexpected terminal startup exec")
    addition = """# INDEXER_GROUP_STARTUP_V579
if [ "${LMCACHE_DSV4_INDEXER_READ_GROUP_ROWS:-0}" != "0" ]; then
  python3 /tmp/install_indexer_read_grouping_v579.py \\
    --lmcache-root "$LMCACHE_SITE" --helper /tmp/indexer_read_grouping.py
fi
"""
    backup = startup.with_name(startup.name + ".pre_indexer_group_v579")
    if backup.exists() and backup.read_text(encoding="utf-8") != source:
        raise ValueError("startup backup differs")
    if not backup.exists():
        shutil.copy2(startup, backup)
    startup.write_text(source.replace(anchor, addition + anchor), encoding="utf-8")
    print("INDEXER_GROUP_STARTUP_WIRED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--wire-startup", type=Path)
    args = parser.parse_args()
    if args.wire_startup:
        wire(args.wire_startup)
    elif args.lmcache_root and args.helper:
        install(args.lmcache_root, args.helper)
    else:
        parser.error("supply --wire-startup or both --lmcache-root and --helper")
