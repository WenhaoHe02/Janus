# SPDX-License-Identifier: Apache-2.0
"""Deploy only two launch-cap helpers into an existing stable overlay."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil


def functions(source: str) -> dict[str, tuple[int, int, str]]:
    """Locate exact top-level helpers, retaining every unrelated source byte."""
    lines = source.splitlines(keepends=True)
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_proxy_query_tile_rows",
            "_proxy_query_launch_rows",
        ):
            result[node.name] = (
                node.lineno - 1,
                node.end_lineno,
                "".join(lines[node.lineno - 1 : node.end_lineno]),
            )
    if len(result) != 2:
        raise ValueError("missing exact launch-cap helper anchors")
    return result


def install(root: Path, replacement: Path) -> None:
    """Patch a fresh worker source before import; no live worker mutation."""
    target = root / "v1/csa_prefill_cp_scorer.py"
    original = target.read_text()
    before, after = functions(original), functions(replacement.read_text())
    lines = original.splitlines(keepends=True)
    for name, (start, end, old) in sorted(
        before.items(), key=lambda item: item[1][0], reverse=True
    ):
        if old == after[name][2]:
            continue
        if "must be 0 or 8192" not in old:
            raise ValueError(f"unexpected deployed helper: {name}")
        lines[start:end] = [after[name][2]]
    updated = "".join(lines)
    compile(updated, str(target), "exec")
    target.write_text(updated)
    print("PROXY_FULL_QUERY_KERNEL_TILE_CAP_V522_INSTALLED", flush=True)


def wire(startup: Path) -> None:
    """Add an opt-in installer to startup, preserving the original backup."""
    original = startup.read_text()
    marker = "# PROXY_FULL_QUERY_TILE_V522"
    if marker in original:
        print("QUERY_TILE_STARTUP_ALREADY_WIRED", flush=True)
        return
    anchor = (
        "exec ${LMCACHE_EXEC_PREFIX:-} python3 -m vllm.entrypoints.openai.api_server"
    )
    if original.count(anchor) != 1:
        raise ValueError("startup final exec anchor differs")
    addition = """# PROXY_FULL_QUERY_TILE_V522
if [ "${LMCACHE_CSA_PROXY_QUERY_TILE_ROWS:-0}" = "2048" ] || \\
   [ "${LMCACHE_CSA_PROXY_QUERY_TILE_ROWS:-0}" = "4096" ]; then
  python3 /tmp/install_proxy_tile_cap_v522.py --lmcache-root "$LMCACHE_SITE"
fi
"""
    if "/tmp/install_prefetch_ctas_v520.py" not in original:
        addition += """# Same communicator-local option as the timed deployment.
if [ "${LMCACHE_SSD_TP_GATHER_MAX_CTAS:-0}" -gt 0 ]; then
  python3 /tmp/install_prefetch_ctas_v520.py --lmcache-root "$LMCACHE_SITE"
fi
"""
    backup = startup.with_name(startup.name + ".pre_v522")
    if backup.exists():
        if backup.read_bytes() != startup.read_bytes():
            raise ValueError(
                "backup differs without installed marker; inspect manually"
            )
    else:
        shutil.copy2(startup, backup)
    startup.write_text(original.replace(anchor, addition + anchor))
    print("QUERY_TILE_STARTUP_WIRED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path)
    parser.add_argument("--wire-startup", type=Path)
    parser.add_argument(
        "--source", type=Path, default=Path("/tmp/proxy_query_tile_source_v522.py")
    )
    args = parser.parse_args()
    if args.wire_startup:
        wire(args.wire_startup)
    else:
        install(args.lmcache_root, args.source)
