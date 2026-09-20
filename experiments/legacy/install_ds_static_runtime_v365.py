# SPDX-License-Identifier: Apache-2.0
"""Bind verified native operators during ordinary worker module import."""

# Standard
import argparse
import ast
from pathlib import Path

MARKER = "JANUS_VERIFIED_COUNTED_RUNTIME"
BINDING = """# JANUS_VERIFIED_COUNTED_RUNTIME
if os.getenv("LMCACHE_DS_STATIC_COUNTED_RUNTIME", "0") == "1":
    from lmcache.integration.vllm.serving_counted_ops_v764 import ensure_counted_ops
    ensure_counted_ops()


"""


def transform(source: str) -> str:
    """Insert one verified module-import hook; reject old inlined binary loaders.

    Returns syntax-checked source, preserving loader methods. Raises ValueError
    on unknown source shape or old embedded binary-loading code. Idempotent for
    this exact hook, rather than accepting any file with a marker substring.
    """
    if MARKER in source:
        if source.count(BINDING) != 1:
            raise ValueError("Incomplete or modified native runtime hook")
        ast.parse(source)
        return source
    if "DS_STATIC_COUNTED_RUNTIME_V365" in source:
        raise ValueError(
            "Start from clean submission source, not an old patched loader"
        )
    anchor = "class TuttiDirectLoader:"
    if source.count(anchor) != 1:
        raise ValueError("Loader class layout changed")
    patched = source.replace(anchor, BINDING + anchor, 1)
    ast.parse(patched)
    return patched


def install(root: Path, helper: Path) -> None:
    """Install the startup helper and hook before importing model workers.

    Args:
        root: LMCache package in an isolated deployment.
        helper: Verified native-runtime Python source.
    Raises:
        ValueError: Input source shape is unsupported.
        OSError: A required source or destination is inaccessible.
    Does not load CUDA, build extensions, modify drivers or access SSDs.
    """
    path = root / "v1/gpu_connector/tutti_direct_loader.py"
    patched = transform(path.read_text())
    runtime = helper.read_text()
    ast.parse(runtime)
    destination = root / "integration/vllm/serving_counted_ops_v764.py"
    destination.write_text(runtime)
    path.write_text(patched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument(
        "--helper",
        type=Path,
        default=Path(__file__).with_name("serving_counted_ops_v764.py"),
    )
    args = parser.parse_args()
    install(args.lmcache_root, args.helper)
