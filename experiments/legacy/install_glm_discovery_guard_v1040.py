# SPDX-License-Identifier: Apache-2.0
"""Apply the model-type GLM discovery guard after legacy deployment overlays."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def patch_adapter(source: str) -> str:
    """Skip DS decoder enumeration only for a configured GLM model.

    Args:
        source: Complete unmodified deployed adapter source.
    Returns:
        Adapter with model-type capture before connector initialization and
        an early GLM return before any DS registry/GC access. Idempotent.
    Raises:
        ValueError: Required function/class structure changed or probe is present.
    """
    if "GLM_MODEL_DISCOVERY_GUARD_V1040" in source:
        return source
    if "DISCOVERY_PROBE_V1039" in source:
        raise ValueError("Cannot promote a diagnostic adapter as production")
    tree = ast.parse(source)
    discoveries = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_deepseek_decoder_layers"
    ]
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Impl"
    ]
    if len(discoveries) != 1 or len(classes) != 1:
        raise ValueError("Required adapter structure absent")
    constructors = [
        n
        for n in classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    ]
    if len(constructors) != 1:
        raise ValueError("Expected one connector constructor")
    discovery = discoveries[0]
    constructor = constructors[0]

    # Insert after a docstring, preserving Python docstrings and global statements.
    def body_start(node: ast.FunctionDef) -> int:
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first.end_lineno
        return first.lineno - 1

    edits = [
        (
            discovery.lineno - 1,
            "# GLM_MODEL_DISCOVERY_GUARD_V1040\n_PREFETCH_MODEL_TYPE_V1040 = None\n\n",
        ),
        (
            body_start(discovery),
            '    if _PREFETCH_MODEL_TYPE_V1040 == "glm_moe_dsa":\n        return []\n',
        ),
        (
            body_start(constructor),
            "        global _PREFETCH_MODEL_TYPE_V1040\n"
            "        _PREFETCH_MODEL_TYPE_V1040 = getattr(\n"
            '            getattr(vllm_config.model_config, "hf_config", None),\n'
            '            "model_type", None)\n',
        ),
    ]
    lines = source.splitlines(keepends=True)
    for index, addition in sorted(edits, reverse=True):
        lines[index:index] = [addition]
    result = "".join(lines)
    compile(result, "guarded_adapter", "exec")
    return result


def main() -> None:
    """Back up and patch the exact deployed adapter, recording source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()
    path = args.lmcache_root / "integration/vllm/vllm_v1_adapter.py"
    original = path.read_text()
    patched = patch_adapter(original)
    if patched == original:
        print("GLM_MODEL_DISCOVERY_GUARD_V1040_ALREADY_PRESENT", flush=True)
        return
    backup = path.with_suffix(".py.before_discovery_guard_v1040")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(path, backup)
    path.write_text(patched)
    print(
        json.dumps(
            dict(
                marker="GLM_MODEL_DISCOVERY_GUARD_V1040",
                before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                after_sha256=hashlib.sha256(patched.encode()).hexdigest(),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
