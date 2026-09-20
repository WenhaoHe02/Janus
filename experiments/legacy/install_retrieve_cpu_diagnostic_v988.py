# SPDX-License-Identifier: Apache-2.0
"""Install opt-in profiling around the public engine retrieve interface."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil


def main() -> None:
    """Validate the public method, preserve source, and install before imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / "v1/cache_engine.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheEngine"
    ]
    if len(classes) != 1 or not any(
        isinstance(node, ast.FunctionDef) and node.name == "retrieve"
        for node in classes[0].body
    ):
        raise RuntimeError("Public LMCacheEngine.retrieve contract missing")
    marker = "# RETRIEVE_CPU_DIAGNOSTIC_V988"
    if marker in source:
        raise RuntimeError("Diagnostic already installed; require a fresh deployment")
    patched = (
        source
        + "\n"
        + marker
        + """
if os.getenv("LMCACHE_SERVING_DIAGNOSTIC_CPU_RETRIEVE_DIR"):
    from pathlib import Path as DiagnosticPath
    from lmcache.integration.vllm.retrieve_cpu_diagnostic_v988 import (
        wrap_retrieve_for_diagnostics,
    )
    LMCacheEngine.retrieve = wrap_retrieve_for_diagnostics(
        LMCacheEngine.retrieve,
        DiagnosticPath(os.environ["LMCACHE_SERVING_DIAGNOSTIC_CPU_RETRIEVE_DIR"]),
    )
"""
    )
    compile(patched, str(target), "exec")
    helper = args.helper.read_text(encoding="utf-8")
    compile(helper, str(args.helper), "exec")
    backup = target.with_name(target.name + ".before_cpu_diagnostic_v988")
    if backup.exists():
        raise RuntimeError("Existing diagnostic backup; refusing overwrite")
    shutil.copy2(target, backup)
    (args.lmcache_root / "integration/vllm/retrieve_cpu_diagnostic_v988.py").write_text(
        helper, encoding="utf-8"
    )
    target.write_text(patched, encoding="utf-8")
    print("RETRIEVE_CPU_DIAGNOSTIC_V988_INSTALLED_NONPRODUCTION", flush=True)


if __name__ == "__main__":
    main()
