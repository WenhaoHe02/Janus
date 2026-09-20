# SPDX-License-Identifier: Apache-2.0
"""Fail before weight loading if the installed connector/native imports differ."""

# Standard
import importlib
import json
import os
from pathlib import Path


def main() -> None:
    """Import, but never instantiate, the loader and model connector before serving."""
    package = importlib.import_module("lmcache")
    ops = importlib.import_module("lmcache.c_ops")
    if getattr(package, "c_ops", None) is not ops:
        raise RuntimeError("LMCache package and importable c_ops must be identical")
    native = importlib.import_module(
        "lmcache.integration.vllm.serving_counted_ops_v764"
    )
    native.ensure_counted_ops()
    importlib.import_module("lmcache.v1.cache_engine")
    importlib.import_module("lmcache.integration.vllm.vllm_v1_adapter")
    receipt = json.loads(Path(os.environ["JANUS_COUNTED_BUILD_RECEIPT"]).read_text())
    if os.environ.get("JANUS_LOADED_COUNTED_SHA256") != receipt["module_sha256"]:
        raise RuntimeError("Verified counted-read module was not installed")
    print(json.dumps(dict(imports_verified=True, model_loaded=False, ssd_opened=False)))


if __name__ == "__main__":
    main()
