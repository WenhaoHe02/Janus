# SPDX-License-Identifier: Apache-2.0
"""Install an optional launch-only binding, leaving normal calls unchanged."""

# Standard
import argparse
from pathlib import Path


def install(root: Path) -> None:
    """Add separate held-GIL symbols to the container's installed loader.

    Args:
        root: Installed LMCache package in a disposable experiment container.

    Raises:
        RuntimeError: The expected loader class anchor is missing or ambiguous.
    """
    path = root / "v1/gpu_connector/tutti_direct_loader.py"
    source = path.read_text()
    if "COUNT_SUBMIT_HOLDGIL_V373" in source:
        return
    if source.count("class TuttiDirectLoader:") != 1:
        raise RuntimeError("unexpected loader class anchor")
    binding = """# COUNT_SUBMIT_HOLDGIL_V373: kernels remain frozen v137.
if os.getenv("LMCACHE_EXPERIMENT_COUNTED_BINDING_V373", "0") == "1":
    import importlib.util as _holdgil_util
    import sys as _holdgil_sys

    _holdgil_name = "count_submit_holdgil_v373"
    _holdgil_extension = _holdgil_sys.modules.get(_holdgil_name)
    if _holdgil_extension is None:
        _holdgil_spec = _holdgil_util.spec_from_file_location(
            _holdgil_name,
            "/tmp/count_submit_holdgil_v373/count_submit_holdgil_v373.so",
        )
        if _holdgil_spec is None or _holdgil_spec.loader is None:
            raise RuntimeError("v373 counted binding unavailable")
        _holdgil_extension = _holdgil_util.module_from_spec(_holdgil_spec)
        _holdgil_sys.modules[_holdgil_name] = _holdgil_extension
        _holdgil_spec.loader.exec_module(_holdgil_extension)
    for _holdgil_symbol in (
        "tutti_read_indexed_device_count_hold_gil",
        "tutti_read_indexed_segments_device_count_hold_gil",
    ):
        setattr(_c_ops, _holdgil_symbol, getattr(_holdgil_extension, _holdgil_symbol))


"""
    source = source.replace(
        "class TuttiDirectLoader:", binding + "class TuttiDirectLoader:", 1
    )
    compile(source, str(path), "exec")
    path.write_text(source)
    print("COUNT_SUBMIT_HOLDGIL_V373_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)
