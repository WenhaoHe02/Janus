# SPDX-License-Identifier: Apache-2.0
"""Compile the recovered counted-read CUDA extension before model startup."""

# Standard
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def source_fingerprint(source: Path) -> dict[str, str]:
    """Hash the exact CUDA/header/binding inputs; reject missing or linked files.

    Args:
        source: Recovered device_count_v137 source directory.
    Returns:
        Relative filenames mapped to SHA256 values.
    Raises:
        ValueError: A required input is missing, linked, or empty.
    """
    result = {}
    for name in (
        "csrc/tutti_kv_ops.cu",
        "csrc/tutti_kv_ops.cuh",
        "device_count_bindings_v137.cpp",
    ):
        path = source / name
        if not path.is_file() or path.is_symlink() or not path.stat().st_size:
            raise ValueError(f"Missing or unsafe native source: {name}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build(source: Path, cache: Path, arch: str, jobs: int) -> dict:
    """Compile counted-read bindings with a source/toolchain-specific cache key.

    Args:
        source: Recovered CUDA source directory.
        cache: Writable compilation cache, not a running model's extension path.
        arch: Explicit TORCH_CUDA_ARCH_LIST value, such as 9.0 for H200.
        jobs: Positive compiler concurrency limit.
    Returns:
        Build receipt including module hash, toolchain and exported symbols.
    Raises:
        ValueError: Invalid configuration or missing sources.
        RuntimeError: CUDA build support or expected symbols are absent.
    Notes:
        Does not access NVMe, install a driver, or patch a running worker.
        The deployment must load the reported module path, not a historical .so.
    """
    if not arch.strip() or jobs < 1:
        raise ValueError("Explicit architecture and positive jobs required")
    sources = source_fingerprint(source)
    # Third Party
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None or torch.version.cuda is None:
        raise RuntimeError("CUDA-enabled PyTorch and CUDA toolkit required")
    identity = dict(
        sources=sources,
        arch=arch,
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        python=platform.python_version(),
        platform=platform.platform(),
        nvcc=subprocess.check_output(
            [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
        ).strip(),
        cxx=subprocess.check_output(
            [os.environ.get("CXX", "c++"), "--version"], text=True
        ).strip(),
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache.resolve() / key
    destination.mkdir(parents=True, exist_ok=True)
    previous = {
        name: os.environ.get(name) for name in ("MAX_JOBS", "TORCH_CUDA_ARCH_LIST")
    }
    try:
        os.environ.update(MAX_JOBS=str(jobs), TORCH_CUDA_ARCH_LIST=arch)
        module = load(
            name="tutti_device_count_v137",
            sources=[
                str(source.resolve() / name)
                for name in sources
                if not name.endswith(".cuh")
            ],
            extra_include_paths=[str(source.resolve() / "csrc")],
            build_directory=str(destination),
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2", "-lineinfo"],
            verbose=True,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    expected = (
        "tutti_read_indexed_device_count",
        "tutti_read_indexed_segments_device_count",
    )
    if any(not callable(getattr(module, name, None)) for name in expected):
        raise RuntimeError("Compiled module does not expose the counted-read API")
    module_path = Path(module.__file__)
    receipt = dict(
        identity=identity,
        module=str(module_path),
        module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        symbols=list(expected),
        device_io_tested=False,
    )
    (destination / "BUILD.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    """Build native I/O bindings without starting an inference workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--receipt", type=Path, help="Stable receipt path consumed by worker startup"
    )
    args = parser.parse_args()
    receipt = build(args.source, args.cache, args.arch, args.jobs)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(args.receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
