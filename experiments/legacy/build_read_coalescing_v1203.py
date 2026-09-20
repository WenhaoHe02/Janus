# SPDX-License-Identifier: Apache-2.0
"""Build isolated range-coalescing READ kernels; never install into a server."""

# Standard
import argparse
import hashlib
import json
import os
from pathlib import Path


DECLARATION = """
void read_coalesced_v1203(
 int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
 int, int, int64_t, at::Tensor, at::Tensor, int64_t, at::Tensor, at::Tensor,
 at::Tensor, at::Tensor, int, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
 int, int, bool, int64_t, int64_t, bool);
"""


def extension_sources(source: str, binding: str, addition: str) -> tuple[str, str]:
    """Append the isolated API while retaining original submit/poll functions.

    Args:
        source: Audited deployed CUDA source.
        binding: Matching original pybind translation unit.
        addition: Range-coalescing CUDA translation-unit suffix.
    Returns:
        CUDA and pybind source strings, without changing input files.
    Raises:
        ValueError: Expected symbols or the unique module declaration are absent.
    """
    marker = 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {'
    required = ('k_poll_batch(', 'k_validate_counted_read(', 'make_qd(')
    if binding.count(marker) != 1 or any(name not in source for name in required):
        raise ValueError('Unexpected deployed source; audit before building')
    if 'read_coalesced_v1203' in source or 'launch_coalesced_v1203' in addition:
        raise ValueError('Duplicate extension or unresolved launcher')
    return source + '\n' + addition, binding.replace(
        marker, DECLARATION + marker +
        '\n  m.def("read_coalesced", &read_coalesced_v1203, '
        'py::call_guard<py::gil_scoped_release>());', 1)


def main() -> None:
    """Build a separate module only with an explicit idle-measurement assertion.

    The caller must verify no GPU experiment is timed before passing the flag.
    This command does not install, register buffers, or access any SSD queues.
    Existing generated files are immutable; a changed build needs a new root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-idle-confirmed', action='store_true')
    args = parser.parse_args()
    if not args.measurement_idle_confirmed:
        raise RuntimeError('Do not compile during timed GPU experiments')
    from torch.utils.cpp_extension import load

    original = Path('/tmp/device_count_v137')
    root = Path('/tmp/read_coalescing_v1203')
    suffix = Path(__file__).with_name('read_range_coalescing_v1203.cu').read_text()
    source = (original / 'csrc/tutti_kv_ops.cu').read_text()
    binding = (original / 'device_count_bindings_v137.cpp').read_text()
    generated, bindings = extension_sources(source, binding, suffix)
    root.mkdir(exist_ok=True)
    for name, content in (
        ('tutti_kv_ops.cu', generated), ('bindings.cpp', bindings),
        ('tutti_kv_ops.cuh', (original / 'csrc/tutti_kv_ops.cuh').read_text()),
    ):
        path = root / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(path)
        if not path.exists():
            path.write_text(content)
    build = root / 'build'
    build.mkdir(exist_ok=True)
    os.environ['MAX_JOBS'] = '2'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    module = load(
        name='tutti_read_coalescing_v1203',
        sources=[str(root / 'tutti_kv_ops.cu'), str(root / 'bindings.cpp')],
        extra_include_paths=[str(root)], build_directory=str(build),
        extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-lineinfo'], verbose=True,
    )
    proof = dict(
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
        module=module.__file__,
        module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        installed=False, device_limits_verified=False, readback_verified=False,
    )
    (root / 'BUILD.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
