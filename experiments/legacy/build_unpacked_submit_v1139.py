# SPDX-License-Identifier: Apache-2.0
"""Build an isolated per-command-doorbell ablation from deployed Tutti source."""
# Standard
import hashlib
import json
import os
from pathlib import Path


def transform(source: str) -> str:
    """Change indexed SQ submission only; preserve commands, poll and scatter.

    One producer constructs each command then publishes its queue tail, without
    waiting for any completion. Both segmented and fixed-size counted reads use
    this kernel. The original extension is not modified or overwritten.
    Raises ValueError when the deployed kernel no longer has the audited shape.
    """
    start = source.index('template <bool Profiled, bool DeviceCounted = false>\n__global__ void k_submit_indexed_sgl_read(')
    end = source.index('\n// Poll one queue-depth-bounded', start)
    old = source[start:end]
    setup = '''  __shared__ uint16_t base_tail;
  if (threadIdx.x == 0) base_tail = *sq_tail_io;
  __syncthreads();

  for (int i = threadIdx.x; i < n_ios; i += blockDim.x) {'''
    if old.count(setup) != 1:
        raise ValueError('Unexpected indexed submit setup')
    body = old[:old.index('\n  // Every producer makes')]
    body = body.replace(setup, '''  // UNPACKED_SUBMIT_V1139: publish each command; do not wait per command.
  if (threadIdx.x != 0) return;
  const uint16_t base_tail = *sq_tail_io;
  for (int i = 0; i < n_ios; ++i) {''')
    comments = '''    // Unlike the legacy path, the controller sees no doorbell until the
    // whole batch is ready. Each thread exclusively owns its 64-byte,
    // naturally aligned slot, so four vector stores safely replace the
    // old 64 byte stores.'''
    body = body.replace(comments, '    // Keep the same vectorized SQE construction; only publication differs.')
    closing = '    slot->cdw12 = byte_len / NVME_LBS - 1u;\n  }\n'
    if body.count(closing) != 1:
        raise ValueError('Unexpected command fields/end')
    body = body.replace(closing, '''    slot->cdw12 = byte_len / NVME_LBS - 1u;
    __threadfence_system();
    const uint16_t new_tail = static_cast<uint16_t>(
        (static_cast<uint32_t>(base_tail) + static_cast<uint32_t>(i) + 1u) %
        qd.q_depth);
    *sq_tail_io = new_tail;
    if constexpr (Profiled) doorbell_timestamps[i] = tutti_globaltimer_ns();
    *qd.sq_db = new_tail;
  }
}
''')
    if '__syncthreads' in body or 'k_poll' in body:
        raise ValueError('Sequential submit contains an unexpected barrier/poll')
    return source[:start] + body + source[end:]


PROBE = r'''
// Pure GPU-memory probe: no BAR, real queue or SSD access.
void probe_submit_v1139(at::Tensor sq, at::Tensor doorbell, at::Tensor tail,
                       at::Tensor iovas, at::Tensor slbas, at::Tensor ids,
                       at::Tensor count, at::Tensor stamps, int qdepth,
                       int bytes, int64_t stream_ptr) {
  auto qd = make_qd(reinterpret_cast<int64_t>(sq.data_ptr()), 0,
                    reinterpret_cast<int64_t>(doorbell.data_ptr()), 0, qdepth, 1);
  k_submit_indexed_sgl_read<true, true><<<1, 256, 0,
      reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
      qd, reinterpret_cast<uint16_t*>(tail.data_ptr()), 1,
      reinterpret_cast<const uint64_t*>(iovas.data_ptr<int64_t>()), 65536,
      reinterpret_cast<const uint64_t*>(slbas.data_ptr<int64_t>()),
      ids.data_ptr<int64_t>(), bytes, ids.numel(),
      reinterpret_cast<uint64_t*>(stamps.data_ptr<int64_t>()),
      count.data_ptr<int32_t>(), slbas.numel(), nullptr);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "submit probe launch failed");
}
'''


def main() -> None:
    """Compile a separate module and save source hashes before worker startup."""
    from torch.utils.cpp_extension import load

    original = Path('/tmp/device_count_v137')
    root = Path('/tmp/unpacked_submit_v1139')
    root.mkdir(exist_ok=True)
    source = (original / 'csrc/tutti_kv_ops.cu').read_text()
    transformed = transform(source) + PROBE
    binding = (original / 'device_count_bindings_v137.cpp').read_text()
    declaration = '''void probe_submit_v1139(at::Tensor, at::Tensor, at::Tensor,
 at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int, int64_t);
'''
    binding = binding.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {', declaration +
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def("probe_submit", &probe_submit_v1139);')
    for name, content in [('tutti_kv_ops.cu', transformed), ('bindings.cpp', binding),
                          ('tutti_kv_ops.cuh', (original / 'csrc/tutti_kv_ops.cuh').read_text())]:
        path = root / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(path)
        path.write_text(content)
    build = root / 'build'
    build.mkdir(exist_ok=True)
    os.environ['MAX_JOBS'] = '2'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    module = load(name='tutti_unpacked_submit_v1139', sources=[str(root / 'tutti_kv_ops.cu'), str(root / 'bindings.cpp')],
        extra_include_paths=[str(root)], build_directory=str(build), extra_cflags=['-O2'],
        extra_cuda_cflags=['-O2', '-lineinfo'], verbose=True)
    proof = dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(transformed.encode()).hexdigest(), module=module.__file__,
        module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        command_count_unchanged=True, per_command_doorbell=True, per_command_wait=False)
    (root / 'BUILD.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
