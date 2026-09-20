# SPDX-License-Identifier: Apache-2.0
"""Build the isolated counted-read extension without replacing live c_ops."""
import os
from pathlib import Path
from torch.utils.cpp_extension import load

source = Path('/tmp/device_count_v137')
build = source/'build'
build.mkdir(exist_ok=True)
os.environ['MAX_JOBS'] = '2'
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
module = load(name='tutti_device_count_v137', sources=[
    str(source/'csrc/tutti_kv_ops.cu'), str(source/'device_count_bindings_v137.cpp'),
], extra_include_paths=[str(source/'csrc')], build_directory=str(build),
    extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-lineinfo'], verbose=True)
print('COUNTED_READ_EXTENSION_BUILT', module.__file__, flush=True)


