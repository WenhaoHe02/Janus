# SPDX-License-Identifier: Apache-2.0
"""Prepare isolated startup copies and a TTFT-only plan; never launch GPUs."""

# Standard
import hashlib
import json
from pathlib import Path

# Local
from flash_ttft_ablation_v1185 import ARMS


def build_sources(originals: dict[str, str], destination: str) -> dict[str, str]:
    """Generate independent startups with strict original-source anchor checks.

    Args:
        originals: Contents of the four normal v916 startup files.
        destination: Absolute directory for the generated startup chain.
    Returns:
        Filename-to-source mapping. Both installers run before worker imports.
    Raises:
        ValueError: The destination or a reviewed source anchor differs.
    """
    if destination != '/tmp/flash_ttft_ablation_startup_v1185':
        raise ValueError('Unexpected destination')
    result = dict(originals)
    for name, child in (
        ('startup_ds_serving_v916.sh', 'exec_ds_serving_v916.py'),
        ('exec_ds_serving_v916.py', 'exec_ds_serving_v911.py'),
        ('exec_ds_serving_v911.py', 'exec_ds_serving_v890.py'),
    ):
        anchor = '/tmp/' + child
        if result[name].count(anchor) != 1:
            raise ValueError(f'Changed startup chain: {name}')
        result[name] = result[name].replace(anchor, destination + '/' + child)
    name = 'exec_ds_serving_v890.py'
    replacements = {
        'from serving_ds_configuration_v890 import environment':
            'from flash_ttft_ablation_v1185 import environment',
        '    flags = environment(arm)':
            "    if arm != 'k_local':\n        raise ValueError('Prediction stays enabled')\n"
            "    flags = environment(os.environ['LMCACHE_TTFT_ABLATION_ARM_V1185'])",
        "    print('DS_SERVING_V890_FINAL_FLAGS', flags, flush=True)":
            "    if flags['LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139'] == '1':\n"
            "        subprocess.run([sys.executable, '/tmp/install_unpacked_submit_v1139.py',\n"
            "                        '--lmcache-root', str(root)], check=True)\n"
            "    if flags['LMCACHE_PACKING_ABLATION_GPU_V1133'] == '1':\n"
            "        subprocess.run([sys.executable, '/tmp/install_gpu_packing_v1133.py',\n"
            "                        '--lmcache-root', str(root)], check=True)\n"
            "    print('DS_SERVING_V890_FINAL_FLAGS', flags, flush=True)",
    }
    for before, after in replacements.items():
        if result[name].count(before) != 1:
            raise ValueError(f'Changed executor anchor: {before}')
        result[name] = result[name].replace(before, after)
    for arm in ARMS:
        result[arm + '.sh'] = (
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'export PYTHONPATH="/tmp:${PYTHONPATH:-}"\n'
            f'export LMCACHE_TTFT_ABLATION_ARM_V1185={arm}\n'
            f'exec bash {destination}/startup_ds_serving_v916.sh\n'
        )
    for name, source in result.items():
        if name.endswith('.py'):
            compile(source, name, 'exec')
    return result


def main() -> None:
    """Write task-owned startup files and manifests without touching services."""
    target = Path('/tmp/flash_ttft_ablation_startup_v1185')
    names = ('startup_ds_serving_v916.sh', 'exec_ds_serving_v916.py',
             'exec_ds_serving_v911.py', 'exec_ds_serving_v890.py')
    original = {name: (Path('/tmp') / name).read_text() for name in names}
    generated = build_sources(original, str(target))
    for name, source in generated.items():
        path = target / name
        if path.exists() and path.read_text() != source:
            raise FileExistsError(path)
    target.mkdir(exist_ok=True)
    for name, source in generated.items():
        (target / name).write_text(source)
    proof = dict(launched=False, original_scripts_unchanged=True,
                 generated_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in generated.items()},
                 source_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in original.items()},
                 prediction_arms=['baseline', 'unsharded'],
                 packing_arms=['baseline', 'no_read_cpu', 'no_read_gpu'],
                 metric='TTFT', measured_output_tokens=1, arrival_rate=None)
    (target / 'PREPARED.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
