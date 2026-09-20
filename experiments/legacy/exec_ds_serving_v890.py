# SPDX-License-Identifier: Apache-2.0
"""Install DS serving fixes after overlays, without GLM prediction hooks."""

# Standard
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# Local
from serving_ds_configuration_v890 import environment


def main() -> None:
    """Validate native assets and install opt-in controls before model imports."""
    spec = importlib.util.find_spec('lmcache')
    if spec is None or spec.origin is None or len(sys.argv) < 2:
        raise RuntimeError('Missing package or server command')
    storage = os.getenv('LMCACHE_STORAGE_MODE', 'tutti')
    arm = {'ssd': 'ssd_only', 'cpu': 'dram'}.get(storage)
    if arm is None:
        arm = 'k_local' if os.getenv('LMCACHE_ABLATION_CSA_ATTENTION_KV_FILTER') == '1' else 'no_prefetch'
    flags = environment(arm)
    os.environ.update(flags)
    import yaml

    config_path = Path(os.environ['LMCACHE_CONFIG_FILE'])
    config = yaml.safe_load(config_path.read_text())
    config['lookup_server_worker_ids'] = list(range(8))
    if arm == 'dram':
        if not config.get('local_cpu') or config.get('local_disk'):
            raise ValueError('DRAM requires CPU-only storage')
        config.update(max_local_cpu_size=128, reserve_local_cpu_size=2,
                      internal_api_server_enabled=True,
                      internal_api_server_host='127.0.0.1',
                      internal_api_server_port_start=6999)
        config.setdefault('extra_config', {})['first_rank_max_local_cpu_size'] = 128
        # Keep the DRAM arm's loaded groups identical to the sparse SSD arm;
        # otherwise it reads every dense KV group and is not a storage-tier
        # comparison. The backend remains LocalCPUBackend (no disk I/O).
        config['extra_config']['dsv4_optimized_kv'] = True
        config['extra_config']['dsv4_optimized_tail_tokens'] = 256
    config_path.write_text(yaml.safe_dump(config))
    root = Path(spec.origin).parent
    assets = json.loads(Path('/tmp/dsv4_native_assets_v826/manifest.json').read_text())
    hashes = {name: hashlib.sha256((Path('/tmp/dsv4_tokenizer_snapshot') / name).read_bytes()).hexdigest()
              for name in ('tokenizer.json', 'tokenizer_config.json')}
    if hashes != assets['flash']['hashes'] or hashes != assets['pro']['hashes']:
        raise ValueError('DS tokenizer snapshot differs')
    for installer, extra in (
        ('install_ds_store_boundary_v737.py', ['--helper', '/tmp/dsv4_store_completion_v737.py']),
        ('install_decode_write_io_v809.py', []),
    ):
        subprocess.run([sys.executable, '/tmp/' + installer, '--lmcache-root', str(root), *extra], check=True)
    subprocess.run([sys.executable, '/tmp/install_mapped_control_v867.py'], check=True)
    (root / 'ds_serving_v890_provenance.json').write_text(json.dumps({
        'flags': flags, 'tokenizer_hashes': hashes,
        'model_config_sha256': hashlib.sha256(Path('/pro_model/config.json').read_bytes()).hexdigest()}))
    print('DS_SERVING_V890_FINAL_FLAGS', flags, flush=True)
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()
