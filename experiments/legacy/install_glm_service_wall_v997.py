# SPDX-License-Identifier: Apache-2.0
"""Install public host-wall scopes without a Python call profiler."""

# Standard
import argparse
from pathlib import Path
import shutil


def main() -> None:
    """Back up the fresh sink source and add diagnostic-only public wrappers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / 'v1/glm_dsa_predictive_prefetch.py'
    source = target.read_text()
    backup = target.with_suffix('.py.before_service_wall_v997')
    if backup.exists():
        raise RuntimeError('Require a fresh service-wall deployment')
    patched = source + '''
# PUBLIC_SERVICE_WALL_V997
if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_SERVICE_WALL_DIR'):
    from pathlib import Path as ServiceWallPath
    from lmcache.integration.vllm.pipeline_cpu_diagnostic_v993 import install_pipeline_profiling
    from lmcache.v1.csa_attention_kv_prefetch_manager import CSAAttentionKVPrefetchManager
    from lmcache.v1.ssd_tp_sharded_prefetch import TorchDistributedShardGather
    service_wall_recorder = install_pipeline_profiling(
        GLMDSAPredictivePrefetchManager, GLMDSAPhysicalPrefetchSink,
        ServiceWallPath(os.environ['LMCACHE_SERVING_DIAGNOSTIC_SERVICE_WALL_DIR']),
        python_profile=False,
    )
    for service_method in (
        'fire_predicted_reads', 'launch_layer_gather',
        'finalize_deferred_shard_gather', 'prepare_topk_block_selection',
        'submit_selection_miss_reads', 'submit_miss_reads', 'wait_for_layer',
    ):
        setattr(CSAAttentionKVPrefetchManager, service_method,
                service_wall_recorder.wrap(
                    getattr(CSAAttentionKVPrefetchManager, service_method), service_method))
    TorchDistributedShardGather.gather_owner_rows_gpu_into = service_wall_recorder.wrap(
        TorchDistributedShardGather.gather_owner_rows_gpu_into, 'owner_gather_host')
    CSAAttentionKVPrefetchManager.set_kv_forward_phase = service_wall_recorder.wrap_forward_phase(
        CSAAttentionKVPrefetchManager.set_kv_forward_phase)
'''
    helper = args.helper.read_text()
    compile(helper, str(args.helper), 'exec')
    compile(patched, str(target), 'exec')
    shutil.copy2(target, backup)
    (args.lmcache_root / 'integration/vllm/pipeline_cpu_diagnostic_v993.py').write_text(helper)
    target.write_text(patched)
    loader = args.lmcache_root / 'v1/gpu_connector/tutti_direct_loader.py'
    loader_source = loader.read_text()
    loader_patched = loader_source + '''
# LIGHT_SERVICE_LOADER_WALL_V1005
if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_LIGHT_SERVICE_WALL') == '1':
    from lmcache.integration.vllm.pipeline_cpu_diagnostic_v993 import wrap_active_pipeline_method
    for loader_method, loader_category in (
        ('ensure_lba_cache', 'lba_ensure'),
        ('load_chunks_to_hbm', 'loader_chunks'),
        ('load_indexed_chunks_to_hbm', 'loader_indexed'),
        ('finish_device_counted_reads', 'counted_finish'),
        ('enqueue_device_counted_read', 'counted_enqueue'),
    ):
        setattr(TuttiDirectLoader, loader_method, wrap_active_pipeline_method(
            getattr(TuttiDirectLoader, loader_method), loader_category))
'''
    compile(loader_patched, str(loader), 'exec')
    shutil.copy2(loader, loader.with_suffix('.py.before_light_service_v1005'))
    loader.write_text(loader_patched)
    print('SERVICE_WALL_V997_INSTALLED_WITHOUT_CPROFILE', flush=True)


if __name__ == '__main__':
    main()
