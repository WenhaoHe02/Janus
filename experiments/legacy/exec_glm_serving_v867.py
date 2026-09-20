# SPDX-License-Identifier: Apache-2.0
"""Install serving code once after legacy overlays and before server import."""

# Standard
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

# Local
from serving_glm_configuration_v766 import environment


def main() -> None:
    """Preserve server argv, apply explicit flags, and fail before model loading."""
    if os.getenv('LMCACHE_SERVING_EXPERIMENT') != '1':
        raise RuntimeError('explicit serving startup not selected')
    flags = environment(
        os.environ['LMCACHE_SERVING_ARM'],
        stage_timing_diagnostic=(
            os.getenv('LMCACHE_SERVING_DIAGNOSTIC_STAGE_TIMING') == '1'
        ),
    )
    os.environ.update(flags)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_ALIGNED_PACKET') == '1':
        if os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Aligned packet comparison requires K')
        os.environ['LMCACHE_CSA_OWNER_KV_PACKET'] = '1'
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_SERVICE_WALL_DIR'):
        if os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Owner service-wall diagnostic requires K')
        tutti_timing = '0' if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_LIGHT_SERVICE_WALL') == '1' else '1'
        flags['LMCACHE_TUTTI_PROFILE'] = tutti_timing
        os.environ['LMCACHE_TUTTI_PROFILE'] = tutti_timing
    compact_diagnostic = os.getenv('LMCACHE_SERVING_DIAGNOSTIC_COMPACT_WIDTH')
    if compact_diagnostic is not None:
        if compact_diagnostic not in ('0', '1') or os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Compact-width diagnostic requires K and a boolean value')
        os.environ['LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH'] = compact_diagnostic
        flags['LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH'] = compact_diagnostic
    os.environ["LMCACHE_TUTTI_MAPPED_CONTROL"] = "1"
    spec = importlib.util.find_spec('lmcache')
    if spec is None or spec.origin is None or len(sys.argv) < 2:
        raise RuntimeError('missing package or original server command')
    root = str(Path(spec.origin).parent)
    if os.getenv('LMCACHE_SERVING_FIX_BACKEND_ALIAS') == '1':
        subprocess.run([sys.executable, '/tmp/install_backend_alias_v995.py',
                        '--lmcache-root', root], check=True)
    for script, arguments in (
        ('install_glm_stable_startup_v764.py', ['--originals', '/tmp', '--helper', '/tmp/serving_counted_ops_v764.py']),
        ('install_decode_write_io_v809.py', []),
        ('install_logical_base_v916.py', []),
    ):
        subprocess.run([sys.executable, '/tmp/' + script, '--lmcache-root', root, *arguments], check=True)
    subprocess.run([sys.executable, "/tmp/install_mapped_control_v867.py"], check=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_DEVICE_OWNER') == '1':
        if os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Device owner candidate requires K')
        os.environ['LMCACHE_GLM_DSA_DEVICE_OWNER_READ'] = '1'
        subprocess.run([sys.executable, '/tmp/install_glm_device_owner_v1006.py',
                        '--lmcache-root', root, '--helper', '/tmp/owner_device_read_v1006.py',
                        '--hook', '/tmp/glm_device_owner_hook_v1006.py'], check=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_ALIGNED_PACKET') == '1':
        subprocess.run([sys.executable, '/tmp/install_aligned_packet_v999.py',
                        '--lmcache-root', root, '--transport',
                        '/tmp/ssd_tp_sharded_prefetch_aligned_v999.py', '--codec',
                        '/tmp/owner_kv_packet_aligned_v999.py'], check=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_SERVICE_WALL_DIR'):
        subprocess.run([sys.executable, '/tmp/install_glm_service_wall_v997.py',
                        '--lmcache-root', root, '--helper',
                        '/tmp/pipeline_cpu_diagnostic_v993.py'], check=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_FIXED_BOUNDARY') == '1':
        if os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Fixed-boundary candidate requires K')
        os.environ['LMCACHE_GLM_DSA_FIXED_BOUNDARY_GATHER'] = '1'
        subprocess.run([sys.executable, '/tmp/install_glm_fixed_boundary_v1011.py',
                        '--lmcache-root', root, '--scheduler', '/tmp/fixed_boundary_gather_v1011.py',
                        '--hooks', '/tmp/glm_fixed_boundary_v1011.py'], check=True)
    dependency_mode = os.getenv('LMCACHE_GLM_PREDICTION_IO_DEPENDENCY', 'legacy')
    if dependency_mode != 'legacy':
        if dependency_mode not in ('capture', 'source') or os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Prediction dependency diagnostic requires K and a supported mode')
        subprocess.run([sys.executable, '/tmp/install_prediction_dependency_v996.py',
                        '--lmcache-root', root, '--helper',
                        '/tmp/prediction_io_dependency_v996.py'], check=True)
        print('PREDICTION_IO_DEPENDENCY_MODE', dependency_mode, flush=True)
    cta_limit = int(os.getenv('LMCACHE_SERVING_DIAGNOSTIC_PREFETCH_CTAS', '0'))
    if cta_limit:
        if cta_limit < 1 or os.environ['LMCACHE_SERVING_ARM'] != 'k_local':
            raise ValueError('Prefetch CTA diagnostic requires the K arm and positive limit')
        os.environ['LMCACHE_SSD_TP_GATHER_MAX_CTAS'] = str(cta_limit)
        os.environ['LMCACHE_SSD_TP_GATHER_MAX_NCHANNELS'] = '0'
        subprocess.run([sys.executable, '/tmp/install_prefetch_ctas_v991.py',
                        '--lmcache-root', root, '--helper',
                        '/tmp/prefetch_nccl_config_v991.py'], check=True)
        print('PREFETCH_CTA_DIAGNOSTIC_LIMIT', cta_limit, flush=True)
    if os.getenv("LMCACHE_SERVING_DIAGNOSTIC_CPU_RETRIEVE_DIR"):
        subprocess.run([sys.executable, "/tmp/install_retrieve_cpu_diagnostic_v988.py",
                        "--lmcache-root", root, "--helper",
                        "/tmp/retrieve_cpu_diagnostic_v988.py"], check=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_PIPELINE_CPU_DIR'):
        subprocess.run([sys.executable, '/tmp/install_pipeline_cpu_v993.py',
                        '--lmcache-root', root, '--helper',
                        '/tmp/pipeline_cpu_diagnostic_v993.py'], check=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_CAPABILITY_AUDIT') == '1':
        subprocess.run([sys.executable, '/tmp/install_device_capability_audit_v994.py',
                        '--lmcache-root', root], check=True)
    print("MAPPED_CONTROL_V867_ENABLED", flush=True)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_STEP_NVTX') == '1':
        if not os.getenv('LMCACHE_SERVING_DIAGNOSTIC_NSYS'):
            raise ValueError('Step NVTX requires a diagnostic Nsight session')
        vllm_spec = importlib.util.find_spec('vllm')
        if vllm_spec is None or vllm_spec.origin is None:
            raise RuntimeError('Missing vLLM package for step profiling')
        subprocess.run([sys.executable,'/tmp/install_model_step_nvtx_v1019.py',
            '--vllm-root',str(Path(vllm_spec.origin).parent),'--lmcache-root',root,
            '--helper','/tmp/model_step_nvtx_v1019.py'],check=True)
    if os.getenv('LMCACHE_GLM_DISCOVERY_DIAGNOSTIC_DIR'):
        if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_STAGE_TIMING') != '1':
            raise ValueError('Discovery probe requires explicit diagnostic mode')
        subprocess.run([sys.executable, '/tmp/install_discovery_probe_v1039.py',
                        '--lmcache-root', root], check=True)
    # Model dispatch is a correctness-preserving fix for every GLM path,
    # not a performance-only K/NP option. Only explicit diagnostic replay opts out.
    os.environ.setdefault('LMCACHE_SERVING_FIX_GLM_DISCOVERY', '1')
    if os.environ['LMCACHE_SERVING_FIX_GLM_DISCOVERY'] == '1':
        if os.getenv('LMCACHE_GLM_DISCOVERY_DIAGNOSTIC_DIR'):
            raise ValueError('Do not mix permanent repair with discovery A/B probe')
        subprocess.run([sys.executable, '/tmp/install_glm_discovery_guard_v1040.py',
                        '--lmcache-root', root], check=True)
    print('GLM_SERVING_V766_FINAL_FLAGS', flags, flush=True)
    diagnostic_session = os.getenv('LMCACHE_SERVING_DIAGNOSTIC_NSYS')
    if diagnostic_session:
        profiler = '/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
        print('DIAGNOSTIC_ONLY_NSYS_SESSION', diagnostic_session, flush=True)
        os.execvp(profiler, [profiler, 'launch', '--session-new', diagnostic_session,
                  '--trace=cuda,nvtx,osrt', '--cuda-event-trace=false',
                  '--cuda-flush-interval=5000', *sys.argv[1:]])
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()
