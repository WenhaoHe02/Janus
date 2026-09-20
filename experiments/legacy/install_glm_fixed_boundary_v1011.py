# SPDX-License-Identifier: Apache-2.0
"""Install foreground scheduling through startup, never reload live workers."""

# Standard
import argparse
from pathlib import Path
import shutil


def main() -> None:
    """Validate the helper and wrapper before changing a fresh GLM module."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root',type=Path,required=True)
    parser.add_argument('--scheduler',type=Path,required=True)
    parser.add_argument('--hooks',type=Path,required=True)
    args=parser.parse_args()
    target=args.lmcache_root/'v1/glm_dsa_predictive_prefetch.py'
    backup=target.with_suffix('.py.before_fixed_boundary_v1011')
    if backup.exists():
        raise RuntimeError('Require a fresh fixed-boundary deployment')
    patched=target.read_text()+'''
# FIXED_FOREGROUND_GATHER_V1011
if os.getenv('LMCACHE_GLM_DSA_FIXED_BOUNDARY_GATHER') == '1':
    from lmcache.integration.vllm.glm_fixed_boundary import install_fixed_boundary
    from lmcache.v1.csa_attention_kv_prefetch_manager import CSAAttentionKVPrefetchManager
    from lmcache.v1.ssd_tp_sharded_prefetch import TorchDistributedShardGather
    install_fixed_boundary(GLMDSAPredictivePrefetchManager, GLMDSAPhysicalPrefetchSink,
                           CSAAttentionKVPrefetchManager, TorchDistributedShardGather)
'''
    changes={target:patched,
        args.lmcache_root/'v1/fixed_boundary_gather.py':args.scheduler.read_text(),
        args.lmcache_root/'integration/vllm/glm_fixed_boundary.py':args.hooks.read_text()}
    for path,source in changes.items():
        compile(source,str(path),'exec')
    shutil.copy2(target,backup)
    for path,source in changes.items():
        path.write_text(source)
    print('GLM_FIXED_BOUNDARY_V1011_INSTALLED',flush=True)


if __name__=='__main__':
    main()
