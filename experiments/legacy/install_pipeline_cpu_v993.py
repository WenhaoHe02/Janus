# SPDX-License-Identifier: Apache-2.0
"""Install request-scoped CPU diagnostics before importing GLM model hooks."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil


def main() -> None:
    """Validate public interfaces and add an explicitly gated source hook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / 'v1/glm_dsa_predictive_prefetch.py'
    source = target.read_text()
    tree = ast.parse(source)
    contracts = {
        'GLMDSAPredictivePrefetchManager': {'begin_request', 'end_request', 'after_source_layer', 'observe_true_topk'},
        'GLMDSAPhysicalPrefetchSink': {'start_consumer_prefetch', 'wait_for_consumer'},
    }
    for name, methods in contracts.items():
        owners = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
        if len(owners) != 1 or not methods <= {node.name for node in owners[0].body if isinstance(node, ast.FunctionDef)}:
            raise RuntimeError(f'Public diagnostic contract missing: {name}')
    marker = '# PIPELINE_CPU_DIAGNOSTIC_V993'
    if marker in source:
        raise RuntimeError('Require a fresh diagnostic deployment')
    patched = source + '\n' + marker + '''
if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_PIPELINE_CPU_DIR'):
    from pathlib import Path as PipelineDiagnosticPath
    from lmcache.integration.vllm.pipeline_cpu_diagnostic_v993 import install_pipeline_profiling
    install_pipeline_profiling(
        GLMDSAPredictivePrefetchManager, GLMDSAPhysicalPrefetchSink,
        PipelineDiagnosticPath(os.environ['LMCACHE_SERVING_DIAGNOSTIC_PIPELINE_CPU_DIR']),
    )
'''
    helper = args.helper.read_text()
    compile(helper, str(args.helper), 'exec')
    compile(patched, str(target), 'exec')
    backup = target.with_suffix('.py.before_pipeline_cpu_v993')
    if backup.exists():
        raise RuntimeError('Diagnostic backup already exists')
    shutil.copy2(target, backup)
    (args.lmcache_root / 'integration/vllm/pipeline_cpu_diagnostic_v993.py').write_text(helper)
    target.write_text(patched)
    print('PIPELINE_CPU_DIAGNOSTIC_V993_INSTALLED_NONPRODUCTION', flush=True)


if __name__ == '__main__':
    main()
