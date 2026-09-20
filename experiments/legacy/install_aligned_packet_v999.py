# SPDX-License-Identifier: Apache-2.0
"""Install the matching aligned packet codec and transport method together."""

# Standard
import argparse
import ast
import builtins
import os
from pathlib import Path
import shutil
import symtable

# Local
from install_glm_stable_startup_v764 import definition, replace_function


def validate_globals(method: ast.FunctionDef, module_source: str) -> None:
    """Reject new function globals absent from the destination module.

    Args:
        method: Replacement method after diagnostic acknowledgements are added.
        module_source: Actual destination module, not the local reference.
    Raises:
        ValueError: A referenced global has no binding in the target or builtins.
    """
    bound = set(dir(builtins)) | {'__name__', '__file__', '__package__'}
    tree = ast.parse(module_source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                bound.update(item.id for item in ast.walk(target) if isinstance(item, ast.Name))
    scopes = symtable.symtable(ast.unparse(method), '<replacement>', 'exec').get_children()
    function_scope = next(scope for scope in scopes if scope.get_name() == method.name)
    referenced = {symbol.get_name() for symbol in function_scope.get_symbols()
                  if symbol.is_global() and symbol.is_referenced()}
    if missing := referenced - bound:
        raise ValueError(f'Unbound replacement globals: {sorted(missing)}')


def main() -> None:
    """Validate the pair, preserve backups, and change only the packet branch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--codec', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / 'v1/ssd_tp_sharded_prefetch.py'
    codec_target = args.lmcache_root / 'v1/owner_kv_packet.py'
    source = target.read_text()
    method = definition(args.transport.read_text(), '_gather_owner_packet',
                        'TorchDistributedShardGather')
    # One acknowledgement during warmup proves the selected branch executes.
    method.body[2:2] = ast.parse('''
if not getattr(self, 'aligned_packet_v999_logged', False):
    print('ALIGNED_OWNER_PACKET_V999_ACTIVE', self.rank, self.world_size, flush=True)
    self.aligned_packet_v999_logged = True
''').body
    validate_globals(method, source)
    patched = replace_function(source, method, 'TorchDistributedShardGather')
    codec = args.codec.read_text()
    definition(codec, 'owner_packet_layout')
    for path, text in ((target, patched), (codec_target, codec)):
        compile(text, str(path), 'exec')
        if path.with_suffix('.py.before_aligned_packet_v999').exists():
            raise RuntimeError('Require a fresh aligned-packet deployment')
    for path, text in ((target, patched), (codec_target, codec)):
        if path.exists():
            shutil.copy2(path, path.with_suffix('.py.before_aligned_packet_v999'))
        path.write_text(text)
    if os.getenv('LMCACHE_SERVING_DIAGNOSTIC_OWNER_WORKERS'):
        sink_path = args.lmcache_root / 'v1/glm_dsa_predictive_prefetch.py'
        sink_source = sink_path.read_text()
        initializer = definition(sink_source, '__init__', 'GLMDSAPhysicalPrefetchSink')
        matches = [i for i, node in enumerate(initializer.body)
                   if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) == 'self._prediction_executor' for target in node.targets)]
        if len(matches) != 1:
            raise ValueError('Owner executor initialization changed')
        initializer.body[matches[0]:matches[0]] = ast.parse(
            "print('GLM_OWNER_EXECUTOR_WORKERS_V1002', owner_read_workers, flush=True)"
        ).body
        sink_patched = replace_function(sink_source, initializer, 'GLMDSAPhysicalPrefetchSink')
        compile(sink_patched, str(sink_path), 'exec')
        shutil.copy2(sink_path, sink_path.with_suffix('.py.before_owner_workers_v1002'))
        sink_path.write_text(sink_patched)
    print('ALIGNED_PACKET_V999_INSTALLED_CODEC_AND_TRANSPORT', flush=True)


if __name__ == '__main__':
    main()
