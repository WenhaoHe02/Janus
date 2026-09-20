# SPDX-License-Identifier: Apache-2.0
"""Narrow startup patch for task-scoped owner-read dependencies and GPU probes."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil

# Local
from install_glm_stable_startup_v764 import definition, replace_function


def build_sources(root: Path) -> dict[Path, str]:
    """Patch only task submission and the generic-read upstream wait.

    Args:
        root: LMCache package root in a fresh deployment.
    Returns:
        Validated source replacements, without writing them.
    Raises:
        ValueError: Reviewed method/anchor counts do not match this deployment.
    """
    sink_path = root / 'v1/glm_dsa_predictive_prefetch.py'
    source = sink_path.read_text()
    owner = 'GLMDSAPhysicalPrefetchSink'
    method = definition(source, '_submit_predicted_layer', owner)
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == 'self._prediction_executor.submit']
    if len(calls) != 1 or ast.unparse(calls[0].args[0]) != 'manager.fire_predicted_reads':
        raise ValueError('Unexpected GLM prediction task submission')
    method.body.insert(0, ast.parse(
        'from lmcache.v1.gpu_connector.prediction_io_dependency import bind_prediction_read_dependency'
    ).body[0])
    calls[0].args[0] = ast.parse('''bind_prediction_read_dependency(
        manager.fire_predicted_reads, target_layer=int(layer_id),
        source_layer=int(profile_source_layer) if profile_source_layer is not None
            else int(event.schedule.source_layer) if event is not None else None
    ) if self._preserve_owner_partition else manager.fire_predicted_reads''', mode='eval').body
    sink = replace_function(source, method, owner)

    loader_path = root / 'v1/gpu_connector/tutti_direct_loader.py'
    source = loader_path.read_text()
    owner = 'TuttiDirectLoader'
    method = definition(source, '_load_batch', owner)

    class ReadWaitPatch(ast.NodeTransformer):
        """Replace one legacy wait and observe it after the existing poll sync."""

        def __init__(self) -> None:
            self.waits = 0
            self.emits = 0

        def visit_Expr(self, node: ast.Expr) -> ast.AST:
            if ast.unparse(node) == 'io_stream.wait_stream(torch.cuda.current_stream())':
                self.waits += 1
                return ast.parse('dependency_probe = wait_for_prediction_read_dependency(io_stream, torch.cuda.current_stream())').body[0]
            return self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | list[ast.AST]:
            if isinstance(node.target, ast.Name) and node.target.id == 'poll_sync_ms':
                self.emits += 1
                return [node, *ast.parse('finish_prediction_read_wait_probe(dependency_probe)\ndependency_probe = None').body]
            return self.generic_visit(node)

    patcher = ReadWaitPatch()
    method = patcher.visit(method)
    if (patcher.waits, patcher.emits) != (1, 1):
        raise ValueError('Unexpected generic-read wait/poll structure')
    insert_at = int(bool(method.body and isinstance(method.body[0], ast.Expr)
                         and isinstance(method.body[0].value, ast.Constant)
                         and isinstance(method.body[0].value.value, str)))
    method.body[insert_at:insert_at] = ast.parse(
        'from lmcache.v1.gpu_connector.prediction_io_dependency import wait_for_prediction_read_dependency, finish_prediction_read_wait_probe\n'
        'dependency_probe = None'
    ).body
    loader = replace_function(source, method, owner)
    return {sink_path: sink, loader_path: loader}


def main() -> None:
    """Back up fresh deployed sources and install the task dependency helper."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args()
    replacements = build_sources(args.lmcache_root)
    for path, source in replacements.items():
        compile(source, str(path), 'exec')
        if path.with_suffix('.py.before_dependency_v996').exists():
            raise RuntimeError('Require a fresh dependency-test deployment')
    helper = args.helper.read_text()
    compile(helper, str(args.helper), 'exec')
    for path, source in replacements.items():
        shutil.copy2(path, path.with_suffix('.py.before_dependency_v996'))
        path.write_text(source)
    (args.lmcache_root / 'v1/gpu_connector/prediction_io_dependency.py').write_text(helper)
    print('PREDICTION_IO_DEPENDENCY_V996_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
