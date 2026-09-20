# SPDX-License-Identifier: Apache-2.0
"""Install the validated GLM v487 components into source before worker import."""

# Standard
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import textwrap


def definition(source: str, name: str, owner: str | None = None) -> ast.FunctionDef:
    """Return one exact function or class method; reject ambiguous source."""
    body = ast.parse(source).body
    if owner:
        classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == owner]
        if len(classes) != 1:
            raise ValueError(f'missing or ambiguous class {owner}')
        body = classes[0].body
    found = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(found) != 1:
        raise ValueError(f'missing or ambiguous function {name}')
    return copy.deepcopy(found[0])


def replace_function(source: str, function: ast.FunctionDef, owner: str | None = None) -> str:
    """Replace an exact existing definition, or append a missing class method."""
    tree = ast.parse(source)
    body = tree.body
    if owner:
        classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == owner]
        if len(classes) != 1:
            raise ValueError(f'missing class {owner}')
        body = classes[0].body
    existing = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == function.name]
    if len(existing) > 1:
        raise ValueError('ambiguous replacement')
    lines = source.splitlines(keepends=True)
    rendered = ast.unparse(ast.fix_missing_locations(function)) + '\n'
    if owner:
        rendered = textwrap.indent(rendered, '    ')
    if existing:
        old = existing[0]
        begin = min([old.lineno] + [node.lineno for node in old.decorator_list]) - 1
        lines[begin:old.end_lineno] = [rendered]
    elif owner:
        lines.insert(classes[0].end_lineno, '\n' + rendered)
    else:
        raise ValueError('new module-level definitions are not allowed')
    result = ''.join(lines)
    ast.parse(result)
    return result


def build_sources(root: Path, originals: Path) -> dict[Path, str]:
    """Build all source changes without writing or importing model modules."""
    changes = {}
    loader_path = root / 'v1/gpu_connector/tutti_direct_loader.py'
    loader = loader_path.read_text()
    loader_reference = (originals / 'tutti_direct_loader_device_v425.py').read_text()
    for name in ('supports_device_counted_read', 'enqueue_device_counted_read', 'finish_device_counted_reads'):
        loader = replace_function(loader, definition(loader_reference, name, 'TuttiDirectLoader'), 'TuttiDirectLoader')
    queue = ast.parse('def device_read_queue_depth(self) -> int:\n    """Return configured queue depth without synchronization or allocation."""\n    return self._q_depth()\n').body[0]
    loader = replace_function(loader, queue, 'TuttiDirectLoader')
    loader += ('\n# Serving extension setup; normal startup, never worker-script execution.\n'
               'if os.getenv("LMCACHE_SERVING_EXPERIMENT", "0") == "1":\n'
               '    from lmcache.integration.vllm.serving_counted_ops_v764 import ensure_counted_ops\n'
               '    ensure_counted_ops()\n')
    changes[loader_path] = loader

    manager_path = root / 'v1/csa_attention_kv_prefetch_manager.py'
    manager = manager_path.read_text()
    original = (originals / 'hotload_glm_device_miss_v425.py').read_text()
    owner = 'CSAAttentionKVPrefetchManager'
    # Preserve original public methods under public names, then install the
    # same wrappers that were tested in v487. No live-object private probing.
    for name in ('submit_selection_miss_reads', 'set_kv_forward_phase'):
        saved = definition(manager, name, owner)
        saved.name = 'serving_original_' + name
        manager = replace_function(manager, saved, owner)
    helpers = ('try_device_selection_miss_correction', 'submit_selection_miss_reads', 'set_kv_forward_phase')
    for name in helpers:
        function = definition(original, name)
        class Defaults(ast.NodeTransformer):
            """Resolve wrapper defaults from class-local original methods."""
            def visit_Name(self, node: ast.Name) -> ast.AST:
                mapping = {'original_submit': 'serving_original_submit_selection_miss_reads',
                           'original_phase': 'serving_original_set_kv_forward_phase'}
                if node.id in mapping:
                    node.id = mapping[node.id]
                return node

            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == 'loader' and node.attr == '_q_depth':
                    node.attr = 'device_read_queue_depth'
                return node
        function = Defaults().visit(function)
        # Legacy wrappers were unannotated; Any retains their runtime contract.
        for argument in function.args.args + function.args.kwonlyargs:
            argument.annotation = ast.Name(id='Any', ctx=ast.Load())
        if function.args.kwarg:
            function.args.kwarg.annotation = ast.Name(id='Any', ctx=ast.Load())
        function.returns = ast.Name(id='Any', ctx=ast.Load())
        # Class-local defaults require originals before the wrapper definition.
        manager = replace_function(manager, function, owner)
    # Defaults of wrappers resolve class names at definition time. Move the
    # saved originals before the wrappers, using one class-body AST rebuild.
    tree = ast.parse(manager)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    saved_methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name.startswith('serving_original_')]
    remaining = [node for node in cls.body if node not in saved_methods]
    if ast.get_docstring(cls):
        cls.body = remaining[:1] + saved_methods + remaining[1:]
    else:
        cls.body = saved_methods + remaining
    manager = ast.unparse(tree) + '\n'
    imports = ('import builtins as builtins_module\nimport os as os_module\n'
               'import torch as torch_module\nfrom lmcache.v1 import device_miss as device_module\n')
    # Insert helper imports immediately before the class; retain any future imports.
    marker = 'class ' + owner + ':'
    if manager.count(marker) != 1:
        raise ValueError('manager class declaration changed')
    manager = manager.replace(marker, imports + '\n' + marker)
    changes[manager_path] = '# SPDX-License-Identifier: Apache-2.0\n' + manager

    transport_path = root / 'v1/ssd_tp_sharded_prefetch.py'
    transport = transport_path.read_text()
    reference = (originals / 'ssd_tp_sharded_prefetch_v432.py').read_text()
    transport = replace_function(transport, definition(reference, 'owner_gpu_route'))
    transport = replace_function(transport, definition(reference, 'gather_owner_rows_gpu_into', 'TorchDistributedShardGather'), 'TorchDistributedShardGather')
    changes[transport_path] = transport
    scorer_path = root / 'v1/csa_prefill_cp_scorer.py'
    reference = (originals / 'csa_prefill_cp_scorer_v437.py').read_text()
    # The old hot patch assumes an earlier full owner-scorer installation.
    # Merely adding its helper to the unpatched module would never call it.
    def public_definitions(source: str) -> set[str]:
        """Return public function/class names for the compatibility gate."""
        return {node.name for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_')}
    if not public_definitions(scorer_path.read_text()) <= public_definitions(reference):
        raise ValueError('golden scorer removes existing public entry points')
    definition(reference, '_score_owner_full_query')
    if '_score_owner_full_query(' not in ast.unparse(definition(reference, 'score_prefill_proxy_rank_local')):
        raise ValueError('golden scorer does not route to the owner implementation')
    changes[scorer_path] = reference
    changes[root / 'v1/device_miss.py'] = (originals / 'device_miss_v429.py').read_text()
    return changes


def main() -> None:
    """Validate every generated file, then install with immutable backups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--originals', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    marker = args.lmcache_root / 'serving_stable_v764.json'
    if marker.exists():
        raise FileExistsError('already installed; inspect original installation evidence')
    changes = build_sources(args.lmcache_root, args.originals)
    changes[args.lmcache_root / 'integration/vllm/serving_counted_ops_v764.py'] = args.helper.read_text()
    evidence = {}
    for path, source in changes.items():
        compile(source, str(path), 'exec')
        backup = path.with_name(path.name + '.before_serving_v764')
        if backup.exists():
            raise FileExistsError(str(backup))
        evidence[str(path.relative_to(args.lmcache_root))] = hashlib.sha256(source.encode()).hexdigest()
    if not args.dry_run:
        for path, source in changes.items():
            if path.exists():
                shutil.copy2(path, path.with_name(path.name + '.before_serving_v764'))
            path.write_text(source)
        marker.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(dict(dry_run=args.dry_run, source_hashes=evidence)))


if __name__ == '__main__':
    main()
