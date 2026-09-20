# SPDX-License-Identifier: Apache-2.0
"""Connect GLM owner prefetch to its existing checked, asynchronous reader."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil

# Local
from install_aligned_packet_v999 import validate_globals
from install_glm_stable_startup_v764 import definition, replace_function


def build(source: str, hook: str) -> str:
    """Patch only owner submission and its request drain; preserve other modes."""
    owner = 'CSAAttentionKVPrefetchManager'
    helper = definition(hook, '_enqueue_glm_device_owner_reads')
    validate_globals(helper, source)
    patched = replace_function(source, helper, owner)
    method = definition(patched, '_prepare_shard_gather', owner)

    class OwnerRead(ast.NodeTransformer):
        """Keep the original local-read fallback byte for byte in the AST."""

        def __init__(self) -> None:
            self.count = 0

        def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
            if (isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'self._issue_local_reads'):
                self.count += 1
                before = ast.parse('''
local_event = (self._enqueue_glm_device_owner_reads(state, owned_to_read)
               if preserve_owner_partition and os.getenv('LMCACHE_GLM_DSA_DEVICE_OWNER_READ') == '1'
               else None)
if local_event is not None:
    local_completed = owned_to_read
else:
    pass
''').body
                before[-1].orelse = [node]
                return before
            return node

    transform = OwnerRead()
    method = transform.visit(method)
    if transform.count != 1:
        raise ValueError('Expected one owner local-read call')
    ast.fix_missing_locations(method)
    validate_globals(method, patched)
    patched = replace_function(patched, method, owner)
    drain = definition(patched, '_deactivate_request_locked', owner)
    anchors = [i for i, node in enumerate(drain.body)
               if isinstance(node, ast.Assign)
               and any(ast.unparse(t) == 'self._pending_raw_lba_cache' for t in node.targets)]
    if len(anchors) != 1:
        raise ValueError('Request cleanup boundary changed')
    drain.body[anchors[0]:anchors[0]] = ast.parse('''
if getattr(self, '_glm_device_owner_used', False):
    self._tutti_loader.finish_device_counted_reads()
    self._glm_device_owner_used = False
''').body
    patched = replace_function(patched, drain, owner)
    compile(patched, '<GLM device owner>', 'exec')
    return patched


def main() -> None:
    """Validate then install into a fresh deployment with immutable backups."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--hook', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / 'v1/csa_attention_kv_prefetch_manager.py'
    backup = target.with_suffix('.py.before_glm_device_owner_v1006')
    if backup.exists():
        raise FileExistsError('Require a fresh deployment')
    patched = build(target.read_text(), args.hook.read_text())
    helper = args.helper.read_text()
    compile(helper, str(args.helper), 'exec')
    shutil.copy2(target, backup)
    (args.lmcache_root / 'v1/owner_device_read.py').write_text(helper)
    target.write_text(patched)
    print('GLM_DEVICE_OWNER_V1006_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
