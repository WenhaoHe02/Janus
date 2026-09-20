# SPDX-License-Identifier: Apache-2.0
"""Opt-in startup installation of communicator-local prefetch CTA controls."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

# Local
from install_glm_stable_startup_v764 import definition, replace_function


def build_transport_source(source: str) -> str:
    """Add options only to the two prefetch communicators in from_model_group.

    Args:
        source: Deployed transport module source before worker imports.
    Returns:
        Source preserving all other methods and all collective call order.
    Raises:
        ValueError: The expected two communicator constructors are absent or
            partially patched, so a safe narrow replacement cannot be proved.
    """
    owner = 'TorchDistributedShardGather'
    method = definition(source, 'from_model_group', owner)
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'dist' and node.func.attr == 'new_group']
    if len(calls) != 2:
        raise ValueError('Expected exactly two prefetch new_group calls')
    patched = [any(keyword.arg is None and isinstance(keyword.value, ast.Call)
                   and isinstance(keyword.value.func, ast.Name)
                   and keyword.value.func.id == 'prefetch_nccl_group_options'
                   for keyword in call.keywords) for call in calls]
    if all(patched):
        return source
    if any(patched) or any(keyword.arg is None or keyword.arg == 'pg_options'
                           for call in calls for keyword in call.keywords):
        raise ValueError('Existing communicator options require manual reconciliation')
    method.body.insert(1, ast.parse(
        'from lmcache.v1.prefetch_nccl_config import prefetch_nccl_group_options'
    ).body[0])
    for call in calls:
        call.keywords.append(ast.keyword(arg=None, value=ast.Call(
            func=ast.Name(id='prefetch_nccl_group_options', ctx=ast.Load()),
            args=[ast.Name(id='dist', ctx=ast.Load())], keywords=[])))
    return replace_function(source, method, owner)


def main() -> None:
    """Back up the deployed file and install a narrowly guarded startup patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args()
    target = args.lmcache_root / 'v1/ssd_tp_sharded_prefetch.py'
    original = target.read_text()
    patched = build_transport_source(original)
    helper = args.helper.read_text()
    compile(helper, str(args.helper), 'exec')
    compile(patched, str(target), 'exec')
    backup = target.with_suffix('.py.before_prefetch_ctas_v991')
    if not backup.exists():
        shutil.copy2(target, backup)
    (args.lmcache_root / 'v1/prefetch_nccl_config.py').write_text(helper)
    target.write_text(patched)
    print('PREFETCH_CTA_PATCH_INSTALLED', json.dumps({
        'before_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'after_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'helper_sha256': hashlib.sha256(helper.encode()).hexdigest(),
    }), flush=True)


if __name__ == '__main__':
    main()
