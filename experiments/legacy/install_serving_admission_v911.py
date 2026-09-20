# SPDX-License-Identifier: Apache-2.0
"""Install scoped DS terminal-admission and owner-capacity fixes after overlays."""

# Standard
import argparse
import ast
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    """Replace a unique deployment anchor, failing closed on version drift."""
    if source.count(old) != 1:
        raise ValueError(f'Expected one deployment anchor: {old[:100]!r}')
    return source.replace(old, new, 1)


def method_span(source: str, cls: str, method: str) -> tuple[int, int]:
    """Locate the complete decorated class method in Python source."""
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    item = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return min([item.lineno] + [d.lineno for d in item.decorator_list]) - 1, item.end_lineno


def install(root: Path, engine_source: Path) -> None:
    """Patch only verified methods/anchors; save each original file once.

    Args:
        root: Installed LMCache package, before worker imports.
        engine_source: Local reviewed engine source supplying the guarded method.
    Raises:
        ValueError: A deployed anchor differs from the inspected version.
    """
    engine = root / 'v1/cache_engine.py'
    source = engine.read_text()
    outputs = {}
    if 'STREAMING_TERMINAL_TAIL_GUARD_V911' not in source:
        replacement = engine_source.read_text()
        begin, end = method_span(source, 'LMCacheEngine', 'lookup_streaming_terminal')
        a, b = method_span(replacement, 'LMCacheEngine', 'lookup_streaming_terminal')
        lines = source.splitlines(keepends=True)
        lines[begin:end] = replacement.splitlines(keepends=True)[a:b]
        outputs[engine] = ''.join(lines)
    client = root / 'v1/lookup_client/lmcache_lookup_client.py'
    source = client.read_text()
    if 'REQUEST_TOTAL_LOOKUP_V911' not in source:
        source = replace_once(source, '            msg_buf = [hashes, offsets]\n',
            '            # REQUEST_TOTAL_LOOKUP_V911: hashes omit the partial tail.\n'
            '            terminal_hint = dict(terminal_hint or {})\n'
            '            terminal_hint["request_total_tokens"] = len(token_ids)\n'
            '            msg_buf = [hashes, offsets]\n')
        source = replace_once(source, '                        terminal_fast_hit = False\n',
            '                        terminal_fast_hit = False\n'
            '                        request_total_tokens = max(sum(int(x) for x in offsets),\n'
            '                            int((terminal_hint or {}).get("request_total_tokens", 0)))\n')
        for name in ('candidate_tokens', 'hint_tokens'):
            anchor = f'                                        token_count={name},\n'
            source = replace_once(source, anchor, anchor +
                '                                        request_total_tokens=request_total_tokens,\n')
        outputs[client] = source
    manager = root / 'v1/indexer_ssd_manager.py'
    source = manager.read_text()
    if 'OWNER_CAPACITY_INTERSECTION_V911' not in source:
        anchor = '                            if device_owner_submit:\n'
        source = replace_once(source, anchor, anchor +
            '                                # OWNER_CAPACITY_INTERSECTION_V911\n'
            '                                # The scorer includes append K, while this vector\n'
            '                                # is already limited to initialized history. Both\n'
            '                                # bounds hold; their minimum drops no selected ID.\n'
            '                                owner_bound = min(int(owner_bound), int(selected_blocks.numel()))\n')
        outputs[manager] = source
    # Check all anchors and syntax before modifying any target.
    for path, content in outputs.items():
        compile(content, str(path), 'exec')
    for path, content in outputs.items():
        backup = path.with_suffix(path.suffix + '.before_v911')
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_text(content)
    print('SERVING_ADMISSION_V911_INSTALLED', [str(path) for path in outputs], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--engine-source', type=Path, default=Path('/tmp/cache_engine_source_v911.py'))
    args = parser.parse_args()
    install(args.lmcache_root, args.engine_source)
