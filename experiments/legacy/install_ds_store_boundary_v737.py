# SPDX-License-Identifier: Apache-2.0
"""Install a narrowly scoped, opt-in DS final-store boundary patch."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def transform(source: str) -> str:
    """Patch the sole full-object store call, preserving all other paths.

    Args:
        source: Installed adapter source after existing overlay installation.
    Returns:
        Modified Python source; an already marked source is returned unchanged.
    Raises:
        ValueError: An anchor is missing or ambiguous.
        SyntaxError: The transformed source is invalid Python.
    """
    if 'LMCACHE_DSV4_FINAL_STORE_BOUNDARY_V737' in source:
        ast.parse(source)
        return source
    anchor = 'from lmcache.v1.manager import LMCacheManager\n'
    call = '            self.lmcache_engine.store(\n'
    argument = ('                is_last_prefill=is_last_prefill,\n'
                '                request_total_tokens=request.prompt_len,\n')
    for token in (anchor, call, argument):
        if source.count(token) != 1:
            raise ValueError('adapter anchor missing or ambiguous: ' + token.strip())
    source = source.replace(anchor,
        'from lmcache.v1.dsv4_store_completion import reaches_last_complete_prompt_chunk\n' + anchor)
    insertion = '''            # Final partial scheduler batches may not call store at all.
            store_is_last_prefill = is_last_prefill
            if (
                os.getenv("LMCACHE_DSV4_FINAL_STORE_BOUNDARY_V737", "0") == "1"
                and request.disagg_spec is None
            ):
                store_is_last_prefill = (
                    is_last_prefill
                    or reaches_last_complete_prompt_chunk(
                        len(token_ids), request.prompt_len, self._lmcache_chunk_size
                    )
                )
                if store_is_last_prefill and not is_last_prefill:
                    logger.info(
                        "DSV4_FINAL_STORE_BOUNDARY_V737 request=%s stored_end=%d prompt_len=%d",
                        request.req_id, len(token_ids), request.prompt_len,
                    )

'''
    source = source.replace(call, insertion + call)
    source = source.replace(argument, argument.replace('=is_last_prefill,', '=store_is_last_prefill,'))
    ast.parse(source)
    return source


def main() -> None:
    """Install with a recoverable source backup, or only validate with dry-run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    target = args.lmcache_root / 'integration/vllm/vllm_v1_adapter.py'
    original = target.read_text()
    updated = transform(original)
    helper = args.helper.read_bytes()
    ast.parse(helper)
    if not args.dry_run:
        backup = target.with_name(target.name + '.before_v737')
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(updated)
        (args.lmcache_root / 'v1/dsv4_store_completion.py').write_bytes(helper)
    print(json.dumps(dict(dry_run=args.dry_run, target=str(target),
                          source_before=hashlib.sha256(original.encode()).hexdigest(),
                          source_after=hashlib.sha256(updated.encode()).hexdigest(),
                          helper_sha256=hashlib.sha256(helper).hexdigest())))


if __name__ == '__main__':
    main()
