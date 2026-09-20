# SPDX-License-Identifier: Apache-2.0
"""Install a guarded vLLM scheduling policy before engine initialization."""

# Standard
import ast
import importlib.util
from pathlib import Path
import shutil
import textwrap

MARKER = '# LMCACHE_PREFILL_FIRST_NO_MIX_V1068'


def replace_once(source: str, before: str, after: str) -> str:
    """Replace exactly one known source anchor, failing on incompatible versions."""
    if source.count(before) != 1:
        raise ValueError(f'Unexpected scheduler source anchor: {before[:90]!r}')
    return source.replace(before, after, 1)


def patch_source(source: str) -> str:
    """Add prefill-then-decode passes without freeing paused requests or KV.

    Args:
        source: Exact deployed Scheduler source containing the known V1 loops.
    Returns:
        Syntactically checked opt-in source. Disabled policy preserves the single
        legacy pass. Enabled policy schedules at most one request per GPU batch,
        admits waiting requests only during prefill pass, then falls back to
        decode in the SAME iteration if no prefill can run. It never forces
        preemption to admit a prefill. Insufficient decode KV must fail visibly
        rather than spin forever.
    Raises:
        ValueError: Required source anchors are missing or duplicated.
        SyntaxError: Generated source is invalid.
    """
    if MARKER in source:
        ast.parse(source)
        return source
    source = replace_once(source, 'logger = init_logger(__name__)',
        'from vllm.v1.core.sched.prefill_first_policy_v1068 import phase_accepts\n\n'
        + MARKER + '\n_LMCACHE_PREFILL_FIRST_NO_MIX = os.getenv(\n'
        '    "LMCACHE_PREFILL_FIRST_NO_MIX", "0"\n).lower() in {"1", "true", "yes", "on"}\n\n'
        'logger = init_logger(__name__)')
    source = replace_once(source,
        '        self.max_num_running_reqs = self.scheduler_config.max_num_seqs',
        '        self.max_num_running_reqs = self.scheduler_config.max_num_seqs\n'
        '        if _LMCACHE_PREFILL_FIRST_NO_MIX:\n'
        '            if self.max_num_running_reqs < 2:\n'
        '                raise ValueError("prefill-first needs >=2 resident slots; GPU batches remain size1")\n'
        '            self.lmcache_empty_schedule_count = 0\n'
        '            logger.info("LMCACHE_PREFILL_FIRST_NO_MIX_V1068 resident_slots=%s batch_limit=1", self.max_num_running_reqs)')
    start = source.index('        # First, schedule the RUNNING requests.\n')
    end = source.index('        # Check if the scheduling constraints are satisfied.\n', start)
    block = source[start:end]
    block = replace_once(block,
        '            request = self.running[req_index]\n',
        '            request = self.running[req_index]\n'
        '            if _LMCACHE_PREFILL_FIRST_NO_MIX and num_scheduled_tokens:\n'
        '                break\n'
        '            if not phase_accepts(request, lmcache_phase):\n'
        '                req_index += 1\n'
        '                continue\n')
    block = replace_once(block,
        '                    # The request cannot be scheduled.\n'
        '                    # Preempt the lowest-priority request.',
        '                    if _LMCACHE_PREFILL_FIRST_NO_MIX:\n'
        '                        # Keep every parked request and its KV intact.\n'
        '                        break\n\n'
        '                    # The request cannot be scheduled.\n'
        '                    # Preempt the lowest-priority request.')
    block = replace_once(block,
        '        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:',
        '        if (not preempted_reqs and self._pause_state == PauseState.UNPAUSED\n'
        '                and lmcache_phase != "decode"\n'
        '                and not (_LMCACHE_PREFILL_FIRST_NO_MIX and num_scheduled_tokens)):')
    block = replace_once(block,
        '            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n',
        '            while (self.waiting or self.skipped_waiting) and token_budget > 0:\n'
        '                if _LMCACHE_PREFILL_FIRST_NO_MIX and num_scheduled_tokens:\n'
        '                    break\n')
    source = source[:start] + (
        '        lmcache_phases = ("prefill", "decode") if _LMCACHE_PREFILL_FIRST_NO_MIX else (None,)\n'
        '        for lmcache_phase in lmcache_phases:\n'
        + textwrap.indent(block, '    ')
        + '            if num_scheduled_tokens:\n'
        '                break\n\n'
    ) + source[end:]
    source = replace_once(source,
        '        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n',
        '        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n'
        '        if _LMCACHE_PREFILL_FIRST_NO_MIX:\n'
        '            assert len(num_scheduled_tokens) <= 1, "prefill-first batch must not mix requests"\n'
        '            self.lmcache_empty_schedule_count = (\n'
        '                0 if total_num_scheduled_tokens or not self.running\n'
        '                else self.lmcache_empty_schedule_count + 1)\n'
        '            if self.lmcache_empty_schedule_count > 10000:\n'
        '                raise RuntimeError("prefill-first made no progress; inspect KV capacity / in-flight work")\n'
        '            if os.getenv("LMCACHE_PREFILL_FIRST_AUDIT", "0") == "1":\n'
        '                logger.info("PREFILL_FIRST_STEP phase=%s selected=%s resident=%s preempted=%s",\n'
        '                            lmcache_phase, num_scheduled_tokens,\n'
        '                            [r.request_id for r in self.running],\n'
        '                            [r.request_id for r in preempted_reqs])\n')
    ast.parse(source)
    return source


def main() -> None:
    """Install before startup imports, backing up the exact scheduler once."""
    spec = importlib.util.find_spec('vllm')
    if spec is None or spec.origin is None:
        raise RuntimeError('vLLM package unavailable')
    directory = Path(spec.origin).parent/'v1/core/sched'
    target = directory/'scheduler.py'
    original = target.read_text()
    patched = patch_source(original)
    if patched == original:
        return
    with (directory/'scheduler.before_v1068.py').open('x') as backup:
        backup.write(original)
    shutil.copyfile(Path(__file__).with_name('prefill_first_policy_v1068.py'),
                    directory/'prefill_first_policy_v1068.py')
    temporary = directory/'scheduler.staged_v1068.py'
    temporary.write_text(patched)
    temporary.replace(target)
    print('PREFILL_FIRST_V1068_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
