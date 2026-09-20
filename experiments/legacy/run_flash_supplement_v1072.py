# SPDX-License-Identifier: Apache-2.0
"""Run the six user-requested prospective Flash cases on both K and NP."""

# Standard
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/tmp/flash_supplement6_v1072')


def main() -> None:
    """Freeze original bodies, then run new paired serial cohorts under host0.

    The parent serial scope owns host restoration. This runs after Flash's old
    four-arm cohorts, before switching to Pro. No completed request is rerun or
    discarded based on the new result. All original output budgets are retained.
    """
    # Local
    from build_existing29_model_inputs_v957 import body_hash
    from diagnose_serving_decode_v973 import assert_idle

    if (ROOT/'COMPLETE.json').exists():
        return
    assert_idle()
    if Path('/proc/sys/vm/compaction_proactiveness').read_text().strip() != '0':
        raise RuntimeError('Supplement must share the parent host0 scope')
    candidates = json.loads((ROOT/'candidates.json').read_text())
    candidates['native_content_warmup'] = True
    if len(candidates['requests']) != 6:
        raise ValueError('Expected six prospective requests')
    if (ROOT/'selection.json').exists():
        raise FileExistsError('Supplement already started; inspect before resuming')
    for row in candidates['requests']:
        with Path(row['source_file']).open() as stream:
            source = next(json.loads(line) for n, line in enumerate(stream,1)
                          if n == row['source_line'])
        if (source['request_id'] != row['source_id'] or source['response_status'] != 200
                or source['completion_tokens'] != row['O']):
            raise ValueError('Original source identity/output differs')
        row['body_sha256'] = body_hash(source['request_body'])
    candidates['cluster_field_note'] = 'selection-region tags, not learned dataset cluster IDs'
    (ROOT/'selection.json').write_text(json.dumps(candidates,indent=2))
    from internal_cohort_v1073 import selected_cohort
    selected_cohort()
    policy_path = Path('/tmp/serving_arrival_policy_v1050.json')
    before = policy_path.read_bytes()
    policy = json.loads(before)
    if policy['mode'] != 'serial' or policy['replace_requested_rate'] != .005:
        raise ValueError('Serial policy changed')
    (ROOT/'arrival_policy.before.json').write_bytes(before)
    if str(ROOT) not in policy['output_roots']:
        policy['output_roots'].append(str(ROOT))
    temporary = ROOT/'arrival_policy.staged.json'
    temporary.write_text(json.dumps(policy,indent=2))
    os.chmod(temporary, policy_path.stat().st_mode & 0o777)
    if policy_path.read_bytes() != before:
        raise RuntimeError('Concurrent arrival policy change')
    temporary.replace(policy_path)
    command = ['python3','/tmp/run_ds_supplement6_v1072.py',
        '--model-kind','flash','--selection',str(ROOT/'selection.json'),
        '--inputs',str(ROOT/'native'), '--output',str(ROOT/'run'),
        '--checkpoint','/home/zbuser02/models/DeepSeek-V4-Flash-HF',
        '--predecessor','/tmp/flash_internal15_serving_v1053/retry3',
        '--predecessor-arms','k_local','no_prefetch','dram','ssd_only',
        '--host-compaction','0','--arms','k_local','no_prefetch','dram','ssd_only','--rates','0.005']
    environment = dict(os.environ,
        LMCACHE_EXPERIMENT_STARTUP_SCRIPT='/tmp/startup_ds_serial_prefill_first_v1068.sh',
        LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES=str(16*1024**3))
    (ROOT/'command.json').write_text(json.dumps(command,indent=2))
    print('FLASH_SUPPLEMENT6_START_K_THEN_NP', flush=True)
    with (ROOT/'run.log').open('x') as log:
        subprocess.run(command,env=environment,stdout=log,stderr=log,check=True,timeout=7200)
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(requests_per_arm=6,
        arms=['k_local','no_prefetch','dram','ssd_only'], arrival_mode='serial',
        selection='targeted_followup_frozen_before_new_measurements')))
    print('FLASH_SUPPLEMENT6_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
