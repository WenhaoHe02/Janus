# SPDX-License-Identifier: Apache-2.0
"""Enable prefill-first only in the separately recorded DS serial retest."""

# Standard
import json
import os
from pathlib import Path
import runpy
import sys

# Local
from install_prefill_first_v1068 import main as install_scheduler


def resident_slot_command(command: list[str]) -> list[str]:
    """Set two resident slots while the patched scheduler enforces batch size1.

    Args:
        command: Existing vLLM server command.
    Returns:
        Copy of the command with max-num-seqs=2.
    Raises:
        ValueError: Existing slot controls are ambiguous or unexpected.
    """
    result = list(command)
    positions = [i for i, value in enumerate(result)
                 if value == '--max-num-seqs' or value.startswith('--max-num-seqs=')]
    if len(positions) > 1:
        raise ValueError('Duplicate max-num-seqs')
    if not positions:
        return result + ['--max-num-seqs', '2']
    i = positions[0]
    if '=' in result[i]:
        if result[i].split('=', 1)[1] not in ('1', '2'):
            raise ValueError('Unexpected resident slot count')
        result[i] = '--max-num-seqs=2'
    else:
        if result[i+1] not in ('1', '2'):
            raise ValueError('Unexpected resident slot count')
        result[i+1] = '2'
    return result


def main() -> None:
    """Install before engine initialization, preserving the existing DS fixes."""
    install_scheduler()
    sys.argv[1:] = resident_slot_command(sys.argv[1:])
    os.environ['LMCACHE_PREFILL_FIRST_NO_MIX'] = '1'
    os.environ['LMCACHE_PREFILL_FIRST_AUDIT'] = '0'
    proof = dict(policy='prefill_first_no_mix_v1068', resident_slots=2,
                 max_requests_per_gpu_batch=1, command=sys.argv[1:],
                 write_boundary='response_and_write_publication_v1067')
    Path('/tmp/prefill_first_provenance_v1068.json').write_text(json.dumps(proof, indent=2))
    print('DS_SERIAL_PREFILL_FIRST_V1068', json.dumps(proof), flush=True)
    runpy.run_path('/tmp/exec_ds_serving_v916.py', run_name='__main__')


if __name__ == '__main__':
    main()
