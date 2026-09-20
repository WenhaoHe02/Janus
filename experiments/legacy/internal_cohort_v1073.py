# SPDX-License-Identifier: Apache-2.0
"""Freeze the explicitly selected nine old plus six prospective internal requests."""

# Standard
import json
from pathlib import Path

ROOT = Path('/tmp/internal15_selected_v1073')


def selected_cohort() -> Path:
    """Return an immutable selected15 manifest; preserve post-hoc selection provenance."""
    retained = json.loads((ROOT/'retained9.json').read_text())
    supplement = json.loads(Path('/tmp/flash_supplement6_v1072/selection.json').read_text())
    rows = retained['requests'] + supplement['requests']
    if (len(retained['requests']) != 9 or len(supplement['requests']) != 6
            or len({r['source_id'] for r in rows}) != 15):
        raise ValueError('Nine plus six source identities are not unique')
    selection = dict(dataset='internal', selected_count=15, requests=rows,
        native_content_warmup=True, performance_metrics_used_for_selection=True,
        retained_original_cases=retained['retained_original_cases'],
        selection_rule='user-selected nine observed Flash K winners plus six candidates fixed before new measurements',
        interpretation='targeted favorable-scenario cohort, not an unbiased population sample',
        original_results_retained=True, output_policy='unaltered original output budgets')
    path = ROOT/'selection.json'
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError('Previously frozen selected15 changed')
    else:
        path.write_text(json.dumps(selection,indent=2))
    return path


def pro_selected_arguments(arguments: list[str]) -> list[str]:
    """Redirect only pending Pro internal retests to the user-approved selected15.

    Other models, public datasets, periodic runs and old files remain unchanged.
    """
    values = list(arguments)
    if ('--model-kind' not in values or values[values.index('--model-kind')+1] != 'pro'
            or '--output' not in values
            or not values[values.index('--output')+1].startswith('/tmp/ds_serial_retests_v1067/pro/internal/')):
        return values
    original = list(values)
    values[values.index('--selection')+1] = str(selected_cohort())
    values[values.index('--inputs')+1] = '/tmp/pro_internal15_native_v1073'
    arm = values[values.index('--arms')+1]
    (ROOT/f'{arm}_argument_override.json').write_text(json.dumps(dict(
        original=original, effective=values, authorization='user fixed9_plus6'),indent=2))
    return values
