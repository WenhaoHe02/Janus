# SPDX-License-Identifier: Apache-2.0
"""Freeze observed non-model interturn waits as a labeled tool-delay proxy."""

# Standard
import json
import math
from pathlib import Path
import statistics


def interturn_gap(previous: dict, current: dict) -> float:
    """Return next request start minus previous model response end, in seconds.

    This includes tools, client processing and network delay. It is not an
    isolated tool-runtime measurement. Negative/incomplete records are rejected.
    """
    gap = (float(current['request_timestamp']) - float(previous['request_timestamp'])
           - float(previous['total_latency_ms'])) / 1000
    if not math.isfinite(gap) or gap < 0 or float(previous['total_latency_ms']) < 0:
        raise ValueError('invalid original interturn clock/latency')
    return gap


def main() -> None:
    """Read only original metadata and save one identical policy for both arms."""
    source = json.loads(Path('/tmp/internal_source64_v904.json').read_text())
    records = {}
    for identity, location in source['requests'].items():
        with Path(location['source_file']).open() as stream:
            row = json.loads(next(line for number, line in enumerate(stream, 1)
                                  if number == location['source_line']))
        if row['request_id'] != identity or not row['is_complete'] or row['response_status'] != 200:
            raise ValueError('original source identity or completion changed')
        records[identity] = {key: row[key] for key in ('request_timestamp', 'total_latency_ms')}
    waits, evidence, invalid = {}, [], []
    for chain in source['unbranched_chains']:
        for previous, current in zip(chain, chain[1:]):
            try:
                gap = interturn_gap(records[previous], records[current])
            except (ValueError, KeyError) as error:
                invalid.append(dict(previous=previous, current=current, error=str(error)))
                continue
            waits[current] = gap
            evidence.append(dict(previous_source_request_id=previous, source_request_id=current,
                previous_timing=records[previous], current_timing=records[current], wait_s=gap))
    result = dict(provenance='Observed interturn non-model idle proxy: request_timestamp(next) - request_timestamp(previous) - total_latency_ms(previous); includes tools/client/network, not isolated tool runtime',
        seconds_by_source_id=waits, evidence=evidence, invalid=invalid,
        applied_identically_across_arms=True,
        first_formal_wait='not applied; initial session arrival is independently scheduled',
        median_wait_s=statistics.median(waits.values()) if waits else None,
        max_wait_s=max(waits.values()) if waits else None)
    output = Path('/tmp/internal_tool_delay_proxy_v907.json')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(valid=len(waits), invalid=len(invalid),
                          median_wait_s=result['median_wait_s'], max_wait_s=result['max_wait_s'])))
    if invalid:
        raise ValueError('invalid delay records require resolution before replay')


if __name__ == '__main__':
    main()
