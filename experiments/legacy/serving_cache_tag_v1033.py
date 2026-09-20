# SPDX-License-Identifier: Apache-2.0
"""Use LMCache's existing request tags to isolate controlled cache snapshots."""

# Standard
import re


def cache_parameters(skip_save: bool, tag: str | None = None) -> dict:
    """Return save policy plus an optional independent-request cache tag.

    Args:
        skip_save: Suppress publication for excluded warmup requests.
        tag: Stable identifier shared by this request's seed and timed replay.
    Returns:
        Existing LMCache kv_transfer_params, without changing prompt tokens.
    Raises:
        ValueError: A tag contains separators reserved by cache-key encoding.
    """
    parameters = {"lmcache.skip_save": skip_save}
    if tag is not None:
        if re.fullmatch(r"[A-Za-z0-9_.:-]+", tag) is None:
            raise ValueError("Invalid independent-request cache tag")
        parameters["lmcache.tag.serving_request"] = tag
    return parameters
