# SPDX-License-Identifier: Apache-2.0
"""Narrow manager-owned hook installed before any GLM model imports."""

# Standard
from typing import Any

# Third Party
import torch


def _enqueue_glm_device_owner_reads(self: Any, state: Any, owned: torch.Tensor) -> Any:
    """Use checked device reads for joined owner IDs; unsupported layouts fall back."""
    from lmcache.v1.owner_device_read import enqueue_owner_ids

    loader = self._tutti_loader
    if (
        self._data_group != 'csa' or not state.chunks or state.block_slot_scatter
        or state.indexed_slba_table is None or state.indexed_dst_rows_table is None
        or state.k_cache_tensor.stride(-1) != 1
        or state.k_cache_tensor.stride(-2) != state.token_bytes
        or not callable(getattr(loader, 'enqueue_device_counted_read', None))
        or not callable(getattr(loader, 'device_read_queue_depth', None))
    ):
        return None
    device = state.k_cache_tensor.device
    # No model-stream dependency: these IDs are already joined host results.
    # The normal prepare stream may wait on a future authoritative indexer.
    with self._prepare_streams_lock:
        streams = getattr(self, '_glm_owner_prepare_streams', None)
        if streams is None:
            streams = self._glm_owner_prepare_streams = {}
        prepare = streams.get(device)
        if prepare is None:
            prepare = streams[device] = torch.cuda.Stream(device=device)
    # Mark before enqueue so a partial exception still retires reader leases.
    self._glm_device_selection_used = True
    self._glm_device_owner_used = True
    event = enqueue_owner_ids(
        loader, owned, state.indexed_slba_table, state.indexed_dst_rows_table,
        state.k_cache_tensor.view(torch.uint8).reshape(state.k_cache_tensor.shape[0], -1),
        state.in_pool_bitmap, prepare,
    )
    if event is not None and owned.numel() and not getattr(self, '_glm_device_owner_ack', False):
        self._glm_device_owner_ack = True
        print('GLM_DEVICE_OWNER_V1006_ACTIVE', device.index, owned.numel(), flush=True)
    return event
