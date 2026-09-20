# SPDX-License-Identifier: Apache-2.0
"""Hot-load DS device-counted reads for GLM IndexShare correction."""

for route in app.routes:
    if route.path == "/run_script":
        importer = route.endpoint.__globals__["importlib"]

builtins_module = importer.import_module("builtins")
ast_module = importer.import_module("ast")
os_module = importer.import_module("os")
torch_module = importer.import_module("torch")
loader_module = importer.import_module(
    "lmcache.v1.gpu_connector.tutti_direct_loader"
)
manager_module = importer.import_module(
    "lmcache.v1.csa_attention_kv_prefetch_manager"
)
device_module = importer.reload(importer.import_module("lmcache.v1.device_miss"))

with builtins_module.open("/tmp/tutti_direct_loader_device_v425.py") as source:
    tree = ast_module.parse(source.read())
for node in tree.body:
    if not builtins_module.isinstance(node, ast_module.ClassDef):
        continue
    if node.name != "TuttiDirectLoader":
        continue
    for function in node.body:
        if not builtins_module.isinstance(function, ast_module.FunctionDef):
            continue
        if function.name not in {
            "supports_device_counted_read",
            "enqueue_device_counted_read",
            "finish_device_counted_reads",
        }:
            continue
        namespace = builtins_module.dict(loader_module.__dict__)
        builtins_module.exec(
            builtins_module.compile(
                ast_module.Module(body=[function], type_ignores=[]),
                "/tmp/tutti_direct_loader_device_v425.py",
                "exec",
            ),
            namespace,
        )
        builtins_module.setattr(
            loader_module.TuttiDirectLoader,
            function.name,
            namespace[function.name],
        )


def try_device_selection_miss_correction(
    self,
    layer_id,
    selection,
    request_token=None,
    _builtins=builtins_module,
    _device=device_module,
    _torch=torch_module,
):
    """Submit an exact GLM selection miss without a host bitmap readback."""
    state = self._layers.get(_builtins.int(layer_id))
    loader = self._tutti_loader
    if (
        self._data_group != "csa"
        or state is None
        or not state.chunks
        or state.block_slot_scatter
        or not selection.bitmap.is_cuda
        or not state.in_pool_bitmap.is_cuda
        or selection.bitmap.device != state.in_pool_bitmap.device
        or state.k_cache_tensor.stride(-1) != 1
        or state.k_cache_tensor.stride(-2) != state.token_bytes
        or not _builtins.callable(
            _builtins.getattr(loader, "enqueue_device_counted_read", None)
        )
    ):
        return False
    token = selection.request_token if request_token is None else request_token
    if token != selection.request_token or token != self.active_request_token:
        raise _builtins.RuntimeError("device correction request is stale")
    limit = _builtins.min(
        _builtins.int(state.chunks[-1].end_compressed_block),
        _builtins.int(state.in_pool_bitmap.numel()),
    )
    io_nbytes = _builtins.int(state.compressed_block_size * state.token_bytes)
    slbas = state.indexed_slba_table
    destinations = state.indexed_dst_rows_table
    capacity = _builtins.min(limit, _builtins.int(loader._q_depth()) - 1)
    if (
        not 0 < limit <= 4096
        or capacity <= 0
        or slbas is None
        or destinations is None
        or not loader.supports_device_counted_read(capacity, io_nbytes)
    ):
        return False
    with state.pending_reads_lock:
        if state.pending_read_count:
            return False
    device = state.k_cache_tensor.device
    with _torch.cuda.device(device):
        stream = _torch.cuda.current_stream(device)
        if selection.ready_event is not None:
            stream.wait_event(selection.ready_event)
        selected = selection.bitmap[:limit].contiguous()
        batches = _device.pack_device_miss_batches(
            selected,
            state.in_pool_bitmap[:limit],
            limit,
            capacity,
        )
        ready = _torch.cuda.Event()
        ready.record(stream)
        destination = state.k_cache_tensor.view(_torch.uint8).reshape(
            state.k_cache_tensor.shape[0], -1
        )
        done = ready
        for ids, count in batches:
            done = loader.enqueue_device_counted_read(
                ids,
                count,
                slbas,
                destinations,
                destination,
                state.in_pool_bitmap,
                ready,
            )
        stream.wait_event(done)
    with state.pending_reads_lock:
        state.last_drain_event = done
    state.true_selected_blocks_bitmap = selection.bitmap
    state.true_selected_covers_cached_prefix = False
    self._glm_device_selection_used = True
    return True


manager_class = manager_module.CSAAttentionKVPrefetchManager
builtins_module.setattr(
    manager_class,
    "try_device_selection_miss_correction",
    try_device_selection_miss_correction,
)

if not builtins_module.hasattr(
    manager_class,
    "_glm_device_selection_original_submit",
):
    manager_class._glm_device_selection_original_submit = (
        manager_class.submit_selection_miss_reads
    )
original_submit = manager_class._glm_device_selection_original_submit


def submit_selection_miss_reads(
    self,
    layer_id,
    selection,
    _original=original_submit,
    _os=os_module,
    _torch=torch_module,
    **kwargs,
):
    """Prefer device-counted GLM correction and retain the CPU fallback."""
    if (
        _os.getenv("LMCACHE_GLM_DEVICE_SELECTION_MISS_READ", "0") == "1"
        and self.try_device_selection_miss_correction(
            layer_id,
            selection,
            request_token=kwargs.get("request_token"),
        )
    ):
        return _torch.empty(0, dtype=_torch.int64)
    return _original(self, layer_id, selection, **kwargs)


builtins_module.setattr(
    manager_class,
    "submit_selection_miss_reads",
    submit_selection_miss_reads,
)

if not builtins_module.hasattr(manager_class, "_glm_device_original_phase"):
    manager_class._glm_device_original_phase = manager_class.set_kv_forward_phase
original_phase = manager_class._glm_device_original_phase


def set_kv_forward_phase(
    self,
    phase,
    _builtins=builtins_module,
    _original=original_phase,
    **kwargs,
):
    """Release counted-read reservations when one model forward finishes."""
    if phase == "inactive" and _builtins.getattr(
        self, "_glm_device_selection_used", False
    ):
        self._tutti_loader.finish_device_counted_reads()
        self._glm_device_selection_used = False
    return _original(self, phase, **kwargs)


builtins_module.setattr(manager_class, "set_kv_forward_phase", set_kv_forward_phase)
os_module.environ["LMCACHE_GLM_DEVICE_SELECTION_MISS_READ"] = "1"
manager = manager_module.get_csa_attention_kv_prefetch_manager()
loader = manager._tutti_loader if manager is not None else None
result = {
    "installed": True,
    "manager": manager is not None,
    "q_depth": loader._q_depth() if loader is not None else None,
    "support_1216": (
        loader.supports_device_counted_read(1216, 64 * 656)
        if loader is not None
        else False
    ),
}
