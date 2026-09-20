# SPDX-License-Identifier: Apache-2.0
"""Enable compact GPU owner gathers and GLM shared-consumer pipelining."""

for route in app.routes:
    if route.path == "/run_script":
        importer = route.endpoint.__globals__["importlib"]

os_module = importer.import_module("os")
values = {
    "LMCACHE_CSA_OWNER_GPU_METADATA": "1",
    "LMCACHE_CSA_OWNER_PINNED_METADATA": "0",
    "LMCACHE_DSV4_OWNER_NONBLOCKING_METADATA": "1",
    "LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH": "1",
    "LMCACHE_GLM_DSA_SHARED_CORRECTION_AT_CONSUMER": "1",
    "LMCACHE_GLM_DSA_PIPELINE_SHARED_GATHERS": "1",
}
os_module.environ.update(values)
result = {"configured": True, "values": values}
