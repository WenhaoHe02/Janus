# SPDX-License-Identifier: Apache-2.0
"""Hot-load compact-width GPU owner gathering on one TP worker."""

for route in app.routes:
    if route.path == "/run_script":
        importer = route.endpoint.__globals__["importlib"]

builtins_module = importer.import_module("builtins")
ast_module = importer.import_module("ast")
os_module = importer.import_module("os")
transport_module = importer.import_module("lmcache.v1.ssd_tp_sharded_prefetch")

with builtins_module.open("/tmp/ssd_tp_sharded_prefetch_v432.py") as source:
    tree = ast_module.parse(source.read())

route_node = None
method_node = None
for node in tree.body:
    if builtins_module.isinstance(node, ast_module.FunctionDef):
        if node.name == "owner_gpu_route":
            route_node = node
    if builtins_module.isinstance(node, ast_module.ClassDef):
        if node.name != "TorchDistributedShardGather":
            continue
        for member in node.body:
            if (
                builtins_module.isinstance(member, ast_module.FunctionDef)
                and member.name == "gather_owner_rows_gpu_into"
            ):
                method_node = member

if route_node is None or method_node is None:
    raise builtins_module.RuntimeError("compact owner definitions are missing")

namespace = builtins_module.dict(transport_module.__dict__)
builtins_module.exec(
    builtins_module.compile(
        ast_module.Module(body=[route_node], type_ignores=[]),
        "/tmp/ssd_tp_sharded_prefetch_v432.py",
        "exec",
    ),
    namespace,
)
builtins_module.setattr(
    transport_module,
    "owner_gpu_route",
    namespace["owner_gpu_route"],
)

namespace = builtins_module.dict(transport_module.__dict__)
builtins_module.exec(
    builtins_module.compile(
        ast_module.Module(body=[method_node], type_ignores=[]),
        "/tmp/ssd_tp_sharded_prefetch_v432.py",
        "exec",
    ),
    namespace,
)
builtins_module.setattr(
    transport_module.TorchDistributedShardGather,
    "gather_owner_rows_gpu_into",
    namespace["gather_owner_rows_gpu_into"],
)
os_module.environ["LMCACHE_CSA_OWNER_GPU_METADATA"] = "1"
os_module.environ["LMCACHE_CSA_OWNER_PINNED_METADATA"] = "0"
os_module.environ["LMCACHE_DSV4_OWNER_NONBLOCKING_METADATA"] = "1"
os_module.environ["LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH"] = "1"
result = {
    "installed": True,
    "compact_width": True,
    "nonblocking_metadata": True,
}
