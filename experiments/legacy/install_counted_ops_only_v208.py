# SPDX-License-Identifier: Apache-2.0
"""Install counted-read c_ops symbols without replacing frozen Python methods."""

for route in app.routes:
    if route.path == "/run_script":
        importer = route.endpoint.__globals__["importlib"]

builtins_module = importer.import_module("builtins")
util_module = importer.import_module("importlib.util")
sys_module = importer.import_module("sys")
ops = importer.import_module("lmcache.c_ops")
loader_module = importer.import_module(
    "lmcache.v1.gpu_connector.tutti_direct_loader"
)
name = "tutti_device_count_v137"
extension = sys_module.modules.get(name)
if extension is None:
    path = "/tmp/device_count_v137/build/tutti_device_count_v137.so"
    spec = util_module.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise builtins_module.RuntimeError("unable to load counted-read extension")
    extension = util_module.module_from_spec(spec)
    sys_module.modules[name] = extension
    spec.loader.exec_module(extension)
for target in (ops, loader_module.__dict__["_c_ops"]):
    target.tutti_read_indexed_device_count = (
        extension.tutti_read_indexed_device_count
    )
    target.tutti_read_indexed_segments_device_count = (
        extension.tutti_read_indexed_segments_device_count
    )
result = {"installed": True}
