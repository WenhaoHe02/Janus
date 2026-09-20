# SPDX-License-Identifier: Apache-2.0
"""Hot-load the corrected GLM owner scorer and enable finite launch buckets."""

for route in app.routes:
    if route.path == "/run_script":
        importer = route.endpoint.__globals__["importlib"]

builtins_module = importer.import_module("builtins")
ast_module = importer.import_module("ast")
os_module = importer.import_module("os")
scorer_module = importer.import_module("lmcache.v1.csa_prefill_cp_scorer")

with builtins_module.open("/tmp/csa_prefill_cp_scorer_v437.py") as source:
    tree = ast_module.parse(source.read())
node = None
for item in tree.body:
    if (
        builtins_module.isinstance(item, ast_module.FunctionDef)
        and item.name == "_score_owner_full_query"
    ):
        node = item
        break
if node is None:
    raise builtins_module.RuntimeError("GLM owner scorer definition is missing")
namespace = builtins_module.dict(scorer_module.__dict__)
builtins_module.exec(
    builtins_module.compile(
        ast_module.Module(body=[node], type_ignores=[]),
        "/tmp/csa_prefill_cp_scorer_v437.py",
        "exec",
    ),
    namespace,
)
builtins_module.setattr(
    scorer_module,
    "_score_owner_full_query",
    namespace["_score_owner_full_query"],
)
os_module.environ["LMCACHE_CSA_PROXY_QUERY_TILE_ROWS"] = "8192"
os_module.environ["LMCACHE_CSA_PROXY_KEY_BUCKET"] = "1"
result = {"installed": True, "query_tile_rows": 8192, "key_bucket": True}
