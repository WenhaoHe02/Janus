# SPDX-License-Identifier: Apache-2.0
"""Install opt-in GLM discovery A/B before worker import, retaining backups."""

# Standard
import argparse
import ast
from pathlib import Path
import shutil


def main() -> None:
    """Validate actual adapter boundaries then append isolated diagnostic hooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()
    path = args.lmcache_root / "integration/vllm/vllm_v1_adapter.py"
    source = path.read_text()
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "_deepseek_decoder_layers"
               for n in tree.body):
        raise ValueError("Actual DS discovery not found")
    if "DISCOVERY_PROBE_V1039" in source:
        raise ValueError("Probe already installed")
    addition = '''
# DISCOVERY_PROBE_V1039 -- diagnostic only, not a formal performance result.
import os as discovery_os_v1039
if discovery_os_v1039.getenv("LMCACHE_GLM_DISCOVERY_DIAGNOSTIC_DIR"):
    from pathlib import Path as DiscoveryPathV1039
    from lmcache.integration.vllm.discovery_probe_v1039 import wrap_discovery
    discovery_model_type_v1039 = None
    discovery_constructor_v1039 = LMCacheConnectorV1Impl.__init__
    def discovery_init_v1039(self, vllm_config, *args, **kwargs):
        global discovery_model_type_v1039
        discovery_model_type_v1039 = getattr(vllm_config.model_config.hf_config, "model_type", None)
        return discovery_constructor_v1039(self, vllm_config, *args, **kwargs)
    LMCacheConnectorV1Impl.__init__ = discovery_init_v1039
    _deepseek_decoder_layers = wrap_discovery(
        _deepseek_decoder_layers, lambda: discovery_model_type_v1039,
        DiscoveryPathV1039(discovery_os_v1039.environ["LMCACHE_GLM_DISCOVERY_DIAGNOSTIC_DIR"]))
'''
    compile(source + addition, str(path), "exec")
    backup = path.with_suffix(".py.before_discovery_v1039")
    if backup.exists():
        raise FileExistsError(backup)
    shutil.copy2(path, backup)
    shutil.copy2("/tmp/discovery_probe_v1039.py", path.parent / "discovery_probe_v1039.py")
    path.write_text(source + addition)
    print("DISCOVERY_PROBE_V1039_INSTALLED", flush=True)


if __name__ == "__main__":
    main()
