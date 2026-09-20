# SPDX-License-Identifier: Apache-2.0
"""Isolate asynchronous GLM proxy scratch and snapshot mutable source inputs."""

# Standard
import argparse
from pathlib import Path

# Local
from install_glm_continuation_v325 import replace_once


def install(lmcache_root: Path, vllm_root: Path) -> None:
    """Install a scoped private WorkspaceManager and immutable source snapshot.

    Args:
        lmcache_root: Installed LMCache package, never a frozen overlay.
        vllm_root: Installed vLLM package used by these same workers.
    Raises:
        RuntimeError: If the deployed official workspace or GLM proxy differs.
    """
    workspace_path = vllm_root / "v1/worker/workspace.py"
    workspace = workspace_path.read_text()
    if "scoped_workspace_manager" not in workspace:
        workspace = replace_once(
            workspace,
            "import inspect\n",
            "from contextlib import contextmanager\n"
            "from contextvars import ContextVar\n"
            "from collections.abc import Iterator\n"
            "import inspect\n",
        )
        anchor = "# Global workspace manager instance\n"
        workspace = replace_once(
            workspace,
            anchor,
            '''# Optional launch-context workspace. Default model execution is unchanged.
_scoped_workspace: ContextVar["WorkspaceManager | None"] = ContextVar(
    "lmcache_proxy_workspace", default=None
)


@contextmanager
def scoped_workspace_manager(manager: "WorkspaceManager") -> Iterator[None]:
    """Use an independent scratch allocator while submitting one async path.

    The caller must retain its manager until the owning CUDA stream completes.
    This only changes Python launch context; no CUDA synchronization is added.
    """
    token = _scoped_workspace.set(manager)
    try:
        yield
    finally:
        _scoped_workspace.reset(token)


'''
            + anchor,
        )
        anchor = "    assert _manager is not None, (\n"
        workspace = replace_once(
            workspace,
            anchor,
            """    override = _scoped_workspace.get()
    if override is not None:
        return override
"""
            + anchor,
        )
        compile(workspace, str(workspace_path), "exec")
        workspace_path.write_text(workspace)

    proxy_path = lmcache_root / "integration/vllm/glm_dsa_vllm_0202.py"
    proxy = proxy_path.read_text()
    if "GLM_PROXY_PRIVATE_WORKSPACE_V326" not in proxy:
        anchor = "            model_stream = torch.cuda.current_stream(device)\n"
        replacement = """            from vllm.v1.worker.workspace import (
                WorkspaceManager, scoped_workspace_manager,
            )
            if getattr(self, "_prediction_workspace", None) is None:
                self._prediction_workspace = WorkspaceManager(device)
                with torch.cuda.stream(self._prediction_stream):
                    self._prediction_workspace.get_simultaneous(
                        ((64 * 1024**2,), torch.uint8)
                    )
                logger.info("GLM_PROXY_PRIVATE_WORKSPACE_V326 bytes=%d", 64 * 1024**2)
            model_stream = torch.cuda.current_stream(device)
            # Residual is modified in-place by the next model layer. Record the
            # source event only after making an independent model-stream snapshot.
            activation = (
                hidden_states.clone() if residual is None
                else hidden_states + residual
            )
"""
        proxy = replace_once(proxy, anchor, replacement)
        anchor = "            hidden_states.record_stream(self._prediction_stream)\n"
        proxy = replace_once(
            proxy,
            anchor,
            "            activation.record_stream(self._prediction_stream)\n",
        )
        anchor = "            with torch.cuda.stream(self._prediction_stream):\n                diag_enabled = "
        proxy = replace_once(
            proxy,
            anchor,
            "            with (torch.cuda.stream(self._prediction_stream),\n"
            "                  scoped_workspace_manager(self._prediction_workspace)):\n"
            "                diag_enabled = ",
        )
        anchor = """                activation = (
                    hidden_states if residual is None else hidden_states + residual
                )
"""
        proxy = replace_once(proxy, anchor, "")
        compile(proxy, str(proxy_path), "exec")
        proxy_path.write_text(proxy)
    print("GLM_PROXY_WORKSPACE_V326_INSTALLED", lmcache_root, vllm_root, flush=True)


def main() -> None:
    """Install only into explicit worker package directories."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    args = parser.parse_args()
    install(args.lmcache_root, args.vllm_root)


if __name__ == "__main__":
    main()
