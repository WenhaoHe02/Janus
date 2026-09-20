# SPDX-License-Identifier: Apache-2.0
"""Prefill-only module scopes for GLM; independent of prediction statistics."""

# Standard
from collections.abc import Callable
import threading
from typing import Any

# Third Party
import torch
from torch.utils.hooks import RemovableHandle


class GLMLayerNvtxObserver:
    """Attach removable GPU-launch scopes without modifying tensor values.

    Args:
        forward_enabled: The existing external-KV prefill guard. A real guard
            is required so queued decode work is not miscounted as prefill.
    """

    def __init__(self, forward_enabled: Callable[[], bool]) -> None:
        if not callable(forward_enabled):
            raise ValueError("GLM profiling requires a prefill guard")
        self._forward_enabled = forward_enabled
        self._handles: list[RemovableHandle] = []
        self._seen: set[tuple[int, str]] = set()

    def bind_decoder(self, decoder: torch.nn.Module, layer_id: int) -> None:
        """Record layer, attention, FFN and available MoE child-module scopes.

        Args:
            decoder: Actual model decoder, not a synthetic probe module.
            layer_id: Zero-based physical transformer layer index.
        """
        self._register(decoder, "model_layer", layer_id)
        attention = getattr(decoder, "self_attn", None)
        mlp = getattr(decoder, "mlp", None)
        for module, name in ((attention, "attention"), (mlp, "ffn_total")):
            if isinstance(module, torch.nn.Module):
                self._register(module, name, layer_id)
        if isinstance(mlp, torch.nn.Module):
            for attribute, name in (
                ("gate", "moe_gate"),
                ("experts", "moe_routed_experts"),
                ("shared_experts", "moe_shared_experts"),
            ):
                module = getattr(mlp, attribute, None)
                if isinstance(module, torch.nn.Module):
                    self._register(module, name, layer_id)

    def close(self) -> None:
        """Remove every handle; safe for repeated lazy reattachment or shutdown."""
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._seen.clear()

    def _register(self, module: torch.nn.Module, name: str, layer_id: int) -> None:
        label = f"event={name}|layer={layer_id}|kind=glm_dsa|phase=prefill"
        identity = (id(module), label)
        if identity in self._seen:
            return
        self._seen.add(identity)
        state = threading.local()

        def enter(_module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
            stack = getattr(state, "stack", None)
            if stack is None:
                stack = []
                state.stack = stack
            stack.append(False)
            if self._forward_enabled():
                torch.cuda.nvtx.range_push(label)
                stack[-1] = True

        def leave(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any
        ) -> None:
            stack = getattr(state, "stack", None)
            if stack and stack.pop():
                torch.cuda.nvtx.range_pop()

        self._handles.append(module.register_forward_pre_hook(enter))
        self._handles.append(module.register_forward_hook(leave, always_call=True))
