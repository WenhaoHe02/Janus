# SPDX-License-Identifier: Apache-2.0
"""Encode original GLM requests on CPU with the checkpoint's exact chat template."""

# Standard
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def encode_body(tokenizer: Any, body: dict[str, Any]) -> list[int]:
    """Apply native templating without truncating messages, tools, or token IDs.

    Args:
        tokenizer: Checkpoint tokenizer with its exact chat template installed.
        body: Original request body containing messages and optional tools/kwargs.
    Returns:
        Native token IDs including the model's generation prefix.
    Raises:
        ValueError: Required content or resulting token IDs are invalid.
    """
    if not body.get("messages"):
        raise ValueError("Original request has no messages")
    kwargs = dict(body.get("chat_template_kwargs") or {})
    if any(key in kwargs for key in ("chat_template", "tokenize", "truncation", "add_generation_prompt", "tools")):
        raise ValueError("Original kwargs conflict with explicit native encoding policy")
    # OpenAI wire arguments are JSON strings; the GLM checkpoint template
    # iterates argument mappings, as the serving chat parser normally supplies.
    messages = json.loads(json.dumps(body["messages"], ensure_ascii=False))
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("GLM tool arguments must decode to an object")
                function["arguments"] = arguments
    tokens = tokenizer.apply_chat_template(
        messages, tools=body.get("tools"), tokenize=True,
        add_generation_prompt=True, truncation=False, return_dict=False, **kwargs,
    )
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Native tokenizer returned invalid token IDs")
    return tokens


def main() -> None:
    """Serve a serial JSON-lines CPU encoder; do not mutate the inference server."""
    # Third Party
    from transformers import AutoTokenizer

    checkpoint = Path("/pro_model")
    snapshot = Path("/tmp/dsv4_tokenizer_snapshot")
    hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        original = (checkpoint / name).read_bytes()
        if original != (snapshot / name).read_bytes():
            raise ValueError("Served tokenizer differs from GLM checkpoint")
        hashes[name] = hashlib.sha256(original).hexdigest()
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "glm_moe_dsa":
        raise ValueError("This encoder requires the loaded GLM checkpoint")
    template = (checkpoint / "chat_template.jinja").read_text()
    hashes["chat_template.jinja"] = hashlib.sha256(template.encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
    tokenizer.chat_template = template
    print(json.dumps(dict(ready=True, assets=hashes, method="checkpoint_template_cpu_native_ids")), flush=True)
    for line in sys.stdin:
        try:
            tokens = encode_body(tokenizer, json.loads(line))
            print(json.dumps(dict(tokens=tokens)), flush=True)
        except Exception as error:
            print(json.dumps(dict(error_type=type(error).__name__, error=str(error))), flush=True)
            raise


if __name__ == "__main__":
    main()
