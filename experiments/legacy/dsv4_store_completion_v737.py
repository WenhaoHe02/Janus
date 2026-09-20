# SPDX-License-Identifier: Apache-2.0
"""Identify the last persistable prefix in chunked DSv4 prefill."""


def reaches_last_complete_prompt_chunk(
    stored_token_end: int, prompt_tokens: int, chunk_size: int
) -> bool:
    """Return whether a store batch covers the prompt's last complete chunk.

    Args:
        stored_token_end: Exclusive end of tokens eligible for this store call.
        prompt_tokens: Original prompt length, including its incomplete tail.
        chunk_size: Number of logical tokens in one persistent chunk.
    Returns:
        True once the last complete prompt chunk is covered. Prompts containing
        no complete chunk return False. This says nothing about whether model
        prefill has finished: an incomplete suffix still needs computation.
    Raises:
        ValueError: Token counts are negative or chunk_size is nonpositive.
    """
    if stored_token_end < 0 or prompt_tokens < 0 or chunk_size <= 0:
        raise ValueError("invalid token count or persistent chunk size")
    last_end = prompt_tokens // chunk_size * chunk_size
    return last_end > 0 and stored_token_end >= last_end
