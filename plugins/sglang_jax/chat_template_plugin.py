"""Chat template helper for tests."""

from typing import Any


def build_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )

