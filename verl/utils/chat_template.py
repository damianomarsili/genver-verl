# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def is_molmo2_chat_template(*, tokenizer: Any = None, processor: Any = None, model_path: str | None = None) -> bool:
    """Detect Molmo2 chat template usage.

    Molmo2 enforces strict user/assistant alternation and its incremental
    follow-up turns need slightly different handling from the Qwen-style
    single-user-turn shortcut used elsewhere in GENVER.
    """
    if processor is not None:
        if processor.__class__.__name__ == "Molmo2Processor":
            return True
        config = getattr(processor, "config", None)
        if getattr(config, "model_type", None) == "molmo2":
            return True

    if tokenizer is not None:
        name_or_path = str(getattr(tokenizer, "name_or_path", "") or "")
        if "molmo2" in name_or_path.lower():
            return True

    if model_path is not None and "molmo2" in str(model_path).lower():
        return True

    return False


def build_molmo2_followup_user_suffix(tokenizer: Any, text: str) -> list[int]:
    """Build the exact Molmo2 suffix for appending a follow-up user turn.

    This preserves the existing GENVER control flow while avoiding the
    Qwen-style shortcut of tokenizing a standalone user message and then
    stripping a synthetic system prompt.
    """
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return tokenizer.encode(suffix, add_special_tokens=False)


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
