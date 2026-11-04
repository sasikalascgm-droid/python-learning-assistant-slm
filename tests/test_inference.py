"""Unit tests for Model Inference Pipeline, Prompt Templates, and Generation Settings."""

import pytest
from src.model.inference import format_qwen_chat_prompt, DEFAULT_SYSTEM_PROMPT
from src.model.load_model import load_config, resolve_cache_directory, check_host_memory_headroom


def test_qwen_chat_prompt_formatting():
    """Verify standard Qwen2.5 ChatML prompt formatting with system and user roles."""
    instruction = "What is a Python dictionary?"
    prompt = format_qwen_chat_prompt(instruction)

    assert "<|im_start|>system" in prompt
    assert "<|im_end|>" in prompt
    assert "<|im_start|>user" in prompt
    assert instruction in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_qwen_chat_prompt_with_code_input():
    """Verify user code inputs are formatted in markdown blocks within the prompt."""
    instruction = "Explain this code:"
    code = "x = [1, 2, 3]\nprint(len(x))"
    prompt = format_qwen_chat_prompt(instruction, input_text=code)

    assert instruction in prompt
    assert "```python" in prompt
    assert "print(len(x))" in prompt


def test_config_loading():
    """Verify model configuration contains valid inference and model parameters."""
    cfg = load_config()
    assert "model" in cfg
    assert "inference" in cfg
    assert cfg["model"]["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["inference"]["max_new_tokens"] == 512
    assert cfg["inference"]["temperature"] == 0.7


def test_memory_headroom_check():
    """Verify host memory query returns sensible non-zero numbers."""
    mem = check_host_memory_headroom()
    assert "total_ram_gb" in mem
    assert "avail_ram_gb" in mem
    assert mem["total_ram_gb"] > 0
    assert mem["avail_ram_gb"] > 0


def test_cache_directory_resolution():
    """Verify cache directory resolution returns a valid string path."""
    cache_dir = resolve_cache_directory()
    assert isinstance(cache_dir, str)
    assert len(cache_dir) > 0
