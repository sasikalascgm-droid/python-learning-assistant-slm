"""Unit Tests for AssistantService, PromptService, and AST Explanation Logic."""

import pytest
from unittest.mock import MagicMock
from src.services.prompt_service import PromptService
from src.services.assistant_service import AssistantService, FALLBACK_KNOWLEDGE_BASE
from src.services.model_service import ModelService


def test_prompt_service_ask_prompt_chatml():
    """PromptService generates correct ChatML tokens for question prompt."""
    ps = PromptService()
    prompt = ps.build_ask_prompt("What is a tuple?")
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert "What is a tuple?" in prompt
    assert "<|im_start|>assistant\n" in prompt
    assert "<|im_end|>" in prompt


def test_prompt_service_explain_code_chatml():
    """PromptService formats code inside markdown block with ChatML framing."""
    ps = PromptService()
    prompt = ps.build_explain_code_prompt("x = [1, 2, 3]\nprint(x)")
    assert "```python" in prompt
    assert "x = [1, 2, 3]" in prompt
    assert "<|im_start|>assistant\n" in prompt


def test_prompt_service_practice_prompt_chatml():
    """PromptService formats practice prompt with topic and difficulty."""
    ps = PromptService()
    prompt = ps.build_practice_prompt("recursion", "intermediate")
    assert "recursion" in prompt
    assert "intermediate" in prompt
    assert "graduated hints" in prompt.lower()


def test_assistant_service_fallback_topics():
    """AssistantService correctly maps standard topics to fallback knowledge base."""
    service = AssistantService()
    
    for topic in ["list", "dict", "loop", "function", "exception"]:
        res = service.answer_question(f"Explain how a {topic} works in Python.")
        assert res["mode"] == "development_fallback"
        assert res["model_status"] == "not_loaded"
        assert len(res["answer"]) > 50
        assert res["code_example"] is not None


def test_assistant_service_ast_analysis_constructs():
    """AssistantService identifies function defs, loops, assignments, and prints via AST."""
    code = (
        "def compute_average(numbers):\n"
        "    total = 0\n"
        "    for n in numbers:\n"
        "        total = total + n\n"
        "    if len(numbers) > 0:\n"
        "        return total / len(numbers)\n"
        "    else:\n"
        "        return 0\n"
        "print(compute_average([10, 20]))"
    )
    service = AssistantService()
    res = service.explain_code(code)
    assert res["syntax_valid"] is True
    assert res["syntax_error"] is None
    lines = res["line_by_line"]
    assert len(lines) >= 8

    # Verify statement detections
    explanations = [item["explanation"] for item in lines]
    assert any("Defines function 'compute_average'" in e for e in explanations)
    assert any("Initializes a for-loop" in e for e in explanations)
    assert any("Assigns value" in e for e in explanations)
    assert any("Evaluates boolean conditional" in e for e in explanations)
    assert any("Returns output value" in e for e in explanations)
    assert any("Outputs value to standard console" in e for e in explanations)


def test_assistant_service_ast_syntax_error_reporting():
    """AssistantService accurately flags syntax errors without executing untrusted code."""
    malformed_code = "for i in range(10)\n    print(i)"
    service = AssistantService()
    res = service.explain_code(malformed_code)
    assert res["syntax_valid"] is False
    assert res["syntax_error"] is not None
    assert "SyntaxError at line 1" in res["syntax_error"]
    assert "⚠️" in res["explanation"]


def test_assistant_service_mocked_model_loaded_inference():
    """AssistantService calls model_service.generate when model is loaded."""
    mock_model_service = MagicMock(spec=ModelService)
    mock_model_service.is_loaded = True
    mock_model_service.generate.return_value = "Mocked SLM answer from Qwen2.5-1.5B."

    service = AssistantService(model_service=mock_model_service)
    
    # Test ask
    ask_res = service.answer_question("What is a decorator?")
    assert ask_res["mode"] == "slm_inference"
    assert ask_res["model_status"] == "loaded"
    assert "Mocked SLM answer" in ask_res["answer"]
    mock_model_service.generate.assert_called_once()

    # Test explain code
    mock_model_service.generate.return_value = "Mocked code explanation."
    explain_res = service.explain_code("x = 5\nprint(x)")
    assert explain_res["mode"] == "slm_inference"
    assert explain_res["model_status"] == "loaded"
    assert "Mocked code explanation" in explain_res["explanation"]

    # Test practice
    mock_model_service.generate.return_value = "Mocked practice problem statement."
    practice_res = service.generate_practice("loops", "beginner")
    assert practice_res["mode"] == "slm_inference"
    assert practice_res["model_status"] == "loaded"
    assert "Mocked practice problem statement" in practice_res["problem_statement"]
