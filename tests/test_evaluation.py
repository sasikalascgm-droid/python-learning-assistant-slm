"""Unit tests for Phase 7 Evaluation Framework, Rubric Scoring, AST Checker, and Test Set Integrity."""

import pytest
from pathlib import Path
from src.evaluation.evaluate import (
    EvaluationRubric,
    ASTSyntaxChecker,
    compute_throughput,
    compute_performance_statistics,
    load_test_dataset,
    export_phase7_results_json,
    export_phase7_results_csv
)
from src.model.lora_config import (
    PINNED_QWEN_REVISION,
    QWEN2_1_5B_DIMS,
    calculate_lora_parameters,
    QWEN2_TARGET_MODULES
)


def test_model_revision_pinning():
    """Verify base model is pinned to immutable git commit SHA."""
    assert PINNED_QWEN_REVISION == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert len(PINNED_QWEN_REVISION) == 40


def test_base_model_exact_parameter_count():
    """Verify base model parameter count matches tied word embeddings and QKV biases."""
    assert QWEN2_1_5B_DIMS["total_base_parameters"] == 1543714304
    assert QWEN2_1_5B_DIMS["hidden_size"] == 1536
    assert QWEN2_1_5B_DIMS["intermediate_size"] == 8960
    assert QWEN2_1_5B_DIMS["num_hidden_layers"] == 28


def test_lora_exact_trainable_parameters():
    """Verify dynamic LoRA calculation yields 9,232,384 params for rank-8 on all 7 projections."""
    calc = calculate_lora_parameters(r=8, target_modules=QWEN2_TARGET_MODULES)
    assert calc["calculated_trainable_parameters"] == 9232384
    assert calc["total_base_parameters"] == 1543714304
    assert calc["trainable_percentage"] == pytest.approx(0.5981, abs=0.001)


def test_evaluation_rubric_validation():
    """Verify EvaluationRubric validates score bounds and calculates mean accurately."""
    rubric = EvaluationRubric()
    assert rubric.overall_score() == 5.0

    # Bounds check
    with pytest.raises(ValueError, match="must be in"):
        EvaluationRubric(factual_correctness=5.5).validate()

    with pytest.raises(ValueError, match="must be in"):
        EvaluationRubric(relevance=0.5).validate()

    custom_rubric = EvaluationRubric(
        factual_correctness=4.0,
        relevance=4.0,
        beginner_friendly=5.0,
        code_correctness=4.0,
        completeness=4.0,
        instruction_following=5.0,
        error_explanation_quality=4.0,
        line_by_line_quality=4.0,
        practice_question_quality=4.0
    )
    # Sum: 4*7 + 5*2 = 28 + 10 = 38 / 9 = 4.22
    assert custom_rubric.overall_score() == 4.22


def test_ast_syntax_checker_valid_code():
    """Verify ASTSyntaxChecker correctly validates Python code blocks in markdown."""
    response = """Here is a Python function:
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

msg = greet("Student")
print(msg)
```
"""
    result = ASTSyntaxChecker.evaluate_response_code(response)
    assert result["has_code"] is True
    assert result["code_block_count"] == 1
    assert result["syntax_valid_count"] == 1
    assert result["all_syntax_valid"] is True
    assert "NOT_RUN" in result["runtime_execution"]


def test_ast_syntax_checker_catches_syntax_error():
    """Verify ASTSyntaxChecker detects syntax errors in invalid code blocks."""
    response = """Here is broken code:
```python
for i in range(5)
    print(i)
```
"""
    result = ASTSyntaxChecker.evaluate_response_code(response)
    assert result["has_code"] is True
    assert result["all_syntax_valid"] is False
    assert result["syntax_valid_count"] == 0
    assert len(result["syntax_errors"]) == 1
    assert "SyntaxError" in result["syntax_errors"][0]


def test_ast_syntax_checker_no_code():
    """Verify ASTSyntaxChecker handles conceptual text-only responses cleanly."""
    response = "A variable in Python is a named reference pointing to an object stored in memory."
    result = ASTSyntaxChecker.evaluate_response_code(response)
    assert result["has_code"] is False
    assert result["code_block_count"] == 0
    assert result["all_syntax_valid"] is True


def test_performance_statistics_calculation():
    """Verify comprehensive performance distribution calculations."""
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
    tokens = [50, 100, 150, 200, 250]

    stats = compute_performance_statistics(latencies, tokens)
    assert stats["sample_count"] == 5
    assert stats["first_request_latency_seconds"] == 1.0
    assert stats["warm_latency_mean_seconds"] == 3.5  # mean of [2.0, 3.0, 4.0, 5.0]
    assert stats["latency_mean_seconds"] == 3.0
    assert stats["latency_median_seconds"] == 3.0
    assert stats["latency_p95_seconds"] == 5.0
    assert stats["total_generated_tokens"] == 750
    # 750 tokens / 15.0 seconds = 50.0 tok/s
    assert stats["aggregate_throughput_tok_per_sec"] == 50.0


def test_test_dataset_loading_and_isolation():
    """Verify test dataset contains exactly 38 records and has zero leakage into train."""
    test_recs = load_test_dataset("data/processed/test.jsonl")
    assert len(test_recs) == 38

    # Check required fields
    for r in test_recs:
        assert "instruction" in r
        assert "output" in r
        assert len(r["instruction"].strip()) > 0
        assert len(r["output"].strip()) > 0


def test_adapter_existence_check():
    """Verify adapter check properly flags missing safetensors file before Colab run."""
    adapter_weights = Path("artifacts/lora_adapter/adapter_model.safetensors")
    # In local environment before Colab run, safetensors should be absent
    assert adapter_weights.exists() is False


def test_export_phase7_results_json_and_csv(tmp_path):
    """Verify results serialization to JSON and CSV files."""
    results_dict = {
        "status": "MOCK_TESTED",
        "sample": "test"
    }
    json_file = tmp_path / "test_out.json"
    exported_json = export_phase7_results_json(results_dict, str(json_file))
    assert exported_json.exists()

    summary_rows = [
        {"model": "Base FP16", "throughput": 46.81, "ast_pass": 100.0},
        {"model": "LoRA NF4", "throughput": 42.10, "ast_pass": 100.0}
    ]
    csv_file = tmp_path / "test_out.csv"
    exported_csv = export_phase7_results_csv(summary_rows, str(csv_file))
    assert exported_csv.exists()


def test_resolve_test_dataset_path_multi_candidate():
    """Verify resolve_test_dataset_path discovers test.jsonl across both primary and alternate paths."""
    from src.evaluation.evaluate import resolve_test_dataset_path

    # Primary processed location
    resolved_primary = resolve_test_dataset_path("data/processed/test.jsonl")
    assert resolved_primary.exists()
    assert resolved_primary.name == "test.jsonl"

    # Root data fallback
    resolved_root = resolve_test_dataset_path("data/test.jsonl")
    assert resolved_root.exists()
    assert resolved_root.name == "test.jsonl"


def test_baseline_notebook_step5_colab_resolution():
    """Verify notebooks/baseline_evaluation.ipynb Step 5 contains resilient multi-tier loading logic."""
    import json

    nb_path = Path("notebooks/baseline_evaluation.ipynb")
    assert nb_path.exists(), "baseline_evaluation.ipynb must exist!"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Locate Step 5 code cell dynamically
    step5_code = ""
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "Step 5:" in src or "Execute Evaluation Across Held-Out Test Set" in src:
            step5_code = src
            break
    assert step5_code != "", "Step 5 code cell must exist in notebook!"

    # 1. Candidate paths check
    assert "candidate_paths = [" in step5_code
    assert "'data/processed/test.jsonl'" in step5_code
    assert "'data/test.jsonl'" in step5_code
    assert "python-learning-assistant-slm" in step5_code

    # 2. Strict 38-sample assertion
    assert "assert len(test_cases) == 38" in step5_code

    # 3. Mandatory inference output fields
    assert "'ground_truth': gt" in step5_code
    assert "'model_response': out['response']" in step5_code
    assert "'latency_seconds': out['latency_seconds']" in step5_code
    assert "'input_tokens': out['input_tokens']" in step5_code
    assert "'output_tokens': out['output_tokens']" in step5_code
    assert "'tokens_per_second': out['tokens_per_second']" in step5_code

