"""Unit tests for Phase 6 Model Optimization, Quantization Configs, AST Validator, and Throughput Metrics."""

import pytest
from src.optimization.quantize import (
    QuantizationConfig,
    GenerationConfig,
    check_quantization_support,
    load_quantized_model_with_adapter,
    ASTCodeValidator,
    calculate_throughput,
    generate_benchmark_summary
)


def test_quantization_config_defaults_and_validation():
    """Verify default 4-bit NF4 configuration and constraint validation."""
    cfg = QuantizationConfig()
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.bnb_4bit_compute_dtype == "float16"

    # Invalid quant_type
    with pytest.raises(ValueError, match="Invalid quant_type"):
        QuantizationConfig(bnb_4bit_quant_type="int8").validate()

    # Invalid compute_dtype
    with pytest.raises(ValueError, match="Invalid compute_dtype"):
        QuantizationConfig(bnb_4bit_compute_dtype="int4").validate()


def test_generation_config_validation():
    """Verify inference generation hyperparameters bounds checking."""
    cfg = GenerationConfig()
    cfg.validate()

    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        GenerationConfig(max_new_tokens=0).validate()

    with pytest.raises(ValueError, match="temperature must be between"):
        GenerationConfig(temperature=-0.5).validate()

    with pytest.raises(ValueError, match="top_p must be in"):
        GenerationConfig(top_p=1.5).validate()

    with pytest.raises(ValueError, match="repetition_penalty must be >= 1.0"):
        GenerationConfig(repetition_penalty=0.8).validate()


def test_quantization_support_diagnostics():
    """Verify check_quantization_support returns diagnostic dict without crashing."""
    diag = check_quantization_support()
    assert "supported" in diag
    assert "cuda_available" in diag
    assert "reason" in diag
    assert isinstance(diag["supported"], bool)
    # On laptop host (CPU only, no CUDA GPU), must report False safely
    assert diag["supported"] is False
    assert len(diag["reason"]) > 0


def test_adapter_loading_error_handling():
    """Verify graceful error reporting when adapter directory is missing."""
    with pytest.raises(Exception):
        load_quantized_model_with_adapter(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_dir="non_existent_adapter_dir"
        )


def test_ast_code_validator_syntax():
    """Verify AST parser correctly differentiates valid and invalid Python snippets."""
    valid_code = "def add(a, b):\n    return a + b\n\nresult = add(2, 3)"
    syntax_ok, err = ASTCodeValidator.validate_syntax(valid_code)
    assert syntax_ok is True
    assert err is None

    invalid_code = "def broken(:\n    pass"
    syntax_ok, err = ASTCodeValidator.validate_syntax(invalid_code)
    assert syntax_ok is False
    assert "SyntaxError" in err


def test_ast_code_validator_code_extraction():
    """Verify extraction of python code blocks from markdown responses."""
    text = "Here is an example:\n```python\nx = [1, 2, 3]\nprint(len(x))\n```\nAnd another:\n```\ny = 10\n```"
    blocks = ASTCodeValidator.extract_code_blocks(text)
    assert len(blocks) == 2
    assert "print(len(x))" in blocks[0]
    assert "y = 10" in blocks[1]


def test_ast_container_sandbox_enforcement():
    """
    Verify security directive: Host-level execution of untrusted generated code
    is strictly refused without operating-system-level container isolation.
    """
    sample_code = "print('Hello World')"
    # When container engine is unavailable on host
    ok, err = ASTCodeValidator.execute_in_container_sandbox(
        sample_code, container_engine_available=False
    )
    assert ok is False
    assert "RUNTIME_EXECUTION_NOT_RUN" in err
    assert "Host-level execution of untrusted generated code prohibited" in err


def test_aggregate_throughput_calculation():
    """Verify aggregate throughput calculation is tokens / seconds, not mean of rates."""
    rate = calculate_throughput(total_tokens=1000, total_seconds=20.0)
    assert rate == 50.0

    # Zero seconds division guard
    rate_zero = calculate_throughput(total_tokens=100, total_seconds=0.0)
    assert rate_zero == 0.0


def test_benchmark_summary_compilation():
    """Verify benchmark summary compiles stats accurately across test requests."""
    results = [
        {"latency_seconds": 2.0, "generated_tokens": 100, "ast_syntax_valid": True},
        {"latency_seconds": 4.0, "generated_tokens": 200, "ast_syntax_valid": True},
        {"latency_seconds": 6.0, "generated_tokens": 300, "ast_syntax_valid": False}
    ]
    gen_cfg = GenerationConfig()
    summary = generate_benchmark_summary(
        config_name="mock_test",
        results=results,
        load_time_seconds=12.5,
        peak_memory_mb=4500.0,
        precision="float16",
        quantization="none",
        gen_config=gen_cfg
    )

    assert summary["config_name"] == "mock_test"
    assert summary["sample_count"] == 3
    assert summary["total_generated_tokens"] == 600
    assert summary["latency_mean_seconds"] == 4.0
    assert summary["latency_median_seconds"] == 4.0
    # 600 tokens / 12.0 seconds = 50.0 tok/s
    assert summary["aggregate_throughput_tok_per_sec"] == 50.0
    # 2 out of 3 AST valid = 66.67%
    assert summary["ast_syntax_pass_rate_pct"] == 66.67
    assert summary["model_load_time_seconds"] == 12.5
