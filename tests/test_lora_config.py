"""Unit tests for LoRA Configuration, Dataset Isolation, Loss Masking, and Training Metrics."""

import pytest
from pathlib import Path
from src.model.lora_config import (
    SLMLoRAConfig,
    get_default_lora_config,
    QWEN2_TARGET_MODULES,
    QWEN2_1_5B_DIMS,
    calculate_lora_parameters,
    compute_qwen2_projection_shapes,
    detect_hardware_precision
)
from src.model.train_lora import (
    load_training_datasets,
    format_chatml_example,
    tokenize_with_prompt_masking,
    DEFAULT_SYSTEM_PROMPT
)
from src.evaluation.training_metrics import TrainingMetricsTracker, EpochMetric


class DummyTokenizer:
    """Mock tokenizer for offline unit testing without downloading 3GB weights."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        # Mock tokenization based on whitespace + special tags
        return [hash(w) % 10000 + 2 for w in text.split()]


def test_default_lora_config():
    """Verify default LoRA hyperparameters adhere to conservative SFT recommendations."""
    cfg = get_default_lora_config()
    assert cfg.r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.05
    assert cfg.learning_rate == 2e-4
    assert cfg.num_train_epochs == 2
    assert cfg.task_type == "CAUSAL_LM"


def test_target_modules_match_qwen2_architecture():
    """Verify LoRA target modules strictly correspond to Qwen2ForCausalLM projection layers."""
    cfg = get_default_lora_config()
    expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert cfg.target_modules == expected
    for mod in cfg.target_modules:
        assert mod in QWEN2_TARGET_MODULES


def test_lora_config_validation_catches_invalid_values():
    """Verify parameter validation rejects negative or invalid hyperparameter bounds."""
    with pytest.raises(ValueError):
        SLMLoRAConfig(r=0).validate()

    with pytest.raises(ValueError):
        SLMLoRAConfig(lora_alpha=-1).validate()

    with pytest.raises(ValueError):
        SLMLoRAConfig(lora_dropout=1.5).validate()

    with pytest.raises(ValueError):
        SLMLoRAConfig(target_modules=["invalid_module_name"]).validate()


def test_dataset_loading_and_test_isolation():
    """Verify training dataset loading and assert zero test set contamination."""
    train_recs, val_recs, hashes = load_training_datasets(
        train_path="data/processed/train.jsonl",
        val_path="data/processed/validation.jsonl",
        test_path_for_leak_check="data/processed/test.jsonl"
    )

    assert len(train_recs) == 306
    assert len(val_recs) == 38
    assert "train_sha256" in hashes
    assert "val_sha256" in hashes


def test_chatml_formatting_and_loss_masking():
    """Verify prompt formatting and ensure prompt tokens are masked with -100."""
    instruction = "What is a Python list?"
    output = "A list is an ordered mutable sequence."
    prompt_str, response_str = format_chatml_example(instruction, output)

    assert "<|im_start|>user" in prompt_str
    assert instruction in prompt_str
    assert output in response_str
    assert "<|im_end|>" in response_str

    tokenizer = DummyTokenizer()
    features = tokenize_with_prompt_masking(prompt_str, response_str, tokenizer, max_length=128)

    labels = features["labels"]
    input_ids = features["input_ids"]

    assert len(labels) == len(input_ids)
    # The first portion (prompt) must be -100
    assert labels[0] == -100
    # The response portion must contain active token IDs (not -100)
    assert labels[-1] != -100
    assert any(lbl != -100 for lbl in labels)


def test_overfitting_detector():
    """Verify overfitting detection logic flags divergence between train and val loss."""
    tracker = TrainingMetricsTracker(run_name="test_run")
    # Epoch 1: Normal convergence
    tracker.record_epoch(epoch=1, train_loss=1.80, val_loss=1.90, duration_seconds=30.0)
    res1 = tracker.detect_overfitting()
    assert res1["overfitting_detected"] is False

    # Epoch 2: Divergence (train loss drops to 0.90, val loss spikes to 2.40)
    tracker.record_epoch(epoch=2, train_loss=0.90, val_loss=2.40, duration_seconds=30.0)
    res2 = tracker.detect_overfitting(threshold_ratio=1.15)
    assert res2["overfitting_detected"] is True
    assert "diverged" in res2["warning"]


def test_metrics_tracker_export(tmp_path):
    """Verify metrics tracker serializes summary to JSON properly."""
    tracker = TrainingMetricsTracker(run_name="unit_test")
    tracker.log_hardware_environment("Tesla T4", 15.36, "2.2.0", "0.10.0")
    tracker.record_epoch(epoch=1, train_loss=1.5, val_loss=1.6, duration_seconds=45.0, gpu_peak_mb=3200.0)

    summary_file = tmp_path / "test_summary.json"
    summary = tracker.export_summary(str(summary_file))

    assert summary_file.exists()
    assert summary["run_name"] == "unit_test"
    assert summary["hardware"]["gpu_name"] == "Tesla T4"
    assert len(summary["epochs"]) == 1
    assert summary["epochs"][0]["train_perplexity"] > 0.0


def test_exact_lora_parameter_math():
    """Verify LoRA parameter calculations match exact projection tensor math for Qwen2.5-1.5B."""
    res = calculate_lora_parameters(r=8, target_modules=QWEN2_TARGET_MODULES)

    # 1. Total trainable parameter count
    assert res["calculated_trainable_parameters"] == 9232384
    # 2. Total base parameter count (tied embeddings + QKV biases: 1,543,714,304)
    assert res["total_base_parameters"] == 1543714304
    # 3. Trainable percentage
    assert res["trainable_percentage"] == pytest.approx(0.5981, abs=0.001)
    # 4. Parameters per layer
    assert res["params_per_layer"] == 329728
    # 5. Layers
    assert res["num_layers"] == 28


def test_per_module_parameter_breakdown():
    """Verify each target projection module's LoRA A & B dimensions."""
    res = calculate_lora_parameters(r=8, target_modules=QWEN2_TARGET_MODULES)
    bd = res["per_module_breakdown"]

    # Attention projections
    assert bd["q_proj"]["params_per_layer"] == 24576  # 8*(1536 + 1536)
    assert bd["k_proj"]["params_per_layer"] == 14336  # 8*(1536 + 256)
    assert bd["v_proj"]["params_per_layer"] == 14336  # 8*(1536 + 256)
    assert bd["o_proj"]["params_per_layer"] == 24576  # 8*(1536 + 1536)

    # MLP projections
    assert bd["gate_proj"]["params_per_layer"] == 83968  # 8*(1536 + 8960)
    assert bd["up_proj"]["params_per_layer"] == 83968    # 8*(1536 + 8960)
    assert bd["down_proj"]["params_per_layer"] == 83968  # 8*(8960 + 1536)


def test_padding_loss_masking_exclusion():
    """Verify padding tokens have attention_mask=0 and labels=-100."""
    instruction = "Hello"
    output = "World"
    prompt_str, response_str = format_chatml_example(instruction, output)
    tokenizer = DummyTokenizer()

    # Prompt+response is ~65 tokens with default system prompt; max_length=128 triggers padding
    features = tokenize_with_prompt_masking(
        prompt_str, response_str, tokenizer, max_length=128, pad_to_max=True
    )

    labels = features["labels"]
    attn = features["attention_mask"]
    input_ids = features["input_ids"]

    assert len(input_ids) == 128
    assert len(labels) == 128
    assert len(attn) == 128

    # Padded tail elements must have attention_mask=0 and labels=-100
    assert attn[-1] == 0
    assert labels[-1] == -100
    # Response tokens in the middle must be active
    assert any(m == 1 for m in attn)
    assert any(l != -100 for l in labels)


def test_test_set_leak_assertion_triggers(tmp_path):
    """Verify load_training_datasets raises AssertionError if any test item is in train."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    test_file = tmp_path / "test.jsonl"

    leak_item = {"instruction": "Leaked question", "input": "", "output": "Ans"}
    safe_item = {"instruction": "Safe question", "input": "", "output": "Ans"}

    import json
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(leak_item) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(safe_item) + "\n")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(leak_item) + "\n")

    with pytest.raises(AssertionError, match="FATAL: Test set contamination detected"):
        load_training_datasets(
            train_path=str(train_file),
            val_path=str(val_file),
            test_path_for_leak_check=str(test_file)
        )


def test_truncation_detection():
    """Verify truncation is cleanly detected when sequence exceeds max_length."""
    prompt_str = "A " * 50
    response_str = "B " * 50
    tokenizer = DummyTokenizer()

    features = tokenize_with_prompt_masking(
        prompt_str, response_str, tokenizer, max_length=20
    )
    assert features["truncated"] is True
    assert len(features["input_ids"]) == 20


def test_precision_detection_tesla_t4_always_fp16():
    """Verify Tesla T4 (SM 7.5) always selects FP16=True, BF16=False even if is_bf16_supported returns True."""
    # Colab anomaly: is_bf16_supported returns True on T4
    use_fp16, use_bf16, prec = detect_hardware_precision(major=7, minor=5, bf16_supported=True)
    assert use_fp16 is True
    assert use_bf16 is False
    assert prec == "float16"

    # Standard check: is_bf16_supported returns False
    use_fp16_2, use_bf16_2, prec_2 = detect_hardware_precision(major=7, minor=5, bf16_supported=False)
    assert use_fp16_2 is True
    assert use_bf16_2 is False
    assert prec_2 == "float16"

    # Auto-configure precision on SLMLoRAConfig
    cfg = get_default_lora_config()
    sel_prec = cfg.auto_configure_precision(major=7, minor=5, bf16_supported=True)
    assert cfg.fp16 is True
    assert cfg.bf16 is False
    assert sel_prec == "float16"


def test_precision_detection_ampere_sm80():
    """Verify Ampere (SM 8.0) selects BF16 only when supported, otherwise falls back to FP16."""
    # SM 8.0 with BF16 support
    use_fp16, use_bf16, prec = detect_hardware_precision(major=8, minor=0, bf16_supported=True)
    assert use_fp16 is False
    assert use_bf16 is True
    assert prec == "bfloat16"

    # SM 8.0 without BF16 support (fallback)
    use_fp16_fb, use_bf16_fb, prec_fb = detect_hardware_precision(major=8, minor=0, bf16_supported=False)
    assert use_fp16_fb is True
    assert use_bf16_fb is False
    assert prec_fb == "float16"

    # Auto-configure precision on SLMLoRAConfig
    cfg = get_default_lora_config()
    sel_prec = cfg.auto_configure_precision(major=8, minor=0, bf16_supported=True)
    assert cfg.fp16 is False
    assert cfg.bf16 is True
    assert sel_prec == "bfloat16"


def test_precision_detection_cpu_fallback():
    """Verify CPU fallback when CUDA is not present."""
    use_fp16, use_bf16, prec = detect_hardware_precision()
    # On the test host (no discrete GPU), must be CPU fallback
    assert use_fp16 is False
    assert use_bf16 is False
    assert prec == "float32"

