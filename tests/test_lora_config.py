"""Unit tests for LoRA Configuration, Dataset Isolation, Loss Masking, and Training Metrics."""

import pytest
from pathlib import Path
from src.model.lora_config import SLMLoRAConfig, get_default_lora_config, QWEN2_TARGET_MODULES
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
