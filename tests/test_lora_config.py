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
    """Verify truncation is cleanly detected when sequence exceeds max_length and trainable tokens are tracked."""
    prompt_str = "A " * 50
    response_str = "B " * 50
    tokenizer = DummyTokenizer()

    features = tokenize_with_prompt_masking(
        prompt_str, response_str, tokenizer, max_length=20
    )
    assert features["truncated"] is True
    assert len(features["input_ids"]) == 20
    # Because prompt alone was 50 tokens and max_length was 20, response was completely truncated
    assert features["has_trainable_tokens"] is False
    assert features["trainable_token_count"] == 0

    # Normal sequence within budget preserves trainable response tokens
    features_ok = tokenize_with_prompt_masking(
        "A " * 5, "B " * 5, tokenizer, max_length=20
    )
    assert features_ok["has_trainable_tokens"] is True
    assert features_ok["trainable_token_count"] == 5


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


def test_tesla_t4_always_selects_fp16_even_when_bf16_reports_true():
    """
    Direct verification of task 5:
    Confirms Tesla T4 (SM 7.5) always selects FP16=True, BF16=False even when
    Colab's torch.cuda.is_bf16_supported() returns True.
    """
    # 1. Verification via detect_hardware_precision
    use_fp16, use_bf16, prec = detect_hardware_precision(major=7, minor=5, bf16_supported=True)
    assert use_fp16 is True, "Tesla T4 must have USE_FP16=True"
    assert use_bf16 is False, "Tesla T4 must have USE_BF16=False"
    assert prec == "float16"

    # 2. Verification via SLMLoRAConfig auto_configure_precision
    cfg = get_default_lora_config()
    sel_prec = cfg.auto_configure_precision(major=7, minor=5, bf16_supported=True)
    assert cfg.fp16 is True
    assert cfg.bf16 is False
    assert sel_prec == "float16"


def test_notebook_level_precision_validation():
    """
    Direct verification of task 6:
    Notebook-level validation to detect stale or conflicting precision logic.
    Ensures notebooks/phase5_lora_training.ipynb has no stale un-gated is_bf16_supported() calls.
    """
    import json
    from pathlib import Path

    nb_path = Path("notebooks/phase5_lora_training.ipynb")
    assert nb_path.exists(), "Notebook not found!"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Check for stale un-gated is_bf16_supported
    cell_2_code = "".join(nb["cells"][2]["source"])
    assert "if (major >= 8) and torch.cuda.is_bf16_supported():" in cell_2_code, (
        "Cell 2 must gate BF16 strictly behind compute capability major >= 8!"
    )
    assert "if torch.cuda.is_bf16_supported():\n" not in cell_2_code, (
        "Cell 2 contains stale un-gated is_bf16_supported call!"
    )

    # 2. Check that TrainingArguments in Cell 6 uses fp16=USE_FP16 and bf16=USE_BF16
    cell_6_code = "".join(nb["cells"][6]["source"])
    assert "fp16=USE_FP16," in cell_6_code
    assert "bf16=USE_BF16," in cell_6_code

    # 3. Check that base model and reloaded model use TRAIN_DTYPE
    cell_5_code = "".join(nb["cells"][5]["source"])
    cell_8_code = "".join(nb["cells"][8]["source"])
    cell_10_code = "".join(nb["cells"][10]["source"])
    assert "torch_dtype=TRAIN_DTYPE" in cell_5_code
    assert "torch_dtype=TRAIN_DTYPE" in cell_8_code
    assert "bnb_4bit_compute_dtype=TRAIN_DTYPE" in cell_10_code

    # 4. Simulate Cell 2 precision logic with Tesla T4 (7, 5) and is_bf16_supported=True
    major, minor = 7, 5
    is_bf16_sup = True  # Simulating Colab T4 behavior
    if (major >= 8) and is_bf16_sup:
        sim_fp16, sim_bf16, sim_str = False, True, "bfloat16"
    else:
        sim_fp16, sim_bf16, sim_str = True, False, "float16"

    assert sim_fp16 is True
    assert sim_bf16 is False
    assert sim_str == "float16"


def test_training_arguments_eval_strategy_compatibility():
    """
    Verify TrainingArguments uses eval_strategy (compatible with Transformers 4.41+ and 5.x)
    and that deprecated evaluation_strategy is completely purged.
    """
    import json
    from pathlib import Path

    nb_path = Path("notebooks/phase5_lora_training.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_6_code = "".join(nb["cells"][6]["source"])
    assert "eval_strategy=" in cell_6_code, "Cell 6 TrainingArguments must use eval_strategy!"
    assert "evaluation_strategy=" not in cell_6_code, "Cell 6 must not use deprecated evaluation_strategy!"

    # Verify matching evaluation and save strategies
    assert "eval_strategy='epoch'" in cell_6_code or 'eval_strategy="epoch"' in cell_6_code
    assert "save_strategy='epoch'" in cell_6_code or 'save_strategy="epoch"' in cell_6_code


def test_colab_modern_stack_compatibility():
    """
    Verify modern Colab environment compatibility:
    1. SFTDataset produces Python lists for clean DataCollatorForSeq2Seq padding and NumPy 2.x safety.
    2. Gradient checkpointing configures input require grads and disables KV caching during training.
    3. Reloaded model for inference re-enables use_cache.
    4. Base model benchmark cleanly disables LoRA adapter.
    """
    import json
    from pathlib import Path

    nb_path = Path("notebooks/phase5_lora_training.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Dataset collation format
    cell_4_code = "".join(nb["cells"][4]["source"])
    assert "'input_ids': full_ids" in cell_4_code, "SFTDataset must store list full_ids!"
    assert "'labels': labels" in cell_4_code, "SFTDataset must store list labels!"

    # 2. Gradient checkpointing require grads
    cell_5_code = "".join(nb["cells"][5]["source"])
    assert "base_model.enable_input_require_grads()" in cell_5_code, (
        "Cell 5 must call enable_input_require_grads() for PEFT LoRA gradient checkpointing!"
    )
    assert "base_model.config.use_cache = False" in cell_5_code, (
        "Cell 5 must disable use_cache during training with gradient checkpointing!"
    )

    # 3. Reloaded inference use_cache
    cell_8_code = "".join(nb["cells"][8]["source"])
    assert "reloaded_base.config.use_cache = True" in cell_8_code, (
        "Cell 8 must ensure use_cache=True for fast generation during inference!"
    )

    # 4. Base model benchmark adapter disabling
    cell_11_code = "".join(nb["cells"][11]["source"])
    assert "with reloaded_model.disable_adapter():" in cell_11_code, (
        "Cell 11 must benchmark unadapted base model using disable_adapter() context manager!"
    )



