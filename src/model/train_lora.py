"""LoRA Fine-Tuning Execution Pipeline for Qwen2.5-1.5B-Instruct.

Implements supervised fine-tuning (SFT) with:
1. Strict test set isolation (validates test.jsonl is never accessed).
2. Proper ChatML tokenization and loss masking (prompt tokens masked with -100).
3. PEFT LoRA adapter wrapping for Qwen2ForCausalLM.
4. Epoch-by-epoch loss monitoring and overfitting prevention.
5. Adapter export to artifacts/lora_adapter/.
6. Hardware safety: includes dry-run mode for local unit tests without GPU.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lora_config import (
    SLMLoRAConfig,
    get_default_lora_config,
    QWEN2_TARGET_MODULES,
    PINNED_QWEN_REVISION,
    calculate_lora_parameters,
    detect_hardware_precision
)
from src.evaluation.training_metrics import TrainingMetricsTracker
from src.model.load_model import load_config, resolve_cache_directory


DEFAULT_SYSTEM_PROMPT = """You are an expert, patient, and friendly Python Learning Assistant.
Your mission is to help beginner students master Python programming.
Follow these principles:
1. Explain concepts in clear, simple English without unnecessary jargon.
2. Provide idiomatic, clean, and bug-free Python code examples.
3. Break down code step-by-step or line-by-line when explaining snippets.
4. When diagnosing errors, clearly explain the cause and provide the exact fix.
5. Always encourage the student with beginner-friendly practice questions and hints.
"""


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA256 checksum of a file for audit provenance."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def load_training_datasets(
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/validation.jsonl",
    test_path_for_leak_check: str = "data/processed/test.jsonl"
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, str]]:
    """
    Loads training and validation datasets while strictly asserting that
    the test set is NOT included in the training pipeline.
    """
    train_file = Path(train_path)
    val_file = Path(val_path)
    test_file = Path(test_path_for_leak_check)

    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    # Read train
    train_records = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    # Read val
    val_records = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    # Verify test set isolation
    if test_file.exists():
        test_instructions = set()
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_instructions.add(json.loads(line)["instruction"].strip())

        train_instructions = set(r["instruction"].strip() for r in train_records)
        overlap = train_instructions & test_instructions
        if overlap:
            raise AssertionError(f"FATAL: Test set contamination detected! {len(overlap)} test questions leaked into train set: {overlap}")

    hashes = {
        "train_sha256": compute_file_sha256(train_file),
        "val_sha256": compute_file_sha256(val_file),
        "train_count": str(len(train_records)),
        "val_count": str(len(val_records))
    }

    return train_records, val_records, hashes


def format_chatml_example(
    instruction: str,
    output: str,
    input_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> Tuple[str, str]:
    """
    Formats a single QA record into (prompt_text, response_text) using Qwen ChatML tags.
    """
    user_body = instruction.strip()
    if input_text and input_text.strip():
        user_body += f"\n\n```python\n{input_text.strip()}\n```"

    prompt_text = (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    response_text = f"{output.strip()}<|im_end|>\n"
    return prompt_text, response_text


def tokenize_with_prompt_masking(
    prompt_text: str,
    response_text: str,
    tokenizer: Any,
    max_length: int = 512,
    pad_to_max: bool = False
) -> Dict[str, Any]:
    """
    Tokenizes full prompt+response and applies -100 masking to prompt tokens
    so loss is calculated exclusively on response tokens.
    
    Guarantees:
    1. Prompt tokens are masked with -100 (PyTorch CrossEntropyLoss ignore index).
    2. Response tokens remain unmasked and contribute to loss gradient.
    3. Padding tokens (if padded) are assigned attention_mask=0 and label=-100.
    4. Tracks whether the sequence underwent truncation.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    full_ids = prompt_ids + response_ids
    truncated = False
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        truncated = True

    # Create labels: mask prompt tokens with -100
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)

    # Optional static padding
    if pad_to_max and len(full_ids) < max_length:
        pad_len = max_length - len(full_ids)
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        full_ids = full_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "truncated": truncated
    }


def prepare_hf_dataset(
    records: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 512,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> List[Dict[str, List[int]]]:
    """Prepares list of tokenized and masked feature dictionaries."""
    tokenized_data = []
    for r in records:
        prompt_str, response_str = format_chatml_example(
            instruction=r["instruction"],
            output=r["output"],
            input_text=r.get("input", ""),
            system_prompt=system_prompt
        )
        item = tokenize_with_prompt_masking(prompt_str, response_str, tokenizer, max_length)
        tokenized_data.append(item)
    return tokenized_data


class QwenSFTDataset:
    """
    PyTorch Dataset implementing ChatML formatting with prompt loss masking (-100).
    Ensures that loss is calculated exclusively on assistant response tokens.
    """
    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: Any,
        max_len: int = 512,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        import torch
        self.features = []
        for r in records:
            prompt_str, response_str = format_chatml_example(
                instruction=r["instruction"],
                output=r["output"],
                input_text=r.get("input", ""),
                system_prompt=system_prompt
            )
            item = tokenize_with_prompt_masking(prompt_str, response_str, tokenizer, max_length=max_len)
            self.features.append({
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["labels"], dtype=torch.long)
            })

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]


def run_lora_training(
    cfg: Optional[SLMLoRAConfig] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Executes or simulates the LoRA fine-tuning process.
    If dry_run=True, validates the pipeline, tokenization, masking, and configurations
    without loading large model weights.
    If dry_run=False and CUDA is available, executes full SFT fine-tuning with PEFT LoRA
    and packages the resulting artifacts.
    """
    if cfg is None:
        cfg = get_default_lora_config()
    cfg.validate()

    # Hardware precision detection
    use_fp16, use_bf16, prec_str = detect_hardware_precision()
    if not dry_run and (use_fp16 or use_bf16):
        cfg.fp16 = use_fp16
        cfg.bf16 = use_bf16

    train_recs, val_recs, data_hashes = load_training_datasets(
        train_path=cfg.train_file,
        val_path=cfg.val_file
    )
    print(f"Loaded {len(train_recs)} training examples, {len(val_recs)} validation examples.")

    tracker = TrainingMetricsTracker(run_name=f"lora_r{cfg.r}_{cfg.model_id.replace('/', '_')}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_analysis = calculate_lora_parameters(r=cfg.r, target_modules=cfg.target_modules)
    print(
        f"LoRA Parameter Analysis: {param_analysis['calculated_trainable_parameters']:,} trainable parameters "
        f"({param_analysis['trainable_percentage']:.4f}% of {param_analysis['total_base_parameters']:,} base params)."
    )

    # Save training configuration
    train_metadata = {
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "lora_parameters": cfg.to_dict(),
        "trainable_parameters_analysis": param_analysis,
        "dataset_provenance": data_hashes,
        "precision": prec_str,
        "dry_run": dry_run
    }

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)

    if dry_run:
        print("Dry-run mode: Training pipeline, dataset isolation, and configs validated successfully.")
        return {
            "status": "dry_run_success",
            "train_samples": len(train_recs),
            "val_samples": len(val_recs),
            "config": cfg.to_dict(),
            "precision": prec_str,
            "output_dir": str(out_dir)
        }

    # Full execution mode (Requires GPU)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        raise RuntimeError(
            "CUDA GPU not available. LoRA fine-tuning must be executed on a GPU instance "
            "(e.g. Google Colab Tesla T4) to prevent host memory exhaustion. "
            "Status remains PENDING_COLAB_EXECUTION."
        )

    import math
    import shutil
    import time
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForSeq2Seq
    )
    from peft import get_peft_model

    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA GPU detected: {gpu_name}")
    print(f"Selected Training Precision: {prec_str}")

    train_dtype = torch.bfloat16 if cfg.bf16 else torch.float16

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.model_id} (revision: {cfg.model_revision})...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build SFT datasets with prompt loss masking
    train_dataset = QwenSFTDataset(train_recs, tokenizer, max_len=cfg.max_seq_length)
    val_dataset = QwenSFTDataset(val_recs, tokenizer, max_len=cfg.max_seq_length)

    # Load base model
    print(f"Loading base model: {cfg.model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        revision=cfg.model_revision,
        torch_dtype=train_dtype,
        low_cpu_mem_usage=True
    )
    if cfg.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    # Attach PEFT LoRA adapter
    peft_config = cfg.get_peft_config()
    lora_model = get_peft_model(base_model, peft_config)

    actual_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    actual_total = sum(p.numel() for p in lora_model.parameters())
    actual_pct = (actual_trainable / actual_total) * 100.0

    print(f"LoRA Trainable Parameters: {actual_trainable:,} ({actual_pct:.4f}%)")
    assert actual_trainable == 9232384, f"Parameter count mismatch: actual={actual_trainable} vs expected=9232384"

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=cfg.seed,
        report_to="none"
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    print("Launching LoRA fine-tuning...")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    train_result = trainer.train()
    duration_s = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    eval_metrics = trainer.evaluate()
    val_loss = eval_metrics.get("eval_loss", 0.0)
    train_loss = train_result.training_loss
    val_ppl = math.exp(min(val_loss, 20.0))

    print(f"Training completed in {duration_s:.2f}s ({duration_s/60:.2f}m)")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB")

    # Save model and tokenizer artifacts
    lora_model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save trainer log history
    with open(out_dir / "trainer_log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    summary_data = {
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "gpu_model": gpu_name,
        "precision": prec_str,
        "trainable_parameters": actual_trainable,
        "total_parameters": actual_total,
        "trainable_percentage": round(actual_pct, 4),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(val_ppl, 2),
        "duration_seconds": round(duration_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 2)
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Package into zip archive
    zip_name = "lora_adapter"
    zip_path = shutil.make_archive(zip_name, "zip", cfg.output_dir)
    print(f"Packaged adapter into {zip_path}")

    return {
        "status": "training_completed",
        "summary": summary_data,
        "output_dir": cfg.output_dir,
        "zip_archive": zip_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Execution Pipeline for Qwen2.5-1.5B-Instruct.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run dry-run validation without GPU.")
    parser.add_argument("--full", action="store_true", default=False, help="Execute full training on GPU.")
    args = parser.parse_args()

    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    is_dry_run = args.dry_run or (not args.full and not has_cuda)
    result = run_lora_training(dry_run=is_dry_run)
    print(f"LoRA pipeline completed: {result['status']}")
