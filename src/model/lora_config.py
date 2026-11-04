"""LoRA Hyperparameter Configuration and Validation Module for Qwen2.5-1.5B-Instruct.

Defines parameter-efficient fine-tuning (PEFT) configurations tailored specifically
to the Qwen2.5 dense decoder-only Transformer architecture (Qwen2ForCausalLM).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


# Verified linear projection module names for Qwen2ForCausalLM architecture
QWEN2_TARGET_MODULES = [
    "q_proj",      # Query projection (self-attention)
    "k_proj",      # Key projection (self-attention)
    "v_proj",      # Value projection (self-attention)
    "o_proj",      # Output projection (self-attention)
    "gate_proj",   # Gate projection (SwiGLU MLP)
    "up_proj",     # Up projection (SwiGLU MLP)
    "down_proj"    # Down projection (SwiGLU MLP)
]


@dataclass
class SLMLoRAConfig:
    """Configuration container for LoRA fine-tuning hyperparameters."""
    # Model Identification
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "main"

    # LoRA Architecture Parameters
    r: int = 8                          # LoRA rank dimension
    lora_alpha: int = 16                # LoRA scaling factor (alpha = 2 * r)
    lora_dropout: float = 0.05          # Regularization dropout for adapter layers
    bias: str = "none"                  # No bias tuning to preserve base model stability
    task_type: str = "CAUSAL_LM"        # Standard causal language modeling task
    target_modules: List[str] = field(default_factory=lambda: list(QWEN2_TARGET_MODULES))

    # Optimization & Batching Parameters
    learning_rate: float = 2e-4         # Standard stable learning rate for rank-8 LoRA
    num_train_epochs: int = 2           # Conservative 2-epoch budget to prevent overfitting on 306 samples
    per_device_train_batch_size: int = 2 # Fits comfortably in 15GB T4 GPU VRAM
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4 # Effective batch size = 2 * 4 = 8
    max_seq_length: int = 512           # Covers 99% of beginner QA prompts
    warmup_ratio: float = 0.05          # 5% linear warmup schedule
    weight_decay: float = 0.01          # L2 regularization weight decay
    seed: int = 42

    # Precision & Runtime
    fp16: bool = True                   # Native on Turing T4 GPUs
    bf16: bool = False                  # Requires Ampere (SM 8.0+) or newer
    gradient_checkpointing: bool = True # Reduces VRAM usage during backpropagation

    # Paths
    output_dir: str = "artifacts/lora_adapter"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/validation.jsonl"

    def validate(self) -> None:
        """Validates configuration values against architectural and training constraints."""
        if self.r <= 0 or self.r > 64:
            raise ValueError(f"LoRA rank 'r' must be between 1 and 64, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be positive, got {self.num_train_epochs}")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        
        # Verify all target modules are recognized in Qwen2 architecture
        for m in self.target_modules:
            if m not in QWEN2_TARGET_MODULES:
                raise ValueError(f"Unrecognized target module '{m}' for Qwen2 architecture. Valid: {QWEN2_TARGET_MODULES}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to dictionary."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.per_device_train_batch_size * self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "output_dir": self.output_dir
        }

    def get_peft_config(self) -> Any:
        """Returns a PEFT LoraConfig object."""
        self.validate()
        try:
            from peft import LoraConfig
            return LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type=self.task_type
            )
        except ImportError:
            raise ImportError("peft package is not installed. Please install peft to construct LoraConfig.")


def get_default_lora_config() -> SLMLoRAConfig:
    """Factory function returning validated standard LoRA configuration."""
    cfg = SLMLoRAConfig()
    cfg.validate()
    return cfg
