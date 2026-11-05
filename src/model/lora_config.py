"""LoRA Hyperparameter Configuration and Validation Module for Qwen2.5-1.5B-Instruct.

Defines parameter-efficient fine-tuning (PEFT) configurations tailored specifically
to the Qwen2.5 dense decoder-only Transformer architecture (Qwen2ForCausalLM).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
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

# Exact architectural dimensions for Qwen/Qwen2.5-1.5B-Instruct
# Derived from Qwen2ForCausalLM with tie_word_embeddings=True and QKV biases:
#   Embeddings: 151,936 * 1536 = 233,373,696
#   28 Layers * (Attention 5,507,072 + MLP 41,287,680 + 2 RMSNorms 3,072) = 1,310,339,072
#   Final RMSNorm = 1,536
#   Total = 233,373,696 + 1,310,339,072 + 1,536 = 1,543,714,304
QWEN2_1_5B_DIMS: Dict[str, int] = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "total_base_parameters": 1543714304,
}

PINNED_QWEN_REVISION = "c644d673f4439c36291a1827471fb3be52382e26"


def detect_hardware_precision(
    major: Optional[int] = None,
    minor: Optional[int] = None,
    bf16_supported: Optional[bool] = None
) -> Tuple[bool, bool, str]:
    """
    Detects whether CUDA hardware natively supports BF16 or FP16 based on GPU compute capability.

    Rules:
    - SM 8.0 or higher AND BF16 supported: use BF16 (fp16=False, bf16=True, precision="bfloat16").
    - Below SM 8.0 (e.g. Tesla T4 SM 7.5): use FP16 (fp16=True, bf16=False, precision="float16").
    - CPU / Fallback: FP32 (fp16=False, bf16=False, precision="float32").

    Parameters:
        major: Optional override for GPU compute capability major version (for testing / mocking).
        minor: Optional override for GPU compute capability minor version (for testing / mocking).
        bf16_supported: Optional override for torch.cuda.is_bf16_supported() (for testing / mocking).

    Returns:
        (use_fp16, use_bf16, precision_display_str)
    """
    # 1. Deterministic evaluation for explicit parameters (unit tests / overrides)
    if major is not None:
        is_bf16_sup = bf16_supported if bf16_supported is not None else False
        if major >= 8 and is_bf16_sup:
            return False, True, "bfloat16"
        return True, False, "float16"

    # 2. Live environment hardware inspection
    try:
        import torch
        if torch.cuda.is_available():
            dev_major, dev_minor = torch.cuda.get_device_capability(0)
            dev_bf16_sup = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
            if dev_major >= 8 and dev_bf16_sup:
                return False, True, "bfloat16"
            return True, False, "float16"
    except Exception:
        pass

    return False, False, "float32"



def compute_qwen2_projection_shapes(dims: Optional[Dict[str, int]] = None) -> Dict[str, Tuple[int, int]]:
    """
    Computes exact (in_features, out_features) dimensions for all 7 linear projections
    in a single Qwen2 transformer block.
    """
    if dims is None:
        dims = QWEN2_1_5B_DIMS
    
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    num_heads = dims["num_attention_heads"]
    num_kv_heads = dims["num_key_value_heads"]
    head_dim = hidden_size // num_heads  # 128

    return {
        "q_proj": (hidden_size, num_heads * head_dim),       # 1536 -> 1536
        "k_proj": (hidden_size, num_kv_heads * head_dim),    # 1536 -> 256
        "v_proj": (hidden_size, num_kv_heads * head_dim),    # 1536 -> 256
        "o_proj": (num_heads * head_dim, hidden_size),       # 1536 -> 1536
        "gate_proj": (hidden_size, intermediate_size),       # 1536 -> 8960
        "up_proj": (hidden_size, intermediate_size),         # 1536 -> 8960
        "down_proj": (intermediate_size, hidden_size),       # 8960 -> 1536
    }


def calculate_lora_parameters(
    r: int = 8,
    target_modules: Optional[List[str]] = None,
    dims: Optional[Dict[str, int]] = None,
    actual_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Calculates exact theoretical and empirical LoRA parameter counts.
    
    For a linear projection W of shape (in_dim, out_dim):
      LoRA A weight shape: (r, in_dim)  -> r * in_dim params
      LoRA B weight shape: (out_dim, r) -> out_dim * r params
      Total params per adapted projection = r * (in_dim + out_dim)
    
    Never hardcodes estimates; computes directly from verified architecture dimensions.
    """
    if dims is None:
        dims = QWEN2_1_5B_DIMS
    if target_modules is None:
        target_modules = list(QWEN2_TARGET_MODULES)

    shapes = compute_qwen2_projection_shapes(dims)
    breakdown: Dict[str, Dict[str, int]] = {}
    params_per_layer = 0

    for mod in target_modules:
        if mod not in shapes:
            raise ValueError(f"Unknown module '{mod}'. Expected one of {list(shapes.keys())}")
        in_dim, out_dim = shapes[mod]
        lora_a = r * in_dim
        lora_b = out_dim * r
        mod_total = lora_a + lora_b
        breakdown[mod] = {
            "in_features": in_dim,
            "out_features": out_dim,
            "lora_a_params": lora_a,
            "lora_b_params": lora_b,
            "params_per_layer": mod_total
        }
        params_per_layer += mod_total

    num_layers = dims["num_hidden_layers"]
    calculated_trainable = params_per_layer * num_layers
    total_base = dims["total_base_parameters"]
    trainable_pct = (calculated_trainable / total_base) * 100.0

    result: Dict[str, Any] = {
        "rank": r,
        "target_modules": target_modules,
        "num_layers": num_layers,
        "params_per_layer": params_per_layer,
        "calculated_trainable_parameters": calculated_trainable,
        "total_base_parameters": total_base,
        "trainable_percentage": round(trainable_pct, 4),
        "per_module_breakdown": breakdown,
        "actual_trainable_parameters": None,
        "actual_total_parameters": None,
        "match": True
    }

    # If an instantiated PyTorch / PEFT model is provided, extract live counts
    if actual_model is not None:
        actual_trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        actual_total = sum(p.numel() for p in actual_model.parameters())
        result["actual_trainable_parameters"] = actual_trainable
        result["actual_total_parameters"] = actual_total
        result["match"] = (actual_trainable == calculated_trainable)
        if not result["match"]:
            result["discrepancy"] = actual_trainable - calculated_trainable

    return result


@dataclass
class SLMLoRAConfig:
    """Configuration container for LoRA fine-tuning hyperparameters."""
    # Model Identification
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = PINNED_QWEN_REVISION

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

    def auto_configure_precision(
        self,
        major: Optional[int] = None,
        minor: Optional[int] = None,
        bf16_supported: Optional[bool] = None
    ) -> str:
        """Automatically aligns fp16 and bf16 with available CUDA hardware."""
        use_fp16, use_bf16, prec_str = detect_hardware_precision(
            major=major, minor=minor, bf16_supported=bf16_supported
        )
        self.fp16 = use_fp16
        self.bf16 = use_bf16
        return prec_str


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

    def compute_trainable_parameters(self, actual_model: Optional[Any] = None) -> Dict[str, Any]:
        """Calculates dynamic trainable parameter counts for this configuration."""
        self.validate()
        return calculate_lora_parameters(
            r=self.r,
            target_modules=self.target_modules,
            actual_model=actual_model
        )

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
