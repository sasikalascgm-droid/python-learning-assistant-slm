"""Training package initialization and alias module."""

from src.model.lora_config import (
    SLMLoRAConfig,
    get_default_lora_config,
    QWEN2_TARGET_MODULES,
    PINNED_QWEN_REVISION,
    calculate_lora_parameters
)
from src.model.train_lora import (
    run_lora_training,
    load_training_datasets
)

__all__ = [
    "SLMLoRAConfig",
    "get_default_lora_config",
    "QWEN2_TARGET_MODULES",
    "PINNED_QWEN_REVISION",
    "calculate_lora_parameters",
    "run_lora_training",
    "load_training_datasets"
]
