"""Model Lifecycle, Hardware Auditing, and Inference Service."""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lora_config import PINNED_QWEN_REVISION
from src.model.load_model import load_config, check_host_memory_headroom, resolve_cache_directory


class ModelService:
    """Manages SLM lifecycle lazily with hardware safeguards."""

    _instance = None

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.model_id = self.config.get("model", {}).get("model_id", "Qwen/Qwen2.5-1.5B-Instruct")
        self.revision = PINNED_QWEN_REVISION
        self.adapter_path = Path("artifacts/lora_adapter")
        self.adapter_weights_file = self.adapter_path / "adapter_model.safetensors"

        # Model instance state (starts UNLOADED to protect host memory)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.adapter_loaded = False
        self.device = "none"

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Singleton accessor for ModelService."""
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def is_adapter_available(self) -> bool:
        """Checks if real fine-tuned adapter weights exist on disk."""
        return self.adapter_weights_file.exists()

    def get_system_health(self) -> Dict[str, Any]:
        """Queries host memory, GPU availability, and model readiness without loading weights."""
        mem_info = check_host_memory_headroom()
        cuda_available = False
        gpu_name = None
        vram_total_gb = 0.0

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        except ImportError:
            pass

        adapter_exists = self.is_adapter_available()

        return {
            "status": "ready" if self.is_loaded else "standby",
            "model_id": self.model_id,
            "revision": self.revision,
            "model_loaded": self.is_loaded,
            "adapter_exists": adapter_exists,
            "adapter_loaded": self.adapter_loaded,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "gpu_vram_gb": vram_total_gb,
            "host_ram_total_gb": mem_info["total_ram_gb"],
            "host_ram_avail_gb": mem_info["avail_ram_gb"],
            "active_configuration": (
                "lora_slm" if self.adapter_loaded
                else ("base_slm" if self.is_loaded else "development_fallback")
            ),
            "adapter_status": "ready_for_loading" if adapter_exists else "pending_colab_training"
        }

    def load_model(self, force_cpu: bool = False) -> Dict[str, Any]:
        """
        Explicitly triggers model loading.
        Checks available RAM/VRAM to prevent host freezes.
        """
        if self.is_loaded and self.model is not None:
            return {"status": "already_loaded", "device": self.device}

        mem_info = check_host_memory_headroom()
        if mem_info["avail_ram_gb"] < 1.5 and not force_cpu:
            raise MemoryError(
                f"Insufficient memory: Host has only {mem_info['avail_ram_gb']} GB available. "
                "Loading 1.5B weights requires at least 2.5 GB RAM headroom."
            )

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        cache_dir = resolve_cache_directory(
            self.config.get("hardware", {}).get("cache_management", {}).get("hf_home")
        )

        target_device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        dtype = torch.float16 if target_device == "cuda" else torch.float32

        print(f"Loading tokenizer '{self.model_id}' (revision={self.revision})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=False,
            cache_dir=cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading weights for '{self.model_id}' on {target_device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            cache_dir=cache_dir
        )

        if target_device == "cuda":
            self.model = self.model.to("cuda")

        # Attempt adapter attach if present
        if self.is_adapter_available():
            try:
                from peft import PeftModel
                print(f"Mounting LoRA adapter from {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
                self.adapter_loaded = True
            except Exception as e:
                print(f"Warning: Could not attach adapter: {e}")

        self.model.eval()
        self.is_loaded = True
        self.device = target_device

        return {
            "status": "loaded_successfully",
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "adapter_loaded": self.adapter_loaded
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """Generates text from prompt using loaded model."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Load model or enable development fallback.")

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
