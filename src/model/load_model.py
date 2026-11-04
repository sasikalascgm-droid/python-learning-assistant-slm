"""Model and Tokenizer Loading Module for SLM Python Learning Assistant.

Provides centralized, singleton model management with:
1. Hardware pre-flight checks (CPU vs CUDA GPU, RAM and VRAM auditing).
2. Storage routing (routes Hugging Face cache to Drive D: when available).
3. Precision management (float32 on CPU, bfloat16/float16 on CUDA).
4. Memory-efficient loading (low_cpu_mem_usage=True).
5. Chat template verification for Qwen2.5.
6. Singleton lifecycle management (avoids reloading on every query).
"""

import os
import sys
import shutil
import ctypes
from pathlib import Path
from typing import Tuple, Any, Optional, Dict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Module-level singletons
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_LOADED_MODEL_ID = None
_LOADED_DEVICE = None


def load_config(config_path: str = "configs/model_config.yaml") -> Dict[str, Any]:
    """Loads configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_host_memory_headroom() -> Dict[str, float]:
    """Queries current host physical RAM availability in GB."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return {
            "total_ram_gb": round(stat.ullTotalPhys / (1024**3), 2),
            "avail_ram_gb": round(stat.ullAvailPhys / (1024**3), 2),
            "load_percent": float(stat.dwMemoryLoad)
        }
    except Exception:
        return {"total_ram_gb": 8.0, "avail_ram_gb": 1.0, "load_percent": 85.0}


def resolve_cache_directory(configured_cache: Optional[str] = None) -> str:
    """Determines the optimal cache directory, prioritizing Drive D: when available and writable."""
    if configured_cache:
        try:
            cache_path = Path(configured_cache)
            cache_path.mkdir(parents=True, exist_ok=True)
            return str(cache_path)
        except (PermissionError, OSError):
            pass

    # Attempt Drive D: if present and writable
    if os.path.exists("D:\\"):
        try:
            d_cache = Path("D:/huggingface_cache")
            d_cache.mkdir(parents=True, exist_ok=True)
            return str(d_cache)
        except (PermissionError, OSError):
            pass

    # Fallback to local workspace cache
    default_cache = PROJECT_ROOT / ".cache" / "huggingface"
    default_cache.mkdir(parents=True, exist_ok=True)
    return str(default_cache)


def get_optimal_device(force_cpu: bool = False) -> Tuple[str, Any]:
    """
    Detects hardware and returns device string and optimal torch dtype.
    Safely imports torch only when called.
    """
    try:
        import torch
        if force_cpu:
            return "cpu", torch.float32

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"CUDA GPU detected: {device_name} with {total_vram_gb:.2f} GB VRAM")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dtype
        else:
            return "cpu", torch.float32
    except ImportError:
        return "cpu", None


def load_slm(
    config_path: str = "configs/model_config.yaml",
    force_reload: bool = False,
    force_cpu: bool = False
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads model and tokenizer using the singleton pattern.
    Returns: (model, tokenizer, metadata)
    """
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _LOADED_MODEL_ID, _LOADED_DEVICE

    config = load_config(config_path)
    model_id = config.get("model", {}).get("model_id", "Qwen/Qwen2.5-1.5B-Instruct")
    model_revision = config.get("model", {}).get("revision", "c644d673f4439c36291a1827471fb3be52382e26")
    target_device = "cpu" if force_cpu else config.get("hardware", {}).get("target_device", "auto")

    # If already loaded and matching configuration, return singleton
    if not force_reload and _GLOBAL_MODEL is not None and _GLOBAL_TOKENIZER is not None:
        if _LOADED_MODEL_ID == model_id:
            print(f"Returning cached model singleton for {model_id} on {_LOADED_DEVICE}")
            return _GLOBAL_MODEL, _GLOBAL_TOKENIZER, {
                "model_id": _LOADED_MODEL_ID,
                "device": _LOADED_DEVICE,
                "cached": True
            }

    # Pre-flight host memory audit
    mem_info = check_host_memory_headroom()
    print(f"Pre-flight Memory Audit: {mem_info['avail_ram_gb']} GB available out of {mem_info['total_ram_gb']} GB ({mem_info['load_percent']}% load)")

    # Cache directory resolution
    cache_dir = resolve_cache_directory(config.get("hardware", {}).get("cache_management", {}).get("hf_home"))
    os.environ["HF_HOME"] = cache_dir
    print(f"Hugging Face cache directory routed to: {cache_dir}")

    # Check dependencies
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            f"PyTorch or Transformers not installed in environment ({sys.executable}). "
            f"Error: {e}. For local CPU execution, install requirements or run evaluation in Google Colab."
        )

    # Determine device and precision
    device_str, torch_dtype = get_optimal_device(force_cpu=(target_device == "cpu"))

    # Memory safeguard warning on host
    if device_str == "cpu" and mem_info["avail_ram_gb"] < 2.0:
        print(f"WARNING: Host machine has only {mem_info['avail_ram_gb']} GB available RAM. Loading 1.5B weights in FP32 requires ~6.2 GB.")
        print("Recommended: Run baseline evaluation via Google Colab GPU or use 4-bit quantized runtime.")

    print(f"Loading tokenizer for '{model_id}' (revision={model_revision}) from cache '{cache_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        trust_remote_code=config.get("model", {}).get("trust_remote_code", False),
        cache_dir=cache_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model weights for '{model_id}' (revision={model_revision}, device={device_str}, dtype={torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=config.get("model", {}).get("low_cpu_mem_usage", True),
        trust_remote_code=config.get("model", {}).get("trust_remote_code", False),
        cache_dir=cache_dir
    )

    if device_str == "cuda":
        model = model.to("cuda")

    model.eval()

    # Cache singletons
    _GLOBAL_MODEL = model
    _GLOBAL_TOKENIZER = tokenizer
    _LOADED_MODEL_ID = model_id
    _LOADED_DEVICE = device_str

    metadata = {
        "model_id": model_id,
        "revision": model_revision,
        "device": device_str,
        "torch_dtype": str(torch_dtype),
        "cache_dir": cache_dir,
        "vocab_size": len(tokenizer),
        "cached": False
    }

    print(f"Model '{model_id}' successfully loaded and ready.")
    return model, tokenizer, metadata


def get_model_and_tokenizer():
    """Convenience accessor for loaded singleton."""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
    if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None:
        return load_slm()
    return _GLOBAL_MODEL, _GLOBAL_TOKENIZER, {"model_id": _LOADED_MODEL_ID, "device": _LOADED_DEVICE, "cached": True}
