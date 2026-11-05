"""Compatibility alias for src.model.train_lora."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.train_lora import *

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

