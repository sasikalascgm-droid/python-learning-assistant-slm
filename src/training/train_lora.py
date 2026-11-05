"""Compatibility alias for src.model.train_lora."""

from src.model.train_lora import *

if __name__ == "__main__":
    import sys
    sys.exit(run_lora_training())
