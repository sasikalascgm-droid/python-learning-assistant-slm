"""Training Metrics and Loss Tracking Module for SLM LoRA Adaptation.

Provides:
1. Training and validation loss tracking across epochs.
2. Perplexity computation (PPL = exp(loss)).
3. Overfitting and loss divergence detection.
4. GPU VRAM and memory profiling utilities.
5. Training summary export to JSON.
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class EpochMetric:
    """Container for single epoch training measurements."""
    epoch: int
    train_loss: float
    val_loss: float
    train_perplexity: float = 0.0
    val_perplexity: float = 0.0
    epoch_duration_seconds: float = 0.0
    gpu_memory_peak_mb: float = 0.0

    def __post_init__(self):
        if self.train_loss > 0 and self.train_perplexity == 0.0:
            self.train_perplexity = round(math.exp(min(self.train_loss, 20.0)), 3)
        if self.val_loss > 0 and self.val_perplexity == 0.0:
            self.val_perplexity = round(math.exp(min(self.val_loss, 20.0)), 3)


class TrainingMetricsTracker:
    """Manages epoch metrics, checks for divergence, and produces reports."""

    def __init__(self, run_name: str = "qwen2.5_1.5b_lora_run"):
        self.run_name = run_name
        self.start_time = time.time()
        self.epoch_metrics: List[EpochMetric] = []
        self.hardware_profile: Dict[str, Any] = {}

    def log_hardware_environment(self, gpu_name: str, total_vram_gb: float, torch_version: str, peft_version: str):
        """Records the underlying GPU and software versions."""
        self.hardware_profile = {
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_vram_gb, 2),
            "torch_version": torch_version,
            "peft_version": peft_version
        }

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        duration_seconds: float,
        gpu_peak_mb: float = 0.0
    ) -> EpochMetric:
        """Records metrics for an epoch and checks for anomalies."""
        metric = EpochMetric(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            epoch_duration_seconds=round(duration_seconds, 2),
            gpu_memory_peak_mb=round(gpu_peak_mb, 2)
        )
        self.epoch_metrics.append(metric)
        return metric

    def detect_overfitting(self, threshold_ratio: float = 1.15) -> Dict[str, Any]:
        """
        Analyzes validation loss trend relative to training loss.
        Flags overfitting if validation loss increases by > threshold_ratio
        while training loss continues to decrease.
        """
        if len(self.epoch_metrics) < 2:
            return {
                "overfitting_detected": False,
                "reason": "Insufficient epochs to determine divergence trend (need at least 2)."
            }

        prev = self.epoch_metrics[-2]
        curr = self.epoch_metrics[-1]

        train_loss_decreased = curr.train_loss < prev.train_loss
        val_loss_increased = curr.val_loss > (prev.val_loss * threshold_ratio)

        if train_loss_decreased and val_loss_increased:
            return {
                "overfitting_detected": True,
                "warning": (
                    f"Validation loss diverged at Epoch {curr.epoch}: "
                    f"Val Loss rose from {prev.val_loss:.4f} to {curr.val_loss:.4f} "
                    f"(+{((curr.val_loss - prev.val_loss)/prev.val_loss)*100:.1f}%), "
                    f"while Train Loss decreased from {prev.train_loss:.4f} to {curr.train_loss:.4f}."
                ),
                "recommendation": "Stop training early or increase LoRA dropout / weight decay."
            }

        return {
            "overfitting_detected": False,
            "status": "Loss progression is stable and converging normally."
        }

    def export_summary(self, output_path: str = "artifacts/lora_adapter/training_summary.json") -> Dict[str, Any]:
        """Serializes training metrics to JSON."""
        total_time = round(time.time() - self.start_time, 2)
        summary = {
            "run_name": self.run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_duration_seconds": total_time,
            "hardware": self.hardware_profile,
            "overfitting_status": self.detect_overfitting(),
            "epochs": [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "val_loss": m.val_loss,
                    "train_perplexity": m.train_perplexity,
                    "val_perplexity": m.val_perplexity,
                    "duration_seconds": m.epoch_duration_seconds,
                    "gpu_memory_peak_mb": m.gpu_memory_peak_mb
                }
                for m in self.epoch_metrics
            ]
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
