"""Post-Adaptation Benchmark and Comprehensive Evaluation Module.

Provides:
1. 1-5 multidimensional educational scoring rubric across 9 quality dimensions.
2. Static AST syntax parser and code block validator (safe, non-executing).
3. Performance metric aggregator (mean/median/P95 latency, true aggregate throughput).
4. Controlled test split evaluation pipeline for held-out test.jsonl (38 records).
5. Transparent execution labeling (MEASURED, ESTIMATED, REFERENCE_UNVERIFIED, NOT RUN, PENDING USER EXECUTION).
6. Export utilities for experiments/phase7_results.json and experiments/phase7_results.csv.
"""

import ast
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.lora_config import PINNED_QWEN_REVISION
from src.model.load_model import load_config


@dataclass
class EvaluationRubric:
    """
    Multidimensional 1-5 pedagogical scoring rubric for Python educational SLM responses.
    Each criterion is evaluated on an integer scale from 1 (poor/failing) to 5 (exemplary).
    """
    factual_correctness: float = 5.0        # Accuracy of Python syntax, semantics, and standard library rules
    relevance: float = 5.0                  # Directly answers the student's question without extraneous tangents
    beginner_friendly: float = 5.0          # Accessible English, gentle tone, clear mental models
    code_correctness: float = 5.0           # Idiomatic Python 3.10 code without syntax errors
    completeness: float = 5.0               # Thorough coverage of core concepts and edge cases
    instruction_following: float = 5.0      # Adherence to requested formatting (e.g. line-by-line, code-only)
    error_explanation_quality: float = 5.0  # Clear diagnosis of exception cause and actionable fix
    line_by_line_quality: float = 5.0       # Clear step-by-step deconstruction of what each line does
    practice_question_quality: float = 5.0  # Pedagogically sound exercise with graduated hints

    def validate(self) -> None:
        """Validates all rubric criteria lie within the 1.0 to 5.0 range."""
        for field_name, val in self.to_dict().items():
            if not (1.0 <= val <= 5.0):
                raise ValueError(f"Rubric score '{field_name}' must be in [1.0, 5.0], got {val}")

    def overall_score(self) -> float:
        """Computes arithmetic mean of all 9 rubric criteria."""
        scores = list(self.to_dict().values())
        return round(sum(scores) / len(scores), 2)

    def to_dict(self) -> Dict[str, float]:
        return {
            "factual_correctness": self.factual_correctness,
            "relevance": self.relevance,
            "beginner_friendly": self.beginner_friendly,
            "code_correctness": self.code_correctness,
            "completeness": self.completeness,
            "instruction_following": self.instruction_following,
            "error_explanation_quality": self.error_explanation_quality,
            "line_by_line_quality": self.line_by_line_quality,
            "practice_question_quality": self.practice_question_quality
        }


class ASTSyntaxChecker:
    """Statically verifies Python code blocks extracted from model outputs."""

    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Extracts python code blocks delimited by triple backticks."""
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
        return [b.strip() for b in blocks if b.strip()]

    @staticmethod
    def validate_code_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Parses code string with ast.parse.
        Safe: pure AST construction; does NOT execute code on host.
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"ParsingError: {e}"

    @classmethod
    def evaluate_response_code(cls, response_text: str) -> Dict[str, Any]:
        """Evaluates all Python code blocks embedded within a response."""
        blocks = cls.extract_code_blocks(response_text)
        if not blocks:
            return {
                "has_code": False,
                "code_block_count": 0,
                "syntax_valid_count": 0,
                "all_syntax_valid": True,
                "runtime_execution": "NOT_APPLICABLE (No code blocks)"
            }

        valid_count = 0
        errors = []
        for idx, block in enumerate(blocks):
            is_valid, err = cls.validate_code_syntax(block)
            if is_valid:
                valid_count += 1
            else:
                errors.append(f"Block {idx+1}: {err}")

        return {
            "has_code": True,
            "code_block_count": len(blocks),
            "syntax_valid_count": valid_count,
            "all_syntax_valid": (valid_count == len(blocks)),
            "syntax_errors": errors,
            "runtime_execution": "NOT_RUN (Host-level execution of untrusted LLM code prohibited without container isolation)"
        }


def compute_throughput(total_tokens: int, total_seconds: float) -> float:
    """Computes true aggregate throughput: total_tokens / total_seconds."""
    if total_seconds <= 0:
        return 0.0
    return round(total_tokens / total_seconds, 2)


def compute_performance_statistics(latencies: List[float], tokens: List[int]) -> Dict[str, Any]:
    """Computes comprehensive latency and throughput distribution metrics."""
    if not latencies or not tokens:
        return {
            "sample_count": 0,
            "latency_mean_seconds": 0.0,
            "latency_median_seconds": 0.0,
            "latency_p95_seconds": 0.0,
            "first_request_latency_seconds": 0.0,
            "warm_latency_mean_seconds": 0.0,
            "total_generated_tokens": 0,
            "aggregate_throughput_tok_per_sec": 0.0
        }

    n = len(latencies)
    sorted_lats = sorted(latencies)
    total_lat = sum(latencies)
    total_tok = sum(tokens)

    median_lat = sorted_lats[n // 2] if n % 2 != 0 else (sorted_lats[n // 2 - 1] + sorted_lats[n // 2]) / 2.0
    p95_idx = int(0.95 * n)
    p95_lat = sorted_lats[min(p95_idx, n - 1)]

    warm_lats = latencies[1:] if n > 1 else latencies
    warm_mean = sum(warm_lats) / len(warm_lats)

    return {
        "sample_count": n,
        "first_request_latency_seconds": round(latencies[0], 3),
        "warm_latency_mean_seconds": round(warm_mean, 3),
        "latency_mean_seconds": round(total_lat / n, 3),
        "latency_median_seconds": round(median_lat, 3),
        "latency_p95_seconds": round(p95_lat, 3),
        "total_generated_tokens": total_tok,
        "aggregate_throughput_tok_per_sec": compute_throughput(total_tok, total_lat)
    }


def load_test_dataset(test_path: str = "data/processed/test.jsonl") -> List[Dict[str, Any]]:
    """Loads held-out test split (strictly 38 records) with validation."""
    path = Path(test_path)
    if not path.exists():
        path = PROJECT_ROOT / test_path
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found at '{test_path}'")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 38:
        print(f"WARNING: Expected 38 test records in {test_path}, found {len(records)}.")

    return records


def export_phase7_results_json(
    results_dict: Dict[str, Any],
    output_path: str = "experiments/phase7_results.json"
) -> Path:
    """Exports structured Phase 7 comparative results to JSON."""
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    return out_file


def export_phase7_results_csv(
    summary_rows: List[Dict[str, Any]],
    output_path: str = "experiments/phase7_results.csv"
) -> Path:
    """Exports high-level comparison matrix to CSV."""
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        return out_file

    fieldnames = list(summary_rows[0].keys())
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    return out_file
