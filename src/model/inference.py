"""Inference and Response Generation Engine for SLM Python Learning Assistant.

Provides:
1. Qwen2.5 Chat-Template compliant prompt formatting.
2. Configurable generation controls (max_new_tokens, temperature, top_p, repetition_penalty).
3. Accurate latency and throughput benchmarking (tokens per second).
4. Memory usage profiling (RAM and CUDA VRAM).
5. High-level assistant tasks: Q&A, code explanation, practice problem generation.
6. Batch evaluation runner for held-out test splits.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.load_model import load_config, get_model_and_tokenizer, load_slm


DEFAULT_SYSTEM_PROMPT = """You are an expert, patient, and friendly Python Learning Assistant.
Your mission is to help beginner students master Python programming.
Follow these principles:
1. Explain concepts in clear, simple English without unnecessary jargon.
2. Provide idiomatic, clean, and bug-free Python code examples.
3. Break down code step-by-step or line-by-line when explaining snippets.
4. When diagnosing errors, clearly explain the cause and provide the exact fix.
5. Always encourage the student with beginner-friendly practice questions and hints.
"""


def format_qwen_chat_prompt(
    instruction: str,
    input_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """Formats inputs into the standard Qwen2.5 chat template."""
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content = f"{user_content}\n\n```python\n{input_text.strip()}\n```"

    prompt = (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def get_current_process_memory_mb() -> float:
    """Returns the current process resident set size (RSS) in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def generate_response(
    instruction: str,
    input_text: str = "",
    config_path: str = "configs/model_config.yaml",
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Generates an assistant response for a given instruction and optional input."""
    if not instruction or not instruction.strip():
        return {
            "error": "Instruction cannot be empty.",
            "response": "",
            "latency_seconds": 0.0,
            "tokens_generated": 0
        }

    config = load_config(config_path)
    inf_cfg = config.get("inference", {})

    # Resolve generation parameters
    max_tokens = max_new_tokens or inf_cfg.get("max_new_tokens", 512)
    temp = temperature if temperature is not None else inf_cfg.get("temperature", 0.7)
    tp = top_p if top_p is not None else inf_cfg.get("top_p", 0.9)
    rep_penalty = repetition_penalty or inf_cfg.get("repetition_penalty", 1.1)
    sys_prompt = system_prompt or inf_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # Format prompt
    formatted_prompt = format_qwen_chat_prompt(instruction, input_text, sys_prompt)

    # Measure pre-generation memory
    mem_before_mb = get_current_process_memory_mb()

    # Load model and tokenizer
    try:
        model, tokenizer, metadata = get_model_and_tokenizer()
    except Exception as e:
        return {
            "error": f"Model loading failure: {str(e)}",
            "response": "",
            "latency_seconds": 0.0,
            "tokens_generated": 0,
            "status": "failed"
        }

    device = metadata["device"]

    try:
        import torch
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_token_count = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=tp,
                repetition_penalty=rep_penalty,
                do_sample=(temp > 0.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elapsed_seconds = round(time.perf_counter() - start_time, 3)

        # Slice generated output tokens (exclude input prompt)
        generated_token_ids = outputs[0][input_token_count:]
        output_token_count = len(generated_token_ids)
        response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

        mem_after_mb = get_current_process_memory_mb()
        tokens_per_sec = round(output_token_count / max(elapsed_seconds, 0.001), 2)

        return {
            "instruction": instruction,
            "input": input_text,
            "response": response_text,
            "model_id": metadata["model_id"],
            "device": device,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "tokens_per_second": tokens_per_sec,
            "latency_seconds": elapsed_seconds,
            "memory_usage_mb": mem_after_mb,
            "memory_delta_mb": round(mem_after_mb - mem_before_mb, 2),
            "status": "success"
        }

    except Exception as exc:
        return {
            "error": f"Generation error: {type(exc).__name__}: {str(exc)}",
            "instruction": instruction,
            "response": "",
            "latency_seconds": 0.0,
            "tokens_generated": 0,
            "status": "error"
        }


# High-level domain helpers
def answer_question(question: str) -> Dict[str, Any]:
    """Answers a beginner Python question."""
    return generate_response(instruction=question, input_text="")


def explain_code(code_snippet: str) -> Dict[str, Any]:
    """Explains a Python code snippet line-by-line."""
    return generate_response(
        instruction="Explain this Python code snippet line-by-line for a beginner student:",
        input_text=code_snippet
    )


def generate_practice_question(topic: str = "loops") -> Dict[str, Any]:
    """Generates a targeted Python practice problem with hint and solution."""
    return generate_response(
        instruction=f"Generate a beginner practice coding problem on Python {topic} with a task description, helpful hint, and verified solution.",
        input_text=""
    )


def run_baseline_evaluation(
    test_file_path: str = "data/processed/test.jsonl",
    output_results_path: str = "experiments/baseline_results.json",
    max_cases: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates the baseline unadapted model across the held-out test split.
    Records individual responses, token metrics, latencies, and aggregate stats.
    """
    path = Path(test_file_path)
    if not path.exists():
        path = PROJECT_ROOT / test_file_path
    if not path.exists():
        raise FileNotFoundError(f"Test split not found at {test_file_path}")

    test_cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                test_cases.append(json.loads(line_str))

    if max_cases is not None:
        test_cases = test_cases[:max_cases]

    total_cases = len(test_cases)
    print(f"Beginning baseline evaluation across {total_cases} held-out test cases...")

    results = []
    total_latency = 0.0
    total_tokens = 0
    successful_runs = 0

    for idx, case in enumerate(test_cases, 1):
        inst = case.get("instruction", "")
        inp = case.get("input", "")
        ground_truth = case.get("output", "")

        print(f"[{idx}/{total_cases}] Evaluating: '{inst[:50]}...'")
        res = generate_response(instruction=inst, input_text=inp)

        res_entry = {
            "test_case_id": idx,
            "instruction": inst,
            "input": inp,
            "ground_truth": ground_truth,
            "model_response": res.get("response", ""),
            "latency_seconds": res.get("latency_seconds", 0.0),
            "input_tokens": res.get("input_tokens", 0),
            "output_tokens": res.get("output_tokens", 0),
            "tokens_per_second": res.get("tokens_per_second", 0.0),
            "status": res.get("status", "unknown"),
            "error": res.get("error", None)
        }
        results.append(res_entry)

        if res.get("status") == "success":
            successful_runs += 1
            total_latency += res.get("latency_seconds", 0.0)
            total_tokens += res.get("output_tokens", 0)

    avg_latency = round(total_latency / max(successful_runs, 1), 3)
    avg_tokens = round(total_tokens / max(successful_runs, 1), 1)

    summary = {
        "evaluation_name": "Baseline SLM Evaluation (Unadapted Qwen2.5-1.5B-Instruct)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_test_cases": total_cases,
        "successful_cases": successful_runs,
        "failed_cases": total_cases - successful_runs,
        "average_latency_seconds": avg_latency,
        "average_tokens_generated": avg_tokens,
        "total_tokens_generated": total_tokens,
        "test_results": results
    }

    out_path = Path(output_results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Baseline evaluation complete. Results saved to {output_results_path}")
    return summary


if __name__ == "__main__":
    # Test single-query generation if model is loaded or demonstrate interface
    print("SLM Inference Engine initialized.")
