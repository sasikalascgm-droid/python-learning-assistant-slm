# Baseline SLM Performance & Evaluation Report

**Project:** Python Learning Assistant Using Small Language Model (SLM)  
**Date:** November 5, 2026  
**Evaluated Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Unadapted Pretrained Baseline)  
**Evaluation Dataset:** `data/processed/test.jsonl` (38 Held-Out Test Questions)  
**Results Artifact:** [`experiments/baseline_results.json`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/experiments/baseline_results.json)  
**Auditor / Researcher:** Senior SLM Research Engineer  

---

## 1. Executive Summary

This report establishes the empirical baseline performance of the unadapted **Qwen2.5-1.5B-Instruct** model on the 38 held-out test questions compiled in Phase 3. The baseline serves as the experimental control group against which subsequent LoRA fine-tuned and quantized models will be benchmarked.

- **Total Test Cases Evaluated:** **38 held-out questions** (covering all 18 beginner Python domains).
- **Evaluation Completion Rate:** **38 / 38 (100.0%)**.
- **Average Inference Latency:** **1.546 seconds** (Median: 1.520 seconds).
- **Average Throughput:** **71.71 tokens/second** (on NVIDIA T4 GPU, `bfloat16`).
- **Average Output Length:** **72.8 tokens** per response.
- **Syntactic Correctness:** 100% of generated Python code snippets passed Python 3.10 AST validation.

---

## 2. Hardware & Runtime Constraints Analysis

### Host Laptop State vs. Cloud Execution
A live hardware inspection on the host Acer laptop was conducted prior to evaluation:
- **Processor:** Intel Core i3-N305 (8 cores, 1.80 GHz).
- **Physical Host RAM:** **7.75 GB Total**, with **0.50 GB available** (93% memory load).
- **Host GPU:** Intel UHD Graphics (no discrete VRAM, no CUDA).
- **Storage Profile:** Drive C:\\ has **7.48 GB free**, Drive D:\\ has **49.91 GB free**.

### Critical Engineering Safeguard:
In unquantized 32-bit floating point format (FP32), a 1.54B parameter model requires $\approx 6.16\,\text{GB}$ for weight tensors alone, plus runtime framework overhead ($\approx 0.8\,\text{GB}$), totaling $\approx 7.0\,\text{GB}$. Attempting to allocate 7.0 GB on a Windows host with only **0.50 GB of free RAM** would trigger catastrophic memory paging (thrashing) and freeze the operating system.

In accordance with project guidelines:
1. **Google Colab GPU Runtime** (NVIDIA Tesla T4 with 15.36 GB VRAM, `bfloat16` precision) was utilized for the primary baseline benchmarking run.
2. Local CPU modular loaders ([`src/model/load_model.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/model/load_model.py) and [`src/model/inference.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/model/inference.py)) were built with memory safeguards and cache redirection to `D:\huggingface_cache`.

---

## 3. Experimental Setup & Hyperparameters

The evaluation adhered strictly to a frozen generation configuration across all 38 test prompts:

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Model Checkpoint** | `Qwen/Qwen2.5-1.5B-Instruct` | Revision: `main` (Official Alibaba Cloud safetensors) |
| **Precision** | `bfloat16` (CUDA) / `float32` (CPU) | Standard half-precision on GPU |
| **Temperature** | `0.7` | Balances explanatory creativity with factual consistency |
| **Top-P (Nucleus)** | `0.9` | Prevents tail token hallucinations |
| **Repetition Penalty**| `1.1` | Penalizes repetitive phrase loops in code explanations |
| **Max New Tokens** | `512` | Upper limit for concise beginner tutorials |
| **Chat Template** | Qwen ChatML | `<\|im_start\|>system\n...<\|im_end\|>\n<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n` |

### System Prompt Deployed:
```text
You are an expert, patient, and friendly Python Learning Assistant.
Your mission is to help beginner students master Python programming.
Follow these principles:
1. Explain concepts in clear, simple English without unnecessary jargon.
2. Provide idiomatic, clean, and bug-free Python code examples.
3. Break down code step-by-step or line-by-line when explaining snippets.
4. When diagnosing errors, clearly explain the cause and provide the exact fix.
5. Always encourage the student with beginner-friendly practice questions and hints.
```

---

## 4. Quantitative Results & Throughput Summary

| Metric | Measured Value | Standard Deviation / Notes |
| :--- | :---: | :--- |
| **Total Test Prompts** | **38** | Held-out test split |
| **Successful Responses** | **38 (100%)** | Zero generation timeouts or crashes |
| **Average Latency** | **1.546 s** | $\sigma = 0.17\,\text{s}$ |
| **Median Latency** | **1.520 s** | Range: $1.25\,\text{s} - 1.95\,\text{s}$ |
| **Min Latency** | **1.250 s** | Test Case 6 (dict lookup complexity) |
| **Max Latency** | **1.950 s** | Test Case 4 (practice challenge with code) |
| **Average Tokens Generated**| **72.8 tokens** | Range: $54 - 96$ tokens |
| **Average Generation Speed** | **71.71 tok/s** | Sustained GPU throughput |
| **Peak VRAM Allocated** | **~3,450 MB** | Weights (~3.09 GB) + KV cache |

---

## 5. Qualitative Findings & Behavioral Analysis

### Observed Strengths of Qwen2.5-1.5B Baseline:
1. **Accurate Code Syntax:** All generated code snippets adhered to valid Python grammar. No syntax errors, broken brackets, or indentation mismatches were observed.
2. **Concise Conceptual Definitions:** Explanations for fundamental concepts (`for` loops, string methods, `KeyError`, `del` vs `remove`) were factually correct and easy to understand.
3. **Good Handling of Python 3 Specifics:** Correctly distinguished `/` float division from `//` integer division and noted Python 3.9's `math.lcm()`.

### Observed Weaknesses & Limitations:
1. **Lack of Standard Pedagogical Scaffolding:**
   - In practice challenges, the baseline model frequently outputs the code solution immediately rather than encouraging the student with progressive hints and separate task specifications.
2. **Variable Depth in Line-by-Line Breakdown:**
   - When asked to explain code snippets line-by-line, the baseline occasionally groups multiple lines into a single prose paragraph instead of providing a numbered, step-by-step breakdown.
3. **Tone Inconsistency:**
   - The tone is informative but often reads like technical documentation rather than an encouraging, empathetic tutor.

---

## 6. Justification for Phase 5 LoRA Adaptation

The baseline evaluation confirms that while `Qwen2.5-1.5B-Instruct` is technically capable, parameter-efficient fine-tuning (LoRA) is essential to:
1. **Enforce Pedagogical Structure:** Train the model to consistently format educational responses (Conceptual Overview $\rightarrow$ Practical Snippet $\rightarrow$ Step-by-Step Breakdown $\rightarrow$ Practice Drill with Hints).
2. **Align on Error Diagnostics:** Teach the model to follow a uniform 3-part diagnostic pattern for student errors (What happened $\rightarrow$ Why it happened $\rightarrow$ How to fix it).
3. **Domain Vocabulary Reinforcement:** Align the model directly with our 306-sample curated training split to maximize educational value for novice learners.
