# Phase 5 Audit & Corrections Report

**Project:** Python Learning Assistant Using Small Language Model (SLM)  
**Date:** November 5, 2026  
**Status:** Audited, Corrected, and Verified (34/34 Unit Tests Passing)  

---

## 1. Executive Summary of Corrections

Following an independent, line-by-line audit of Phase 5 deliverables, multiple critical technical discrepancies were discovered and corrected across codebase files, configurations, notebooks, and reports:

| Category | Prior Phase 5 Status | Audited Finding | Corrective Action Implemented | Impacted Files |
| :--- | :--- | :--- | :--- | :--- |
| **LoRA Trainable Parameters** | Reported as "~4.5 million parameters (~0.29%)" | Heuristic underestimated parameter count by failing to account for SwiGLU MLP intermediate dimension (8960 vs 1536). Real architecture has **9,232,384** trainable parameters (**0.5981%**). | Derived exact projection equations; implemented `calculate_lora_parameters()`; added unit tests. | `src/model/lora_config.py`, `src/model/train_lora.py`, `tests/test_lora_config.py` |
| **Base Model Revision Pinning** | `main` unpinned branch | Risk of upstream weight mutation or breaking changes. Pinned immutable commit hash required. | Pinned model to commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` across all loaders and configs. | `configs/model_config.yaml`, `src/model/load_model.py`, `src/model/lora_config.py`, `notebooks/phase5_lora_training.ipynb` |
| **Config Hyperparameter Alignment** | `configs/model_config.yaml` had $r=16, \alpha=32$, 3 epochs, `bf16=true` | Inconsistent with validated `SLMLoRAConfig` ($r=8, \alpha=16$, 2 epochs, `fp16=true`). Turing T4 lacks hardware BF16. | Fully aligned YAML config with validated LoRA architecture and Tesla T4 precision constraints. | `configs/model_config.yaml` |
| **Loss Masking & Padding** | Static masking without padding exclusion | Needed explicit padding label masking (`-100`) and attention mask zeroing (`0`) for robust batching. | Enhanced `tokenize_with_prompt_masking()` to support dynamic padding and explicit `-100` label masking. | `src/model/train_lora.py`, `tests/test_lora_config.py` |
| **Licensing Attribution Scope** | Brief mention of Apache-2.0 | Ambiguity regarding whether project claimed copyright on base model weights. | Added explicit exclusion in `LICENSE` clarifying base model weights belong to Alibaba Cloud / Qwen Team under Apache 2.0. | `LICENSE` |
| **Baseline Provenance** | Reference data in `baseline_results.json` | 38 baseline records are unverified reference samples. | Preserved original backup, maintained `verification_status: "reference_baseline_unverified"`. | `experiments/baseline_results.json`, `experiments/baseline_results.backup.json` |

---

## 2. In-Depth LoRA Parameter Calculation & Mathematical Proof

### The Origin of the Discrepancy
The previous Phase 5 report quoted:
> *"Trainable Parameters: ~4.5 Million parameters (~0.29% of base model weights)."*

This heuristic assumed that each linear projection had dimensions roughly equal to $d_{\text{model}} \times d_{\text{model}} = 1536 \times 1536$. However, in modern decoder-only LLMs using SwiGLU activations (such as `Qwen2ForCausalLM`), the MLP intermediate projections (`gate_proj`, `up_proj`, `down_proj`) have an intermediate dimension of **8960**, which is almost 6 times wider than the hidden dimension!

### Exact Mathematical Derivation
For an un-biased linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, standard LoRA introduces two low-rank matrices:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$

The number of trainable parameters per projection is:
$$\text{Params} = r \times d_{\text{in}} + d_{\text{out}} \times r = r \times (d_{\text{in}} + d_{\text{out}})$$

For `Qwen/Qwen2.5-1.5B-Instruct` ($d_{\text{model}} = 1536$, $d_{\text{mlp}} = 8960$, $n_{\text{heads}} = 12$, $n_{\text{kv\_heads}} = 2$, $d_{\text{head}} = 128$, $r = 8$):

| Projection Module | Input Dimension ($d_{\text{in}}$) | Output Dimension ($d_{\text{out}}$) | LoRA $A$ ($8 \times d_{\text{in}}$) | LoRA $B$ ($d_{\text{out}} \times 8$) | Parameters per Layer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `q_proj` (Query) | 1,536 | 1,536 | 12,288 | 12,288 | **24,576** |
| `k_proj` (Key - GQA) | 1,536 | 256 | 12,288 | 2,048 | **14,336** |
| `v_proj` (Value - GQA) | 1,536 | 256 | 12,288 | 2,048 | **14,336** |
| `o_proj` (Output) | 1,536 | 1,536 | 12,288 | 12,288 | **24,576** |
| `gate_proj` (SwiGLU Gate) | 1,536 | 8,960 | 12,288 | 71,680 | **83,968** |
| `up_proj` (SwiGLU Up) | 1,536 | 8,960 | 12,288 | 71,680 | **83,968** |
| `down_proj` (SwiGLU Down)| 8,960 | 1,536 | 71,680 | 12,288 | **83,968** |
| **Total per Transformer Block** | — | — | — | — | **329,728** |

Across all **28 transformer layers**:
$$\text{Total Trainable LoRA Parameters} = 28 \times 329,728 = \mathbf{9,232,384}$$

Base Model Parameter Total:
$$\text{Total Base Parameters} = \mathbf{1,543,714,816}$$

Trainable Percentage:
$$\text{Trainable Percentage} = \frac{9,232,384}{1,543,714,816} \times 100\% = \mathbf{0.5981\%}$$

This calculation is now implemented dynamically in `calculate_lora_parameters()` in [`src/model/lora_config.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/model/lora_config.py) and verified by unit test `test_exact_lora_parameter_math()` in [`tests/test_lora_config.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/tests/test_lora_config.py).

---

## 3. Dataset Integrity & Training Loss Masking Safeguards

1. **Strict Test Set Isolation Assertion:**
   - Programmatically validated that `data/processed/test.jsonl` contains zero overlap with `data/processed/train.jsonl` or `validation.jsonl`.
   - Added unit test `test_test_set_leak_assertion_triggers()` verifying that an `AssertionError` is raised if test prompts contaminate training inputs.
2. **Response-Only Loss Masking:**
   - ChatML prompt formatting:
     `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
   - Prompt tokens receive label `-100` (PyTorch CrossEntropyLoss ignore index).
   - Only the assistant response tokens (`output<|im_end|>\n`) contribute to the loss gradient.
   - Padded tail elements receive `attention_mask = 0` and `labels = -100`.
   - Verified by unit tests `test_chatml_formatting_and_loss_masking()` and `test_padding_loss_masking_exclusion()`.

---

## 4. Hardware Realism & License Scope Clarification

1. **Laptop Constraints Acknowledged:**
   - Host: Intel Core i3-N305, 7.75 GB RAM (~0.5–0.8 GB free), Intel UHD Graphics (no CUDA GPU).
   - Local training remains strictly prohibited to protect host system stability.
2. **Pinned Base Model Revision:**
   - Base model: `Qwen/Qwen2.5-1.5B-Instruct` pinned to commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
3. **Licensing Framework:**
   - Base Model Weights: **Apache-2.0 License** (Alibaba Cloud / Qwen Team).
   - Repository Source Code: **MIT License**.
   - Curated Dataset & Technical Reports: **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
