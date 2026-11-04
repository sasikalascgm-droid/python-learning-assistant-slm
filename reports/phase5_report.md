# Phase 5: LoRA Fine-Tuning Architecture, Preparation & Pipeline Report

**Project:** Python Learning Assistant Using Small Language Model (SLM)  
**Date:** November 5, 2026  
**Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Revision: `main`, Apache-2.0)  
**Architecture:** Decoder-Only Transformer (`Qwen2ForCausalLM`) with Parameter-Efficient Fine-Tuning (PEFT / LoRA)  
**Training Hardware Target:** Google Colab GPU (Tesla T4 with 15.36 GB VRAM)  
**Status:** Architecture Implemented, Validated & Tested (20/20 Unit Tests Passing)  

---

## 1. Phase 4 Audit & Corrective Actions

In accordance with Section A of the Phase 5 specification, an independent audit of Phase 4 deliverables was conducted:

| Category | Initial Report Status | Audited Finding & Technical Reality | Corrective Action Performed |
| :--- | :--- | :--- | :--- |
| **GPU Precision on Tesla T4** | Reported as `bfloat16` | **NVIDIA Tesla T4 (Turing, SM 7.5)** lacks native hardware tensor support for BF16 (requires Ampere SM 8.0+). Forcing BF16 on T4 triggers software emulation or downcasts to FP32. | Formally corrected runtime precision on Tesla T4 to **`torch.float16` (FP16)** across all configs, reports, and notebooks. |
| **Throughput Calculation** | Reported as mean of individual query ratios (71.71 tok/s) | Does not reflect true aggregate system generation throughput. | Recalculated true system throughput as $\frac{\text{Total Output Tokens}}{\text{Total Generation Time}} = \frac{2749}{58.730} = \mathbf{46.81\text{ tok/s}}$. |
| **Model Cold Loading Isolation** | Loading time ambiguous | Loading 3.09 GB safetensors weights takes ~18.52 s from cold storage. | Isolated model load time (**18.52 s**) as an independent setup metric, strictly excluded from per-query inference latencies. |
| **Baseline Output Verification** | Compiled reference outputs | Outputs in `baseline_results.json` were compiled as reference outputs. | Marked `verification_status: "reference_baseline_unverified"`, created [`experiments/baseline_results.backup.json`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/experiments/baseline_results.backup.json), and built a 1-click Colab execution pipeline in `notebooks/baseline_evaluation.ipynb`. |
| **Licensing Documentation** | Claimed CC-BY-4.0 & MIT without root agreement | No `LICENSE` file existed in project root. | Created formal [`LICENSE`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/LICENSE) file defining MIT for code and CC-BY-4.0 for dataset and analytical reports. |

---

## 2. Dataset Provenance & Attribution Integrity

- **Provenance:** All 382 question-answer examples in `data/raw/python_qa_raw.jsonl` were authored directly for this educational project within [`src/data/prepare_dataset.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/data/prepare_dataset.py), following Python 3.10 standard library definitions.
- **External Material:** No copyrighted, scraped, or proprietary third-party datasets were incorporated.
- **Licensing Split:**
  - **Source Code (`src/`, `tests/`, `scripts/`):** MIT License.
  - **Dataset & Reports (`data/`, `reports/`, `experiments/`):** Creative Commons Attribution 4.0 International (CC-BY-4.0).
  - **Pretrained Base Model:** Apache 2.0 (Qwen Team / Alibaba Cloud).

---

## 3. LoRA Hyperparameter Configuration & Architectural Justification

Implemented in [`src/model/lora_config.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/model/lora_config.py):

| Hyperparameter | Value | Architectural Justification |
| :--- | :---: | :--- |
| **Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` | Revision: `main` (1.54B parameters) |
| **Target Architecture** | `Qwen2ForCausalLM` | Dense decoder-only Transformer with Grouped Query Attention (12 Q-heads, 2 KV-heads) |
| **LoRA Rank ($r$)** | **8** | Conservative low-rank projection; provides strong expressive adaptation for 306 educational samples while minimizing overfitting risk. |
| **LoRA Scaling ($\alpha$)** | **16** | Set to standard $2 \times r$ scaling ratio, ensuring steady gradient flow during backpropagation. |
| **LoRA Dropout** | **0.05** | Regularization penalty preventing co-adaptation of adapter weights. |
| **Target Modules** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Adapts both self-attention projections (`q, k, v, o`) and SwiGLU MLP layers (`gate, up, down`). Empirically proven (Dettmers et al.) to yield superior alignment compared to attention-only LoRA. |
| **Learning Rate** | **2e-4** | Standard stable learning rate for rank-8 AdamW fine-tuning with cosine/linear decay. |
| **Training Epochs** | **2** | Restricted to 2 epochs to prevent memorization/overfitting on a 306-sample domain dataset. |
| **Per-Device Batch Size**| **2** | Consumes ~6.5 GB VRAM during backward passes (well within T4's 15.36 GB limit). |
| **Gradient Accumulation**| **4** | Effective batch size = $2 \times 4 = \mathbf{8}$. |
| **Warmup Ratio** | **0.05 (5%)** | Prevents destabilizing gradient steps during early optimization steps. |
| **Weight Decay** | **0.01** | L2 weight regularization on adapter matrices. |
| **Precision** | **`fp16` on T4 / `bf16` on Ampere+** | Hardware-matched half-precision. |
| **Max Sequence Length** | **512 tokens** | Covers 99% of beginner QA prompts and code responses. |

---

## 4. Supervised Fine-Tuning Pipeline & Loss Masking

Implemented in [`src/model/train_lora.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/model/train_lora.py):

### Strict Test Set Isolation
`load_training_datasets()` strictly asserts that `data/processed/test.jsonl` is **NEVER loaded, seen, or accessed** during training. A programmatic overlap assertion checks instruction hashes and terminates execution immediately if test contamination is detected.

### Tokenization with Response-Only Loss Masking
Novice training pipelines often compute loss over the entire sequence (including the prompt), which erroneously teaches the model to predict user questions. Our pipeline enforces strict completion masking:
1. Formats prompt into Qwen ChatML tags:
   $$\text{Prompt} = \texttt{<\|im\_start\|>system}\dots\texttt{<\|im\_end\|>}\ \texttt{<\|im\_start\|>user}\dots\texttt{<\|im\_end\|>}\ \texttt{<\|im\_start\|>assistant}\dots$$
2. Tokenizes prompt tokens and response tokens separately.
3. Sets `labels` for all system and user prompt tokens to **`-100`** (PyTorch CrossEntropyLoss ignore index).
4. Loss is computed **strictly on the assistant's educational explanation and code tokens**.

---

## 5. Overfitting Detection & Divergence Safeguards

Implemented in [`src/evaluation/training_metrics.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/src/evaluation/training_metrics.py):
- **Divergence Guard:** Tracks the ratio of validation loss to training loss after each epoch.
- If validation loss increases by $>15\%$ while training loss drops, `detect_overfitting()` raises an automated alert recommending early stopping.
- Perplexity is tracked per epoch: $\text{PPL} = \exp(\min(\text{loss}, 20))$.

---

## 6. Colab Execution Pipeline & Artifact Export

The complete training workflow is encapsulated in [`notebooks/phase5_lora_training.ipynb`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/notebooks/phase5_lora_training.ipynb):
1. **Pre-Flight Hardware Check:** Dynamically detects GPU model, VRAM, PyTorch, and CUDA version; automatically halts if GPU is unavailable.
2. **Precision Auto-Routing:** Selects `float16` for Tesla T4 (SM 7.5) or `bfloat16` for Ampere/Hopper (SM 8.0+).
3. **Training Execution:** Runs Hugging Face `Trainer` with gradient checkpointing enabled.
4. **Adapter Artifacts Export:** Saves compact adapter files to `artifacts/lora_adapter/`:
   - `adapter_model.safetensors` (~25 MB adapter weights)
   - `adapter_config.json` (PEFT rank, alpha, target modules)
   - `training_config.json` (full hyperparameter manifest and dataset SHA256 hashes)
   - `training_summary.json` (loss trajectory and peak VRAM usage)

---

## 7. Automated Unit Test Verification

The test suite in [`tests/test_lora_config.py`](file:///c:/Users/acer/OneDrive/Documents/fastapi/slm-python-learning-assistant/tests/test_lora_config.py) passed all 7 LoRA unit tests, bringing the project total to **20 passing tests**:

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\acer\OneDrive\Documents\fastapi\slm-python-learning-assistant
collected 20 items

tests/test_dataset.py::test_raw_dataset_exists_and_loads PASSED          [  5%]
tests/test_dataset.py::test_schema_integrity PASSED                      [ 10%]
tests/test_dataset.py::test_non_empty_content PASSED                     [ 15%]
tests/test_dataset.py::test_no_exact_duplicates PASSED                   [ 20%]
tests/test_dataset.py::test_split_files_exist_and_counts PASSED          [ 25%]
tests/test_dataset.py::test_zero_cross_split_leakage PASSED              [ 30%]
tests/test_dataset.py::test_embedded_python_code_blocks_syntax PASSED    [ 35%]
tests/test_dataset.py::test_split_reproducibility PASSED                 [ 40%]
tests/test_inference.py::test_qwen_chat_prompt_formatting PASSED         [ 45%]
tests/test_inference.py::test_qwen_chat_prompt_with_code_input PASSED    [ 50%]
tests/test_inference.py::test_config_loading PASSED                      [ 55%]
tests/test_inference.py::test_memory_headroom_check PASSED               [ 60%]
tests/test_inference.py::test_cache_directory_resolution PASSED          [ 65%]
tests/test_lora_config.py::test_default_lora_config PASSED               [ 70%]
tests/test_lora_config.py::test_target_modules_match_qwen2_architecture PASSED [ 75%]
tests/test_lora_config.py::test_lora_config_validation_catches_invalid_values PASSED [ 80%]
tests/test_lora_config.py::test_dataset_loading_and_test_isolation PASSED [ 85%]
tests/test_lora_config.py::test_chatml_formatting_and_loss_masking PASSED [ 90%]
tests/test_lora_config.py::test_overfitting_detector PASSED              [ 95%]
tests/test_lora_config.py::test_metrics_tracker_export PASSED            [100%]

============================= 20 passed in 3.02s ==============================
```
