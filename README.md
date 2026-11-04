# Python Learning Assistant Using Small Language Model (SLM)

An end-to-end, locally runnable AI application designed to help beginner students learn Python programming concepts using an instruction-tuned Small Language Model (SLM).

---

## 📌 Project Overview

This project implements a complete Small Language Model (SLM) solution for beginner Python education without relying on paid hosted LLM APIs (such as OpenAI or Gemini).

### Core Capabilities
- **Beginner Q&A:** Answers conceptual Python questions in plain, accessible language.
- **Code Explanation:** Deconstructs Python snippets line-by-line.
- **Error Diagnostics:** Explains exceptions and common syntax/runtime errors with actionable fixes.
- **Code Generation:** Provides idiomatic, beginner-friendly Python examples.
- **Practice Generation:** Creates targeted exercises with conceptual hints.

---

## 💻 Hardware & Environment Profile (Verified Audit)

The environment has been independently audited and verified on the physical Acer laptop:
- **Operating System:** Windows 11 Home / Pro (Build 10.0.26200, AMD64)
- **CPU:** Intel(R) Core(TM) i3-N305 (8 physical Alder Lake-N cores, 8 threads, ~1.80 GHz base)
- **Host RAM:** **7.75 GB Total Physical RAM** (~0.88 GB free under nominal desktop load, 88%–93% load)
- **Pagefile (Virtual Memory):** 12.95 GB Total (2.30 GB Available)
- **Dedicated GPU:** **None** (System uses integrated **Intel(R) UHD Graphics**; no NVIDIA discrete GPU or CUDA toolkit)
- **Storage Profile:**
  - **Drive C:\\ (Primary OS):** 136.96 GB Total, **7.62 GB Free**
  - **Drive D:\\ (Data Partition):** 50.00 GB Total, **49.91 GB Free** (used as overflow / backup storage)
- **Python Toolchain:** Python 3.10.11 (`C:\Users\acer\AppData\Local\Programs\Python\Python310\python.exe`)
- **Version Control:** Git 2.54.0.windows.1

### Key Engineering Decisions for this Environment
1. **Local Inference:** Lightweight SLM running in CPU mode with memory-conscious configurations (INT4/GGUF or quantized settings, constrained token budgets) to avoid out-of-memory errors on 7.75 GB RAM.
2. **LoRA Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT / LoRA) executed on **Google Colab GPU** (free tier NVIDIA T4 with 15 GB VRAM) to guarantee fast convergence and eliminate local OOM crashes.
3. **Storage Hygiene:** Models and virtual environments are strictly isolated; Drive C: cache is kept bounded to protect the ~7.6 GB free space, with Drive D: (~49.9 GB free) available for cache redirection if required.

---

## 📁 Repository Structure

```
slm-python-learning-assistant/
│
├── README.md                           # Master project documentation
├── requirements.txt                    # Project dependencies for Python 3.10
├── .gitignore                          # Git ignore rules for virtualenvs, models, caches
├── .env.example                        # Environment variable template
│
├── configs/
│   └── model_config.yaml               # Centralized hyperparameters & model settings
│
├── data/
│   ├── raw/                            # Unprocessed data source files
│   ├── processed/                      # Cleaned JSONL data splits
│   ├── train.jsonl                     # 80% Training set
│   ├── validation.jsonl                # 10% Validation set
│   └── test.jsonl                      # 10% Held-out test set
│
├── notebooks/
│   └── slm_lora_training.ipynb         # Google Colab LoRA fine-tuning notebook
│
├── scripts/
│   └── inspect_env.py                  # Standalone hardware & system inspection utility
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_dataset.py          # Dataset generation, cleaning, validation, splitting
│   ├── model/
│   │   ├── __init__.py
│   │   └── load_model.py               # Auto-device model & tokenizer loader
│   │   ├── inference.py                # CPU-optimized inference pipeline
│   │   └── train_lora.py               # LoRA training execution script
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py                 # Multi-metric benchmark (AST, validity, latency)
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── quantize.py                 # Quantization & memory optimization
│   └── api/
│       ├── __init__.py
│       ├── main.py                     # FastAPI REST API endpoints
│       └── schemas.py                  # Pydantic request/response schemas
│
├── app/
│   └── streamlit_app.py                # Interactive web interface
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py                 # Unit tests for dataset validation & deduplication
│   ├── test_api.py                     # API route & payload schema tests
│   └── test_inference.py               # Inference integration tests
│
├── experiments/
│   ├── baseline_results.json           # Raw outputs & latencies of baseline SLM
│   ├── optimization_results.json       # Memory & latency benchmark after optimization
│   └── evaluation_results.json         # Comparative evaluation records
│
└── reports/
    ├── hardware_audit.md               # Independent hardware audit & memory calculations
    ├── phase1_audit.md                 # Phase 1 audit & remediation log
    ├── model_comparison.md             # Phase 2: SLM Candidate Comparison & Selection
    ├── dataset_report.md               # Phase 3: Dataset statistics and split validation
    ├── baseline_report.md              # Phase 4: Baseline model performance analysis
    ├── optimization_report.md          # Phase 6: Optimization benchmarks
    ├── benchmark_report.md             # Phase 7: Pre-vs-post fine-tuning comparison
    └── final_report.md                 # Phase 8: Comprehensive technical project report
```

---

## 🚀 Development Workflow Roadmap

- [x] **Phase 1: Environment & Workspace Verification** — Hardware auditing, git initialization, modular project layout, initial configurations.
- [x] **Phase 1 Audit & Remediation** — Independent re-audit, `scripts/inspect_env.py`, memory footprint modeling, updated documentation.
- [ ] **Phase 2: SLM Research & Model Selection** — Comparative research on Qwen2.5-1.5B, SmolLM2-1.7B, and Llama-3.2-1B against local CPU & Colab GPU profiles.
- [ ] **Phase 3: Dataset Preparation** — Domain-specific Python QA creation (300–500 samples), 80/10/10 split, AST & JSONL validation.
- [ ] **Phase 4: Baseline Model Implementation** — Local CPU inference, latency & memory tracking, baseline evaluation.
- [ ] **Phase 5: LoRA Fine-Tuning** — Adapter training on Google Colab GPU, adapter export, training logs.
- [ ] **Phase 6: Model Optimization** — Quantization and generation parameter tuning.
- [ ] **Phase 7: Comprehensive Evaluation** — Automated AST parsing, technical accuracy, speed and memory benchmarks on held-out test split.
- [ ] **Phase 8: FastAPI Backend & Streamlit Frontend** — Production REST API, interactive UI, Pytest suite, final technical report.

---

## 📄 License & Attribution
Designed for educational purposes as an internship / mentor assignment. Free and open-source models and libraries only.
