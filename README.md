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

## 💻 Hardware & Environment Profile

Based on Phase 1 system inspection:
- **Operating System:** Windows 11 Home / Pro (x86_64 AMD64)
- **CPU:** Intel64 Family 6 Model 190 (8 cores)
- **Host RAM:** 7.75 GB Total (approx. 1.0 GB currently free)
- **Disk Storage:** 136.96 GB Total (7.29 GB Free)
- **Dedicated GPU:** None detected (No NVIDIA GPU / CUDA). Local execution uses CPU.
- **Python Version:** 3.10.11 (`C:\Users\acer\AppData\Local\Programs\Python\Python310\python.exe`)
- **Version Control:** Git 2.54.0

### Key Engineering Decisions for this Environment
1. **Local Inference:** Lightweight SLM (≤ 1.5B parameters) running in CPU mode with memory-conscious configurations (e.g. `torch.float32` on CPU, optimized token generation budgets).
2. **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT / LoRA) executed on Google Colab GPU (T4 free tier) to prevent local OOM (Out Of Memory) issues on the 8 GB host RAM.
3. **Storage Hygiene:** Models and virtual environments are strictly isolated; disk cache is kept tidy to preserve the ~7.3 GB free disk space.

---

## 📁 Repository Structure

```
slm-python-assistant/
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
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── prepare_dataset.py          # Dataset generation, cleaning, validation, splitting
│   ├── model/
│   │   ├── __init__.py
│   │   ├── load_model.py               # Auto-device model & tokenizer loader
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
- [ ] **Phase 2: SLM Research & Model Selection** — Comparative research on Qwen2.5-1.5B, SmolLM2-1.7B, and Llama-3.2-1B.
- [ ] **Phase 3: Dataset Preparation** — Domain-specific Python QA creation (300–500 samples), 80/10/10 split, AST & JSONL validation.
- [ ] **Phase 4: Baseline Model Implementation** — Local CPU inference, latency & memory tracking, baseline evaluation.
- [ ] **Phase 5: LoRA Fine-Tuning** — Adapter training on Google Colab GPU, adapter export, training logs.
- [ ] **Phase 6: Model Optimization** — Quantization and generation parameter tuning.
- [ ] **Phase 7: Comprehensive Evaluation** — Automated AST parsing, technical accuracy, speed and memory benchmarks on held-out test split.
- [ ] **Phase 8: FastAPI Backend & Streamlit Frontend** — Production REST API, interactive UI, Pytest suite, final technical report.

---

## 📄 License & Attribution
Designed for educational purposes as an internship / mentor assignment. Free and open-source models and libraries only.
