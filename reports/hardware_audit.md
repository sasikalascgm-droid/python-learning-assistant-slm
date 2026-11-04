# Hardware and Environment Audit Report

**Date of Audit:** November 5, 2026  
**Auditor:** Senior SLM Research Engineer & AI Developer  
**Target Environment:** Local Windows Laptop (Acer)  
**Inspection Utility:** `scripts/inspect_env.py` (Registry, Windows Kernel32 APIs, Python 3.10 stdlib)

---

## 1. Executive Summary

An independent, rigorous hardware re-audit was executed on the physical laptop to resolve conflicting assumptions from initial planning (where 16 GB RAM and an unknown GPU were initially estimated). 

The audit conclusively reveals:
1. **Physical RAM:** **7.75 GB Total** (not 16 GB), with available RAM fluctuating between **0.50 GB and 1.00 GB** under standard operating conditions (88%–93% memory load).
2. **GPU:** **Integrated Intel UHD Graphics** only (Intel driver 32.0.101.7088). No dedicated NVIDIA GPU, no CUDA toolkit, and no discrete VRAM exist on this system.
3. **CPU:** **Intel Core i3-N305** (8 physical Alder Lake-N efficiency cores, 8 threads, base frequency 1.80 GHz).
4. **Storage:**
   - **Drive C:\\ (Primary OS):** 136.96 GB Total, **7.62 GB Free** (5.5% free space).
   - **Drive D:\\ (Secondary Partition):** 50.00 GB Total, **49.91 GB Free** (99.8% free space).
5. **Python & Runtime:** **Python 3.10.11** 64-bit (`C:\Users\acer\AppData\Local\Programs\Python\Python310\python.exe`) and **Git 2.54.0.windows.1**.

---

## 2. Detailed Hardware Specifications & Verification Methods

| Hardware Subsystem | Detected Specification | Verification Method | Technical Implications |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 Home/Pro (Build 10.0.26200, AMD64) | `platform.platform()`, `sys.platform` | Requires Windows-compatible binaries and POSIX-free scripting. |
| **Processor (CPU)** | Intel(R) Core(TM) i3-N305 (8 Cores, 8 Threads, ~1805 MHz base) | Windows Registry: `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0` (`ProcessorNameString`) | 8 E-cores provide solid multi-threaded CPU matrix operations, but lack AVX-512. AVX2 supported. |
| **Graphics (GPU)** | Intel(R) UHD Graphics (Driver 32.0.101.7088) | Windows Registry: `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}` | No hardware CUDA or Tensor Cores. Cannot run PyTorch with CUDA acceleration locally. |
| **CUDA / Drivers** | `CUDA_PATH` not set; `HKLM\SOFTWARE\NVIDIA Corporation` does not exist | Registry & Environment variable inspection | PyTorch must run with `device='cpu'`. LoRA training must use Google Colab GPU (T4/A100). |
| **Physical Memory (RAM)** | Total: **7.75 GB** (8,316,641,280 bytes)<br>Available: **0.88 GB** (fluctuates ~0.50–1.00 GB)<br>Current Load: **88%–93%** | Windows Kernel32 API call: `GlobalMemoryStatusEx` (`MEMORYSTATUSEX`) | **Critical bottleneck.** Running unquantized models (>1B parameters) locally in FP32 will immediately exhaust RAM. |
| **Virtual Memory (Pagefile)** | Total: **12.95 GB**<br>Available: **2.30 GB** | Windows Kernel32 API call: `GlobalMemoryStatusEx` | Paging can absorb temporary allocation spikes but causes massive I/O latency. |
| **Drive C: Storage** | Total: 136.96 GB<br>Used: 129.34 GB<br>Free: **7.62 GB** | `shutil.disk_usage("C:\\")` | Hugging Face cache and virtual environments must be carefully bounded to avoid filling Drive C:. |
| **Drive D: Storage** | Total: 50.00 GB<br>Used: 0.09 GB<br>Free: **49.91 GB** | `shutil.disk_usage("D:\\")` | **Key Asset:** Ample storage buffer. Can be utilized for dataset snapshots, offline backups, or HF cache redirection if Drive C: becomes cramped. |
| **Python Toolchain** | Python 3.10.11 (AMD64)<br>Git 2.54.0.windows.1 | Direct process invocation and path resolution | Clean, modern toolchain matching all requirements. |

---

## 3. SLM Precision & Memory Footprint Modeling

To establish realistic constraints for local inference and fine-tuning, the theoretical and empirical memory footprints for 1.0B to 1.7B parameter models were calculated across different precision formats:

$$\text{Weight Memory (GB)} \approx \frac{\text{Parameter Count} \times \text{Bytes per Parameter}}{10^9}$$

$$\text{Total Operational RAM} = \text{Model Weights} + \text{Runtime Framework Overhead (0.6--0.9 GB)} + \text{KV Cache / Context Activations}$$

### Theoretical vs. Operational RAM Requirement Table

| Precision Format | Bits / Parameter | Bytes / Parameter | Model Size (~1.5B Params) | PyTorch Runtime Overhead | Context (2k tokens) KV Cache | Total Working RAM Required | Viability on Laptop (0.88 GB Avail / 7.75 GB Total) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (Single)** | 32 bits | 4 bytes | **~6.16 GB** | ~0.80 GB | ~0.35 GB | **~7.31 GB** | ❌ **Fatal OOM / Thrashing** (exceeds available free RAM; forces heavy disk paging). |
| **FP16 / BF16 (Half)** | 16 bits | 2 bytes | **~3.08 GB** | ~0.70 GB | ~0.20 GB | **~3.98 GB** | ⚠️ **Unstable on CPU** (PyTorch CPU operations in fp16/bf16 often downcast to fp32 or run slow emulation; exceeds free RAM). |
| **INT8 (8-bit Quant)** | 8 bits | 1 byte | **~1.54 GB** | ~0.60 GB | ~0.15 GB | **~2.29 GB** | ⚠️ **Marginal** (requires bitsandbytes or ONNX Runtime; exceeds immediate free RAM unless pagefile absorbs it). |
| **INT4 (4-bit / GGUF Q4)**| 4 bits | 0.5 bytes | **~0.85 – 1.05 GB** | ~0.40 GB | ~0.10 GB | **~1.35 – 1.55 GB** | ✅ **Feasible with lightweight runtime** (e.g. GGUF / ONNX / Torch CPU with dynamic quant or Colab for full precision). |

---

## 4. Key Architectural & Resource Decisions

Based on the verified hardware constraints:

1. **No Local Unquantized FP32 Inference of Large SLMs:**
   - Loading unquantized 1.5B+ weights (~6.1 GB) into memory when only 0.88 GB is free will cause Windows to freeze or crash the process via `MemoryError`.
2. **LoRA Fine-Tuning Environment: Strictly Google Colab GPU (T4 / A100):**
   - Fine-tuning requires storing model weights, optimizer states (AdamW takes $8\times \text{trainable params}$ bytes), gradients, and activations.
   - Even with LoRA ($r=16$), backpropagation on an 8-core CPU with 7.75 GB RAM is infeasible and would take dozens of hours with high OOM risk. Google Colab T4 (15 GB VRAM) is the designated, reproducible training platform.
3. **Storage Hygiene & Drive D: Strategy:**
   - All code, datasets, configurations, and adapter weights are kept compact within the project directory.
   - Large raw weight caches (`~/.cache/huggingface`) can be redirected to `D:\huggingface_cache` if Drive C: drops below 5 GB.
4. **Local FastAPI Backend Architecture:**
   - The FastAPI backend will implement clean abstraction: an inference engine interface allowing CPU inference with memory-efficient settings, or lightweight mock/remote endpoints for resource-constrained environments.
