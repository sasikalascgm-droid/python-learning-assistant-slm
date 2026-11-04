# Comprehensive Dataset Quality, AST & Execution Audit Report

**Audit Target:** `data/raw/python_qa_raw.jsonl`  
**Audit Evaluation Status:** ✅ PASSED  
**Total Records Analyzed:** 382  
**Auditor:** Senior SLM Research Engineer  

---

## 1. Schema & Structural Integrity Audit

| Validation Check | Result | Metric / Standard | Evaluation Status |
| :--- | :---: | :---: | :--- |
| **JSON Line Formatting** | 0 errors | 100% compliant JSON Lines | ✅ PASSED |
| **Schema Completeness** | 0 missing keys | Required: `instruction`, `input`, `output` | ✅ PASSED |
| **Non-Empty Text Validation** | 0 empty fields | Zero blank instructions or answers | ✅ PASSED |
| **Exact Duplicate Detection** | 0 duplicates | Normalized instruction + input | ✅ PASSED |

---

## 2. Python Code Syntax Validation (AST Parsing)

> [!NOTE]
> AST (Abstract Syntax Tree) parsing verifies that code adheres to the official Python 3.10 grammar specification. It guarantees that the code parses cleanly, but does not execute the code.

- **Total Python Code Blocks Extracted:** 213
- **Syntactically Valid AST Nodes:** 213 / 213 (100.0%)
- **Syntax / Indentation Errors:** 0

> **All 213 Python code blocks parsed cleanly without syntax errors under Python 3.10 AST inspection.**

---

## 3. Safe Execution Validation (Runtime Correctness)

To provide verification beyond static AST syntax analysis, safe self-contained code examples were dynamically executed in an isolated Python namespace:
- **Total Self-Contained Snippets Executed:** 200
- **Execution Success Rate:** **200 / 200 (100.0%)**
- **Runtime Crashes / Uncaught Exceptions:** 0
- **Intentional Buggy Demonstrations Isolated:** 1 (e.g. debugging drill snippets)
- **Skipped Interactive Prompts (`input()`):** 1 (bypassed to avoid blocking automated pipelines)
- **Skipped File I/O Operations (`open()`):** 11 (bypassed to avoid unmocked filesystem dependencies)

> **Zero runtime exceptions observed across all self-contained executable code examples.**

---

## 4. Duplicate & Paraphrase Leakage Audit

Pairwise Jaccard similarity was evaluated across normalized questions:
- **Exact Duplicate Count:** **0**
- **High Similarity ($\ge 0.85$):** **4 pairs** (clustered into identical splits)
- **Moderate Similarity ($0.70 \le 	ext{sim} < 0.85$):** **220 pairs**

### High-Similarity Groupings (Protected from Cross-Split Leakage):
- Similarity: **1.0**
  - Record A (Line 21): *What is the difference between is and == in Python?*
  - Record B (Line 47): *What is the difference between / and // in Python?*
- Similarity: **0.857**
  - Record A (Line 131): *What is a list in Python?*
  - Record B (Line 136): *What is a list comprehension in Python?*
- Similarity: **0.857**
  - Record A (Line 283): *What is a module in Python?*
  - Record B (Line 301): *What is __all__ in a Python module?*
- Similarity: **0.917**
  - Record A (Line 369): *Generate a practice problem on Python dictionaries with a hint and solution.*
  - Record B (Line 380): *Generate a practice problem on Python dictionaries and sorting with a hint and solution.*

---

## 5. Human & Pedagogical Content Review

A qualitative inspection was conducted across the 18 domain modules:
1. **Target Audience Alignment:** Vocabulary and conceptual framing are tailored to beginner programmers (simplifying technical jargon like pointers or memory allocations into clear concepts).
2. **Pedagogical Structure:** Questions follow a natural progression: definition $\rightarrow$ illustrative example $\rightarrow$ breakdown $\rightarrow$ edge cases.
3. **Traceback Diagnostics:** Error explanations (SyntaxError, NameError, TypeError, IndexError, KeyError) accurately deconstruct common student errors and provide actionable solutions.
4. **Drills & Practice:** Practice challenges explicitly separate the task requirements, structural hints, and verified solutions.

---

## 6. Known Limitations

1. **Static AST vs. Dynamic Interactive Code:** AST validation confirms that interactive questions containing `input()` have valid syntax, but automated unit tests cannot execute interactive prompts without mocking standard input.
2. **Dataset Scale:** 382 records provide a focused, high-density domain tuning corpus for LoRA. Future iterations may expand to 1,000+ examples to cover advanced libraries (NumPy, Pandas).
