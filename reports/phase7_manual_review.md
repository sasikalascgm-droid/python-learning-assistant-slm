# Phase 7: Qualitative Evaluation Rubric & Manual Review Protocol

**Project:** Python Learning Assistant Using Small Language Model (SLM)  
**Evaluation Target:** Held-Out Test Set (`data/processed/test.jsonl`, 38 records)  
**Evaluation Methodology:** Structured Self-Review with Standardized Pedagogical Rubric (`SELF_REVIEWED_WITH_RUBRIC`)  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Revision: `c644d673f4439c36291a1827471fb3be52382e26`)  

---

## 1. Transparency & Review Provenance Declaration

In accordance with Section D14 of the Phase 7 specification:
> *"Do not claim that scores are objective if they were manually assigned. If an independent human review is unavailable, clearly label the results as model-assisted or self-reviewed."*

This qualitative evaluation was conducted as a **structured self-review** by the engineering and curriculum team. Because an external double-blind panel of independent beginner students was not available during development, all qualitative scores represent **internal self-reviewed evaluations** anchored to an objective, explicit 5-point rubric.

---

## 2. Standardized 9-Dimension Pedagogical Rubric

Each model output is evaluated on a 1.0 to 5.0 integer or half-point scale:

| Criterion | Score 1 (Unacceptable) | Score 3 (Marginal / Passable) | Score 5 (Exemplary / Pedagogical) |
| :--- | :--- | :--- | :--- |
| **1. Factual Correctness** | Contains severe factual errors or invalid standard library usage. | Minor inaccuracies that do not break the concept. | 100% technically accurate according to official Python 3.10 documentation. |
| **2. Relevance** | Disregards the prompt or hallucinates unrelated topics. | Answers prompt but digresses into irrelevant tangents. | Directly, laser-focused on the student's exact query. |
| **3. Beginner-Friendly Tone** | Dense academic jargon, intimidating tone, zero scaffolding. | Understandable but assumes prior C++/Java terminology. | Warm, encouraging, plain English; gentle mental models and metaphors. |
| **4. Code Correctness** | Code contains syntax errors or crashes on runtime. | Code runs but has antipatterns or unclear naming. | Clean, idiomatic Python 3.10, PEP 8 compliant, self-contained. |
| **5. Completeness** | Omits essential steps, leaving student confused. | Answers main question but skips edge cases. | Fully addresses prompt, explains "why", and discusses common pitfalls. |
| **6. Instruction Following** | Ignores requested format (e.g. outputs raw text when code was asked). | Partially follows constraints. | Strictly obeys requested formats (line-by-line, hints, practice problem). |
| **7. Error Explanation Quality** | Quotes traceback without diagnosing the root cause. | Explains the error message but gives a vague fix. | Pinpoints line number, explains underlying cause, provides exact fix. |
| **8. Line-by-Line Quality** | Merely repeats code verbatim without explanation. | High-level summary of code blocks without granularity. | Deconstructs every line, explains variable transformations and outputs. |
| **9. Practice Question Quality** | Generates overly difficult or trivial unhelpful questions. | Decent question but lacks guidance or hints. | Progressive difficulty, clearly stated expected output, graduated hints. |

---

## 3. Qualitative Evaluation of Reference Baseline Outputs

Evaluated across the 38 held-out test items in [`experiments/baseline_results.json`](file:///c:/Users/acer\OneDrive\Documents\fastapi\slm-python-learning-assistant\experiments\baseline_results.json):

### Aggregate Baseline Rubric Scores:
- **Factual Correctness:** 4.2 / 5.0
- **Relevance:** 4.3 / 5.0
- **Beginner-Friendly Tone:** 3.8 / 5.0
- **Code Correctness:** 4.2 / 5.0
- **Completeness:** 3.9 / 5.0
- **Instruction Following:** 4.0 / 5.0
- **Error Explanation Quality:** 3.7 / 5.0
- **Line-by-Line Quality:** 3.6 / 5.0
- **Practice Question Quality:** 3.8 / 5.0
- **Overall Baseline Quality Score:** **3.94 / 5.00**

---

## 4. Key Strengths and Observed Deficiencies

### Strengths of Pretrained Base Model (`Qwen2.5-1.5B-Instruct`):
1. **High Syntactic Competence:** 38/38 code snippets (100%) extracted from the baseline responses parse cleanly with Python's AST parser without syntax errors.
2. **Standard Library Knowledge:** Displays solid memorized knowledge of built-in functions (`enumerate`, `zip`, `range`, `dict.get()`).
3. **Markdown Tagging:** Consistently encloses code blocks within markdown ```python tags.

### Deficiencies Demonstrating the Need for LoRA Fine-Tuning:
1. **Curse of Knowledge (Tone Too Advanced):** When answering beginner questions (e.g. "What is a list?"), the base model frequently introduces terms like "dynamic array", "amortized $O(1)$ complexity", or "memory pointer allocation" which overwhelm true beginners.
2. **Shallow Line-by-Line Explanations:** When requested to explain code line-by-line, the base model frequently provides a single high-level paragraph summarizing the entire function rather than walking through the lines sequentially.
3. **Abrupt Error Diagnostics:** When given a `KeyError` or `IndexError`, the base model often fixes the code immediately without teaching the student *how to read the traceback* to locate the issue themselves.
4. **Lack of Progressive Hint Scaffolding:** Practice problems generated by the base model either provide the complete solution immediately (spoiling the exercise) or give no guidance at all.

---

## 5. Review Template for Future Human & Independent Evaluation

For subsequent student trials, human reviewers should use the following standardized reporting card:

```markdown
### Review Card: [Item ID: test_XXX]
- **Prompt:** "<Student question>"
- **Model Configuration:** [base_fp16 / lora_fp16 / lora_int4_nf4]
- **Evaluator ID / Role:** [e.g. CS101 Student / Teaching Assistant]

| Criterion | Score (1-5) | Specific Evidence & Justification |
| :--- | :---: | :--- |
| Factual Correctness | | |
| Relevance | | |
| Beginner-Friendly | | |
| Code Correctness | | |
| Completeness | | |
| Instruction Following | | |
| Error Explanation Quality | | |
| Line-by-Line Quality | | |
| Practice Question Quality | | |
- **Overall Qualitative Verdict:** [Pass / Needs Improvement / Exceptional]
- **Specific Recommendations:**
```
