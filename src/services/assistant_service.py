"""Assistant Orchestrator Service with Dynamic Fallback & Static AST Analysis."""

import ast
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.prompt_service import PromptService
from src.services.model_service import ModelService


FALLBACK_KNOWLEDGE_BASE = {
    "list": {
        "concept": "A list in Python is an ordered, mutable sequence that can store elements of various data types.",
        "example": "fruits = ['apple', 'banana', 'orange']\nfruits.append('mango')\nprint(fruits[0])  # Output: apple",
        "key_points": ["Lists use square brackets [].", "Lists are mutable (can be modified in-place).", "Elements are zero-indexed."]
    },
    "dict": {
        "concept": "A dictionary is an unordered, mutable collection of key-value pairs where each key must be unique and immutable.",
        "example": "student = {'name': 'Alice', 'grade': 'A'}\nstudent['age'] = 20\nprint(student.get('name'))  # Output: Alice",
        "key_points": ["Dictionaries use curly braces {}.", "Keys must be unique and hashable (e.g., strings, ints).", "Values are accessed using keys, not indices."]
    },
    "loop": {
        "concept": "Loops in Python allow you to iterate over a sequence (for loop) or execute a block while a condition remains True (while loop).",
        "example": "for i in range(1, 4):\n    print(f'Count: {i}')\n# Output: Count: 1, Count: 2, Count: 3",
        "key_points": ["The 'for' loop iterates over iterables like lists or ranges.", "The 'while' loop repeats until the test expression evaluates to False.", "Use 'break' to exit early or 'continue' to skip an iteration."]
    },
    "function": {
        "concept": "A function is a reusable block of code that runs only when called. It accepts parameters and can return a result.",
        "example": "def add(a: int, b: int) -> int:\n    return a + b\n\nresult = add(5, 3)\nprint(result)  # Output: 8",
        "key_points": ["Defined using the 'def' keyword.", "Return values are sent back using the 'return' statement.", "Promotes the DRY (Don't Repeat Yourself) principle."]
    },
    "exception": {
        "concept": "Exceptions are runtime errors detected during code execution. Python handles them gracefully using try-except blocks.",
        "example": "try:\n    num = int('invalid')\nexcept ValueError as e:\n    print(f'Caught error: {e}')",
        "key_points": ["Code that might raise an error goes inside the 'try' block.", "The 'except' block catches and handles specific exceptions.", "The 'finally' block always executes regardless of exceptions."]
    }
}


class AssistantService:
    """Coordinates pedagogical responses, AST validation, and fallback mechanisms."""

    def __init__(
        self,
        model_service: Optional[ModelService] = None,
        prompt_service: Optional[PromptService] = None
    ):
        self.model_service = model_service or ModelService.get_instance()
        self.prompt_service = prompt_service or PromptService()

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answers student Python questions using SLM or development fallback."""
        t0 = time.time()
        q_clean = question.strip()

        # If model is loaded, execute live SLM inference
        if self.model_service.is_loaded:
            prompt = self.prompt_service.build_ask_prompt(q_clean)
            raw_response = self.model_service.generate(prompt)
            dt = round(time.time() - t0, 3)
            return {
                "question": q_clean,
                "answer": raw_response,
                "mode": "slm_inference",
                "model_status": "loaded",
                "latency_seconds": dt
            }

        # Development Fallback Mode
        matched_topic = None
        q_lower = q_clean.lower()
        for key in FALLBACK_KNOWLEDGE_BASE:
            if key in q_lower:
                matched_topic = key
                break

        if matched_topic:
            info = FALLBACK_KNOWLEDGE_BASE[matched_topic]
            answer = (
                f"### Explanation:\n{info['concept']}\n\n"
                f"### Key Concepts:\n" + "\n".join(f"- {p}" for p in info['key_points']) + "\n\n"
                f"### Python Example:\n```python\n{info['example']}\n```"
            )
            code_ex = info['example']
        else:
            answer = (
                f"Here is a guide to understanding your question: **{q_clean}**\n\n"
                "In Python, clarity and simplicity are core design philosophies. "
                "Always check variable types, ensure proper indentation (4 spaces), "
                "and leverage built-in data structures (lists, dicts, sets, tuples) for clean solutions.\n\n"
                "```python\n# Example illustration\nx = 10\nprint(f'Value of x: {x}')\n```"
            )
            code_ex = "x = 10\nprint(f'Value of x: {x}')"

        dt = round(time.time() - t0, 3)
        return {
            "question": q_clean,
            "answer": answer,
            "code_example": code_ex,
            "mode": "development_fallback",
            "model_status": "not_loaded",
            "note": "Response generated via development fallback. Real SLM inference requires loaded Qwen2.5-1.5B model.",
            "latency_seconds": dt
        }

    def explain_code(self, code: str) -> Dict[str, Any]:
        """Statically inspects code with AST and breaks it down line-by-line."""
        t0 = time.time()
        code_clean = code.strip()

        # 1. Static AST Syntax Parsing (Completely safe; never executes arbitrary code)
        syntax_valid = True
        syntax_error = None
        try:
            ast.parse(code_clean)
        except SyntaxError as e:
            syntax_valid = False
            syntax_error = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"

        # 2. Line-by-line static breakdown
        lines = code_clean.split("\n")
        line_breakdown = []
        for idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            explanation = "Executes statement."
            if "=" in line_str and not line_str.startswith("if") and not line_str.startswith("while"):
                explanation = "Assigns value to variable."
            elif line_str.startswith("def "):
                fn_name = line_str.split("(")[0].replace("def ", "").strip()
                explanation = f"Defines function '{fn_name}'."
            elif line_str.startswith("for "):
                explanation = "Initializes a for-loop iterating over a sequence."
            elif line_str.startswith("if ") or line_str.startswith("elif "):
                explanation = "Evaluates boolean conditional branch."
            elif line_str.startswith("else:"):
                explanation = "Default branch if preceding conditions evaluate to False."
            elif line_str.startswith("return "):
                explanation = "Returns output value from function to caller."
            elif "print(" in line_str:
                explanation = "Outputs value to standard console."

            line_breakdown.append({
                "line_number": idx,
                "code": line,
                "explanation": explanation
            })

        # 3. Model or Fallback Generation
        if self.model_service.is_loaded and syntax_valid:
            prompt = self.prompt_service.build_explain_code_prompt(code_clean)
            raw_response = self.model_service.generate(prompt)
            dt = round(time.time() - t0, 3)
            return {
                "code": code_clean,
                "syntax_valid": True,
                "syntax_error": None,
                "explanation": raw_response,
                "line_by_line": line_breakdown,
                "mode": "slm_inference",
                "model_status": "loaded",
                "latency_seconds": dt
            }

        # Fallback explanation
        if not syntax_valid:
            overall_exp = (
                f"### ⚠️ Code Syntax Issue Detected:\n{syntax_error}\n\n"
                "Please fix the syntax error above before running the snippet. "
                "Common syntax mistakes include missing colons `:` at the end of loop/function headers, "
                "unmatched parentheses, or incorrect indentation."
            )
        else:
            overall_exp = (
                f"The provided Python code contains {len(lines)} lines and is syntactically valid.\n\n"
                "### Line-by-Line Walkthrough:\n" +
                "\n".join(f"- **Line {item['line_number']}** (`{item['code'].strip()}`): {item['explanation']}" for item in line_breakdown)
            )

        dt = round(time.time() - t0, 3)
        return {
            "code": code_clean,
            "syntax_valid": syntax_valid,
            "syntax_error": syntax_error,
            "explanation": overall_exp,
            "line_by_line": line_breakdown,
            "mode": "development_fallback",
            "model_status": "not_loaded",
            "note": "Analysis generated via static AST analysis and fallback logic.",
            "latency_seconds": dt
        }

    def generate_practice(self, topic: str, difficulty: str = "beginner") -> Dict[str, Any]:
        """Generates progressive practice exercises with hints."""
        t0 = time.time()
        t_clean = topic.strip()
        diff_clean = difficulty.strip().lower()

        if self.model_service.is_loaded:
            prompt = self.prompt_service.build_practice_prompt(t_clean, diff_clean)
            raw_response = self.model_service.generate(prompt)
            dt = round(time.time() - t0, 3)
            return {
                "topic": t_clean,
                "difficulty": diff_clean,
                "title": f"Practice: {t_clean.title()}",
                "problem_statement": raw_response,
                "starter_code": f"# Write your solution for '{t_clean}' below:\n",
                "hints": ["Read the problem carefully.", "Test edge cases.", "Use standard built-ins."],
                "solution": None,
                "mode": "slm_inference",
                "model_status": "loaded",
                "latency_seconds": dt
            }

        # Fallback practice generator
        problems = {
            "list": {
                "title": "Reverse and Filter Positive Numbers",
                "problem": "Write a function `filter_positives(nums)` that takes a list of integers and returns only the positive numbers in reverse order.",
                "starter": "def filter_positives(nums: list[int]) -> list[int]:\n    # Your code here\n    pass\n",
                "hints": [
                    "Hint 1: You can filter numbers using a list comprehension with an 'if num > 0' condition.",
                    "Hint 2: To reverse a list, you can use slice notation `[::-1]` or the `reversed()` function.",
                    "Hint 3: Combining both: `[x for x in nums if x > 0][::-1]`"
                ],
                "solution": "def filter_positives(nums):\n    return [x for x in nums if x > 0][::-1]"
            },
            "dict": {
                "title": "Word Frequency Counter",
                "problem": "Write a function `count_words(words)` that takes a list of words and returns a dictionary with each word's count.",
                "starter": "def count_words(words: list[str]) -> dict[str, int]:\n    # Your code here\n    pass\n",
                "hints": [
                    "Hint 1: Initialize an empty dictionary `counts = {}`.",
                    "Hint 2: Iterate over each word. Use `counts[w] = counts.get(w, 0) + 1`.",
                    "Hint 3: Return the `counts` dictionary."
                ],
                "solution": "def count_words(words):\n    counts = {}\n    for w in words:\n        counts[w] = counts.get(w, 0) + 1\n    return counts"
            }
        }

        matched = None
        for k in problems:
            if k in t_clean.lower():
                matched = problems[k]
                break

        if matched:
            p_data = matched
        else:
            p_data = {
                "title": f"Mastering {t_clean.title()}",
                "problem": f"Write a clean, functional Python program demonstrating core concepts of '{t_clean}'.",
                "starter": f"# Implement solution for {t_clean}\ndef solve():\n    pass\n",
                "hints": [
                    f"Hint 1: Review Python 3.10 syntax for {t_clean}.",
                    "Hint 2: Break the problem into small, manageable helper functions.",
                    "Hint 3: Print intermediate variables to trace execution."
                ],
                "solution": f"# Sample solution for {t_clean}\nprint('Concept mastered!')"
            }

        dt = round(time.time() - t0, 3)
        return {
            "topic": t_clean,
            "difficulty": diff_clean,
            "title": p_data["title"],
            "problem_statement": p_data["problem"],
            "starter_code": p_data["starter"],
            "hints": p_data["hints"],
            "solution": p_data["solution"],
            "mode": "development_fallback",
            "model_status": "not_loaded",
            "note": "Practice problem generated via development fallback.",
            "latency_seconds": dt
        }
