"""Prompt Template and ChatML Construction Service for Qwen2.5-1.5B-Instruct."""

DEFAULT_SYSTEM_PROMPT = """You are an expert, patient, and friendly Python Learning Assistant.
Your mission is to help beginner students master Python programming concepts.
Follow these core educational principles:
1. Explain concepts in clear, simple English without unnecessary academic jargon.
2. Provide clean, idiomatic, and bug-free Python 3.10 code examples with comments.
3. Break down code step-by-step or line-by-line when explaining snippets.
4. When diagnosing errors, clearly explain the root cause and provide the exact fix.
5. Always encourage the student and provide beginner-friendly hints for practice.
"""


class PromptService:
    """Constructs ChatML prompts adhering to Qwen2.5 chat formatting."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt.strip()

    def format_chatml(self, user_content: str) -> str:
        """Formats user input into Qwen ChatML tags."""
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def build_ask_prompt(self, question: str) -> str:
        """Constructs prompt for answering student Python questions."""
        user_msg = (
            f"Question: {question.strip()}\n\n"
            "Please provide a simple, beginner-friendly explanation and a clear, practical Python code example."
        )
        return self.format_chatml(user_msg)

    def build_explain_code_prompt(self, code: str) -> str:
        """Constructs prompt for deconstructing code line by line."""
        user_msg = (
            "Please explain the following Python code line-by-line in plain English, "
            "explaining what each line accomplishes and what the final output is:\n\n"
            f"```python\n{code.strip()}\n```"
        )
        return self.format_chatml(user_msg)

    def build_practice_prompt(self, topic: str, difficulty: str = "beginner") -> str:
        """Constructs prompt for generating targeted practice exercises."""
        user_msg = (
            f"Generate a {difficulty}-level Python practice problem about '{topic.strip()}'.\n"
            "Include:\n"
            "1. Problem Title and Description\n"
            "2. Expected Input/Output\n"
            "3. Starter Code\n"
            "4. Three graduated hints\n"
            "5. Complete solution code"
        )
        return self.format_chatml(user_msg)
