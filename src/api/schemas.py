"""Pydantic Request and Response Schemas for FastAPI Endpoints."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class RootResponse(BaseModel):
    """Schema for root endpoint GET /."""
    project: str = Field(..., description="Project name")
    version: str = Field(..., description="Application version")
    status: str = Field(..., description="Service status")
    docs_url: str = Field(..., description="Interactive Swagger API documentation URL")


class HealthResponse(BaseModel):
    """Schema for health status endpoint GET /health."""
    status: str = Field(..., description="System operational status")
    project: str = Field(..., description="Project name")
    version: str = Field(..., description="Service version")
    model_id: str = Field(..., description="Configured Small Language Model ID")
    revision: str = Field(..., description="Pinned Git commit SHA of base model")
    model_loaded: bool = Field(..., description="Whether model weights are currently resident in memory")
    adapter_exists: bool = Field(..., description="Whether trained LoRA adapter weights exist on disk")
    adapter_loaded: bool = Field(..., description="Whether LoRA adapter is attached to active model")
    cuda_available: bool = Field(..., description="Whether an NVIDIA CUDA GPU is available")
    gpu_name: Optional[str] = Field(None, description="Model of detected CUDA GPU")
    gpu_vram_gb: float = Field(0.0, description="Available GPU VRAM in gigabytes")
    host_ram_total_gb: float = Field(..., description="Total physical system RAM in gigabytes")
    host_ram_avail_gb: float = Field(..., description="Free physical system RAM in gigabytes")
    active_configuration: str = Field(..., description="Current active execution mode")
    adapter_status: str = Field(..., description="Status of LoRA adapter availability")


class AskRequest(BaseModel):
    """Schema for student question request POST /ask."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language Python question from student",
        examples=["What is a list comprehension and how do I use it?"]
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        return cleaned


class AskResponse(BaseModel):
    """Schema for student question response POST /ask."""
    question: str
    answer: str
    code_example: Optional[str] = None
    mode: str = Field(..., description="'slm_inference' or 'development_fallback'")
    model_status: str = Field(..., description="'loaded' or 'not_loaded'")
    latency_seconds: float
    note: Optional[str] = None


class ExplainCodeRequest(BaseModel):
    """Schema for code explanation request POST /explain-code."""
    code: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Python code snippet to be analyzed and explained line by line",
        examples=["def square(x):\n    return x * x\n\nprint(square(4))"]
    )

    @field_validator("code")
    @classmethod
    def validate_code_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Code cannot be empty or whitespace only.")
        return cleaned


class LineExplanation(BaseModel):
    """Schema for single line code deconstruction."""
    line_number: int
    code: str
    explanation: str


class ExplainCodeResponse(BaseModel):
    """Schema for code explanation response POST /explain-code."""
    code: str
    syntax_valid: bool
    syntax_error: Optional[str] = None
    explanation: str
    line_by_line: List[LineExplanation] = []
    mode: str
    model_status: str
    latency_seconds: float
    note: Optional[str] = None


class PracticeRequest(BaseModel):
    """Schema for practice exercise generation POST /practice."""
    topic: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Python programming topic",
        examples=["lists", "dictionaries", "for loops", "functions"]
    )
    difficulty: str = Field(
        "beginner",
        description="Exercise difficulty level ('beginner', 'intermediate', 'advanced')"
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Topic cannot be empty or whitespace only.")
        return cleaned

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        allowed = {"beginner", "intermediate", "advanced"}
        low = v.strip().lower()
        if low not in allowed:
            raise ValueError(f"Difficulty must be one of {allowed}, got '{v}'.")
        return low


class PracticeResponse(BaseModel):
    """Schema for practice exercise response POST /practice."""
    topic: str
    difficulty: str
    title: str
    problem_statement: str
    starter_code: Optional[str] = None
    hints: List[str] = []
    solution: Optional[str] = None
    mode: str
    model_status: str
    latency_seconds: float
    note: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Schema for explicit model load trigger POST /model/load."""
    force_cpu: bool = Field(False, description="Force loading on CPU even if CUDA is detected")


class LoadModelResponse(BaseModel):
    """Schema for model load response POST /model/load."""
    status: str
    model_id: str
    revision: str
    device: str
    adapter_loaded: bool
