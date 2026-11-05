"""Comprehensive Unit and Integration Tests for FastAPI Endpoints, Schemas, and Handlers."""

import pytest
from pydantic import ValidationError
from fastapi import HTTPException, status
from src.api.main import app, read_root
from src.api.schemas import (
    RootResponse,
    HealthResponse,
    AskRequest,
    AskResponse,
    ExplainCodeRequest,
    ExplainCodeResponse,
    PracticeRequest,
    PracticeResponse,
    LoadModelRequest,
    LoadModelResponse
)
from src.api.routes.health import get_health, load_model_explicitly
from src.api.routes.ask import ask_question
from src.api.routes.explain_code import explain_code
from src.api.routes.practice import generate_practice
from src.services.model_service import ModelService
from src.services.assistant_service import AssistantService


@pytest.fixture
def model_service():
    """Provides the singleton ModelService."""
    return ModelService.get_instance()


@pytest.fixture
def assistant_service(model_service):
    """Provides AssistantService configured with ModelService."""
    return AssistantService(model_service=model_service)


def test_fastapi_app_registration_and_openapi():
    """FastAPI app has all routes registered and generates valid OpenAPI spec."""
    openapi_schema = app.openapi()
    assert openapi_schema["info"]["title"] == "Python Learning Assistant API"
    assert openapi_schema["info"]["version"] == "1.0.0"

    paths = list(openapi_schema["paths"].keys())
    assert "/" in paths
    assert "/health" in paths
    assert "/model/load" in paths
    assert "/ask" in paths
    assert "/explain-code" in paths
    assert "/practice" in paths


def test_root_endpoint_handler():
    """read_root() returns project metadata, version, status, and docs link."""
    response = read_root()
    assert isinstance(response, RootResponse)
    assert response.project == "slm-python-learning-assistant"
    assert response.version == "1.0.0"
    assert response.status == "operational"
    assert response.docs_url == "/docs"


def test_health_endpoint_handler(model_service):
    """get_health() returns system metrics, standby status, and pinned revision."""
    response = get_health(model_service)
    assert isinstance(response, HealthResponse)
    assert response.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert response.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert response.model_loaded is False
    assert response.adapter_exists is False
    assert response.cuda_available is False
    assert response.host_ram_total_gb > 0
    assert response.host_ram_avail_gb >= 0
    assert response.active_configuration == "development_fallback"
    assert response.adapter_status == "pending_colab_training"


def test_model_load_insufficient_memory_protection(model_service):
    """load_model_explicitly() handles memory headroom protection gracefully."""
    # When available RAM < 1.5 GB and force_cpu is False, raises HTTPException 503 or succeeds if memory is free
    req = LoadModelRequest(force_cpu=False)
    try:
        res = load_model_explicitly(req, model_service)
        assert isinstance(res, LoadModelResponse)
    except HTTPException as exc:
        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Insufficient memory" in exc.detail


def test_ask_endpoint_valid_question(assistant_service):
    """ask_question() handles valid questions and returns pedagogical answers."""
    req = AskRequest(question="How do lists work in Python and how do I append items?")
    res = ask_question(req, assistant_service)
    assert isinstance(res, AskResponse)
    assert res.question == req.question
    assert "list" in res.answer.lower()
    assert res.code_example is not None
    assert res.mode == "development_fallback"
    assert res.model_status == "not_loaded"
    assert res.latency_seconds >= 0


def test_ask_endpoint_general_question(assistant_service):
    """ask_question() provides helpful advice for general programming questions."""
    req = AskRequest(question="What is PEP 8 and why is code styling important?")
    res = ask_question(req, assistant_service)
    assert isinstance(res, AskResponse)
    assert res.mode == "development_fallback"
    assert "Python" in res.answer


def test_ask_request_validation():
    """AskRequest Pydantic validation rejects empty, whitespace, and short questions."""
    # Empty string
    with pytest.raises(ValidationError):
        AskRequest(question="")

    # Whitespace only
    with pytest.raises(ValidationError):
        AskRequest(question="    ")

    # Less than 3 chars
    with pytest.raises(ValidationError):
        AskRequest(question="ab")

    # Valid question trims whitespace
    valid = AskRequest(question="  What is a set?  ")
    assert valid.question == "What is a set?"


def test_explain_code_valid_syntax(assistant_service):
    """explain_code() analyzes valid Python snippets and returns line-by-line breakdown."""
    code = "def add(x, y):\n    return x + y\n\nresult = add(2, 3)\nprint(result)"
    req = ExplainCodeRequest(code=code)
    res = explain_code(req, assistant_service)
    assert isinstance(res, ExplainCodeResponse)
    assert res.syntax_valid is True
    assert res.syntax_error is None
    assert len(res.line_by_line) >= 4
    assert res.mode == "development_fallback"

    line_numbers = [item.line_number for item in res.line_by_line]
    assert 1 in line_numbers
    assert 2 in line_numbers


def test_explain_code_syntax_error(assistant_service):
    """explain_code() identifies syntax error details via AST without crashing."""
    code = "def broken(\n    print('Unclosed parenthesis')"
    req = ExplainCodeRequest(code=code)
    res = explain_code(req, assistant_service)
    assert isinstance(res, ExplainCodeResponse)
    assert res.syntax_valid is False
    assert res.syntax_error is not None
    assert "SyntaxError" in res.syntax_error
    assert "⚠️" in res.explanation


def test_explain_code_request_validation():
    """ExplainCodeRequest validates non-empty code snippet."""
    with pytest.raises(ValidationError):
        ExplainCodeRequest(code="")

    with pytest.raises(ValidationError):
        ExplainCodeRequest(code="   \n  \t  ")

    valid = ExplainCodeRequest(code="  x = 1\nprint(x)  ")
    assert valid.code == "x = 1\nprint(x)"


def test_practice_endpoint_valid_request(assistant_service):
    """generate_practice() produces structured practice exercise with hints."""
    req = PracticeRequest(topic="lists", difficulty="beginner")
    res = generate_practice(req, assistant_service)
    assert isinstance(res, PracticeResponse)
    assert res.topic == "lists"
    assert res.difficulty == "beginner"
    assert len(res.title) > 0
    assert len(res.problem_statement) > 0
    assert len(res.hints) >= 1
    assert res.mode == "development_fallback"


def test_practice_request_validation():
    """PracticeRequest enforces valid topics and allowed difficulty levels."""
    # Invalid difficulty
    with pytest.raises(ValidationError):
        PracticeRequest(topic="loops", difficulty="extreme")

    # Empty topic
    with pytest.raises(ValidationError):
        PracticeRequest(topic="", difficulty="beginner")

    # Whitespace topic
    with pytest.raises(ValidationError):
        PracticeRequest(topic="   ", difficulty="intermediate")

    # Allowed difficulties case-insensitive
    req_upper = PracticeRequest(topic="functions", difficulty="INTERMEDIATE")
    assert req_upper.difficulty == "intermediate"

    req_adv = PracticeRequest(topic="classes", difficulty="Advanced")
    assert req_adv.difficulty == "advanced"
