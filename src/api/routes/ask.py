"""Student Q&A API Endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from src.api.schemas import AskRequest, AskResponse
from src.api.dependencies import get_assistant_service
from src.services.assistant_service import AssistantService

router = APIRouter(tags=["Student Assistance"])


@router.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    assistant_service: AssistantService = Depends(get_assistant_service)
) -> AskResponse:
    """
    Answers natural-language Python questions for beginner students.
    
    - When SLM model is loaded: generates live answer via Qwen2.5-1.5B (ChatML).
    - When SLM model is standby: generates deterministic pedagogical response via development fallback.
    - Transparently tags execution mode in response.
    """
    try:
        result = assistant_service.answer_question(payload.question)
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            code_example=result.get("code_example"),
            mode=result["mode"],
            model_status=result["model_status"],
            latency_seconds=result["latency_seconds"],
            note=result.get("note")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering your question: {str(e)}"
        )
