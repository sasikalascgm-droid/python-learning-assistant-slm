"""Practice Exercise Generation API Endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from src.api.schemas import PracticeRequest, PracticeResponse
from src.api.dependencies import get_assistant_service
from src.services.assistant_service import AssistantService

router = APIRouter(tags=["Practice Problem Generation"])


@router.post("/practice", response_model=PracticeResponse)
def generate_practice(
    payload: PracticeRequest,
    assistant_service: AssistantService = Depends(get_assistant_service)
) -> PracticeResponse:
    """
    Generates targeted Python practice exercises tailored to student topic and difficulty.
    Returns problem statement, starter code, graduated hints, and solution code.
    """
    try:
        result = assistant_service.generate_practice(
            topic=payload.topic,
            difficulty=payload.difficulty
        )
        return PracticeResponse(
            topic=result["topic"],
            difficulty=result["difficulty"],
            title=result["title"],
            problem_statement=result["problem_statement"],
            starter_code=result.get("starter_code"),
            hints=result.get("hints", []),
            solution=result.get("solution"),
            mode=result["mode"],
            model_status=result["model_status"],
            latency_seconds=result["latency_seconds"],
            note=result.get("note")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating practice: {str(e)}"
        )
