"""Code Explanation & Line-by-Line Deconstruction Endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from src.api.schemas import ExplainCodeRequest, ExplainCodeResponse, LineExplanation
from src.api.dependencies import get_assistant_service
from src.services.assistant_service import AssistantService

router = APIRouter(tags=["Code Explanation"])


@router.post("/explain-code", response_model=ExplainCodeResponse)
def explain_code(
    payload: ExplainCodeRequest,
    assistant_service: AssistantService = Depends(get_assistant_service)
) -> ExplainCodeResponse:
    """
    Deconstructs student Python code line by line and diagnoses syntax errors.
    
    Security Guarantee:
    - Pure static AST parsing: Submitted code is NEVER executed or evaluated on host.
    - Accurately reports line-number syntax issues if code contains SyntaxErrors.
    """
    try:
        result = assistant_service.explain_code(payload.code)
        lines = [
            LineExplanation(
                line_number=item["line_number"],
                code=item["code"],
                explanation=item["explanation"]
            )
            for item in result.get("line_by_line", [])
        ]
        return ExplainCodeResponse(
            code=result["code"],
            syntax_valid=result["syntax_valid"],
            syntax_error=result.get("syntax_error"),
            explanation=result["explanation"],
            line_by_line=lines,
            mode=result["mode"],
            model_status=result["model_status"],
            latency_seconds=result["latency_seconds"],
            note=result.get("note")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while explaining code: {str(e)}"
        )
