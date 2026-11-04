"""Main FastAPI Application Entrypoint."""

import sys
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import RootResponse
from src.api.routes import health, ask, explain_code, practice


app = FastAPI(
    title="Python Learning Assistant API",
    version="1.0.0",
    description=(
        "Production-grade, end-to-end REST API for the Python Learning Assistant, "
        "powered by Qwen2.5-1.5B-Instruct SLM. Features beginner Q&A, line-by-line "
        "code explanation with static AST verification, and progressive practice generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register route modules
app.include_router(health.router)
app.include_router(ask.router)
app.include_router(explain_code.router)
app.include_router(practice.router)


@app.get("/", response_model=RootResponse, tags=["General"])
def read_root() -> RootResponse:
    """Returns application name, version, status, and link to interactive Swagger docs."""
    return RootResponse(
        project="slm-python-learning-assistant",
        version="1.0.0",
        status="operational",
        docs_url="/docs"
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handles data validation failures with clear HTTP 422 responses."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
