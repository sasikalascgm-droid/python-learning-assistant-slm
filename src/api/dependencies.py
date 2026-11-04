"""FastAPI Dependency Injection Providers."""

from src.services.assistant_service import AssistantService
from src.services.model_service import ModelService
from src.services.prompt_service import PromptService


def get_model_service() -> ModelService:
    """Dependency provider for ModelService singleton."""
    return ModelService.get_instance()


def get_prompt_service() -> PromptService:
    """Dependency provider for PromptService."""
    return PromptService()


def get_assistant_service() -> AssistantService:
    """Dependency provider for AssistantService."""
    return AssistantService(
        model_service=get_model_service(),
        prompt_service=get_prompt_service()
    )
