"""Health and System Status API Endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from src.api.schemas import HealthResponse, LoadModelRequest, LoadModelResponse
from src.api.dependencies import get_model_service
from src.services.model_service import ModelService

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
def get_health(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """
    Returns system health status:
    - SLM load status (starts unloaded to preserve host memory)
    - Hardware profile (host RAM free, CUDA GPU presence)
    - Adapter existence on disk
    - Active execution configuration
    """
    sys_health = model_service.get_system_health()
    return HealthResponse(
        status=sys_health["status"],
        project="slm-python-learning-assistant",
        version="1.0.0",
        model_id=sys_health["model_id"],
        revision=sys_health["revision"],
        model_loaded=sys_health["model_loaded"],
        adapter_exists=sys_health["adapter_exists"],
        adapter_loaded=sys_health["adapter_loaded"],
        cuda_available=sys_health["cuda_available"],
        gpu_name=sys_health["gpu_name"],
        gpu_vram_gb=sys_health["gpu_vram_gb"],
        host_ram_total_gb=sys_health["host_ram_total_gb"],
        host_ram_avail_gb=sys_health["host_ram_avail_gb"],
        active_configuration=sys_health["active_configuration"],
        adapter_status=sys_health["adapter_status"]
    )


@router.post("/model/load", response_model=LoadModelResponse)
def load_model_explicitly(
    payload: LoadModelRequest = LoadModelRequest(),
    model_service: ModelService = Depends(get_model_service)
) -> LoadModelResponse:
    """
    Explicitly triggers model loading.
    Protected by memory headroom pre-flight audit to safeguard host RAM.
    """
    try:
        res = model_service.load_model(force_cpu=payload.force_cpu)
        return LoadModelResponse(
            status=res["status"],
            model_id=res.get("model_id", model_service.model_id),
            revision=res.get("revision", model_service.revision),
            device=res.get("device", "cpu"),
            adapter_loaded=res.get("adapter_loaded", False)
        )
    except MemoryError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot load model: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model loading failed: {str(e)}"
        )
