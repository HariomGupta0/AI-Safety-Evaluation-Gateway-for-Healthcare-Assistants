from datetime import datetime
from typing import Dict, Optional, Any
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.gateway.pipeline import gateway_pipeline
from app.observability.tracer import tracer
from app.ai.rag import rag_engine

router = APIRouter()


class ChatRequest(BaseModel):
    """Schema for incoming client chat requests."""
    message: str = Field(..., min_length=1, description="Patient medical query or message.")
    user_id: Optional[str] = Field("api_user", description="Identifier for the user or session.")
    channel: Optional[str] = Field("api", description="Client channel name (e.g. 'api', 'web', 'whatsapp').")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="Optional profile (e.g. age, gender).")


class ChatResponse(BaseModel):
    """Schema for outgoing AI Gateway responses."""
    response: str
    trace_id: str
    status: str
    pii_redacted: Dict[str, int]
    model_used: str
    tokens_used: int
    duration_ms: float
    disclaimer_appended: bool
    is_fallback: bool


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """
    Main Gateway chat endpoint.
    Routes request through Input Guard -> Orchestrator (RAG + LLM) -> Output Guard -> Fallback.
    """
    try:
        result = gateway_pipeline.process(
            user_id=payload.user_id or "api_user",
            message=payload.message,
            channel=payload.channel or "api",
            user_profile=payload.user_profile
        )
        return ChatResponse(
            response=result.response,
            trace_id=result.trace_id,
            status=result.status,
            pii_redacted=result.pii_redacted,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            disclaimer_appended=result.disclaimer_appended,
            is_fallback=result.is_fallback
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Gateway error: {str(e)}")


@router.get("/health")
async def health_endpoint():
    """Health check endpoint displaying system and index status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.ENVIRONMENT,
        "mock_mode": settings.is_mock_mode,
        "rag_records": len(rag_engine.df) if rag_engine.df is not None else 0
    }


@router.get("/metrics")
async def metrics_endpoint(x_api_key: Optional[str] = Header(default=None)):
    """
    Observability endpoint returning latency percentiles,
    request volumes, and privacy-sanitized trace logs.
    """
    summary = tracer.get_metrics_summary()

    metrics_key = settings.METRICS_API_KEY
    has_trace_access = bool(metrics_key and x_api_key == metrics_key)
    if not has_trace_access:
        summary["recent_traces"] = []
        summary["detail_level"] = "aggregate"
        return summary

    summary["detail_level"] = "detailed"
    return summary
