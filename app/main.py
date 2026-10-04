import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import settings

# Setup standard logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="AI Safety & Evaluation Gateway",
    description=(
        "Prototype AI Safety and Observability Gateway for Healthcare Assistants. "
        "Provides deterministic PII redaction, prompt injection defense, medical Q&A RAG retrieval, "
        "and latency tracing."
    ),
    version="1.0.0"
)

# CORS configuration:
# - Development default: allow_origins=["*"], credentials disabled (wildcard + credentials is
#   rejected by browsers per the CORS spec, so credentials must be False with "*").
# - Production: set ALLOWED_ORIGINS in .env to specific domains; credentials are then enabled.
_wildcard_cors = settings.ALLOWED_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=not _wildcard_cors,   # credentials=True requires specific origins
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Twilio-Signature"],
)

logger.info(
    "CORS configured | origins=%s | environment=%s",
    settings.ALLOWED_ORIGINS,
    settings.ENVIRONMENT
)

from app.channels.whatsapp import router as whatsapp_router

# Register API and Channel routes
app.include_router(api_router)
app.include_router(whatsapp_router)


@app.get("/")
async def root():
    return {
        "message": "AI Safety & Evaluation Gateway is active.",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }
