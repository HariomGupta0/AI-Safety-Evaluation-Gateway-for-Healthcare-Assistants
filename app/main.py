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
        "Production-grade AI Safety and Observability Gateway for Healthcare Assistants. "
        "Provides deterministic PII redaction, prompt injection defense, MedQuAD RAG retrieval, "
        "and latency tracing."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
