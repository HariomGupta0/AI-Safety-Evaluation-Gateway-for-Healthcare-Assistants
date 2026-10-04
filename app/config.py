import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server Settings
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Settings
    # In development the default is ["*"] (open).
    # In staging/production, set this explicitly e.g. https://yourdomain.com
    # Leaving it empty in non-development mode disables cross-origin requests.
    ALLOWED_ORIGINS: List[str] = ["*"]

    # LLM Settings
    GROQ_API_KEY: Optional[str] = None
    PRIMARY_MODEL: str = "llama-3.1-8b-instant"
    FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.2

    # RAG Settings
    MEDQUAD_CSV_PATH: str = "./data/sample_medquad.csv"
    RAG_TOP_K: int = 3

    # Safety Guardrails
    MAX_INPUT_LENGTH: int = 1500
    MIN_INPUT_LENGTH: int = 2
    APPEND_MEDICAL_DISCLAIMER: bool = True

    # Twilio WhatsApp Settings
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None

    # Observability
    METRICS_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "medical-safety-gateway"

    @property
    def is_mock_mode(self) -> bool:
        """Return True if no Groq API key is configured, enabling Mock LLM mode."""
        return not bool(self.GROQ_API_KEY and self.GROQ_API_KEY.strip())

    @property
    def is_development(self) -> bool:
        """Return True when running in development mode."""
        return self.ENVIRONMENT.lower() == "development"


settings = Settings()
