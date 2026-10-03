import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    Provides safe, graceful fallback responses when primary systems fail
    or when input/output safety guards are triggered.
    """

    DEFAULT_SAFETY_FALLBACK = (
        "I cannot answer this query because it triggered our healthcare safety guidelines. "
        "For urgent medical questions or emergencies, please consult a licensed healthcare professional "
        "or contact emergency medical services immediately (e.g., dial 108 or 911)."
    )

    DEFAULT_SYSTEM_FALLBACK = (
        "I apologize, but our healthcare assistant is temporarily unable to process your request. "
        "Please try again shortly or consult a doctor for personalized clinical advice."
    )

    DEFAULT_INJECTION_FALLBACK = (
        "Your request could not be processed due to a security policy violation. "
        "Please rephrase your medical question."
    )

    @classmethod
    def get_fallback_response(
        cls,
        reason: str,
        is_injection: bool = False,
        is_unsafe: bool = False
    ) -> str:
        """Return the appropriate safe fallback response based on error condition."""
        logger.warning(f"Invoking fallback handler. Reason: {reason}")
        if is_injection:
            return cls.DEFAULT_INJECTION_FALLBACK
        if is_unsafe:
            return cls.DEFAULT_SAFETY_FALLBACK
        return cls.DEFAULT_SYSTEM_FALLBACK
