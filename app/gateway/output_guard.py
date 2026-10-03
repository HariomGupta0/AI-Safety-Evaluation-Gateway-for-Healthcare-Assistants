import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from app.config import settings
from app.gateway.input_guard import InputGuard


@dataclass
class OutputGuardResult:
    """Result of running Output Guard on model response."""
    passed: bool
    final_text: str
    original_text: str
    pii_detected: bool = False
    is_unsafe: bool = False
    disclaimer_added: bool = False
    rejection_reason: Optional[str] = None
    duration_ms: float = 0.0


class OutputGuard:
    """
    Deterministic Output Guard for AI Safety Gateway.

    Responsibilities:
    1. Output PII check: Ensures model didn't leak/echo sensitive PII.
    2. Basic safety rules: Flags harmful or toxic medical recommendations.
    3. Response validation: Checks format and non-emptiness.
    4. Medical disclaimer: Automatically appends mandatory clinical disclaimer.
    """

    STANDARD_DISCLAIMER = (
        "\n\n*Disclaimer: This information is for educational purposes only and is not a "
        "substitute for professional medical advice, diagnosis, or treatment. Always consult a "
        "qualified healthcare provider for any health concerns.*"
    )

    UNSAFE_PATTERNS = [
        re.compile(r'\b(commit\s+suicide|self[- ]harm|overdose\s+deliberately)\b', re.IGNORECASE),
        re.compile(r'\b(take\s+(?:50|100|\d{3,})\s*(?:tablets|pills|mg\s+of\s+cyanide))\b', re.IGNORECASE),
        re.compile(r'\b(synthesize\s+(?:ricin|anthrax|sarin|poisons?))\b', re.IGNORECASE),
    ]

    def __init__(self):
        self.input_guard = InputGuard()

    def check_safety(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check text against harmful content and unsafe patterns."""
        for pattern in self.UNSAFE_PATTERNS:
            match = pattern.search(text)
            if match:
                return True, f"Harmful content pattern detected: '{match.group(0)}'"
        return False, None

    def validate(self, text: str) -> OutputGuardResult:
        """
        Run full Output Guard checks on raw LLM output.
        """
        start_time = time.perf_counter()

        # Step 1: Format & Empty Check
        if not text or not text.strip():
            duration_ms = (time.perf_counter() - start_time) * 1000
            return OutputGuardResult(
                passed=False,
                final_text="",
                original_text=text or "",
                rejection_reason="LLM produced an empty response.",
                duration_ms=round(duration_ms, 2)
            )

        cleaned = text.strip()
        if len(cleaned) < 10:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return OutputGuardResult(
                passed=False,
                final_text=cleaned,
                original_text=text,
                rejection_reason="LLM response is too short to be clinically meaningful.",
                duration_ms=round(duration_ms, 2)
            )

        # Step 2: Unsafe content check
        is_unsafe, reason = self.check_safety(cleaned)
        if is_unsafe:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return OutputGuardResult(
                passed=False,
                final_text="",
                original_text=text,
                is_unsafe=True,
                rejection_reason=f"Safety violation: {reason}",
                duration_ms=round(duration_ms, 2)
            )

        # Step 3: PII Check on Output (Scrub if model regurgitated any)
        scrubbed_text, pii_counts = self.input_guard.redact_pii(cleaned)
        pii_detected = len(pii_counts) > 0

        # Step 4: Medical Disclaimer Enforcement
        disclaimer_added = False
        final_text = scrubbed_text
        if settings.APPEND_MEDICAL_DISCLAIMER:
            if "disclaimer" not in final_text.lower():
                final_text = final_text + self.STANDARD_DISCLAIMER
                disclaimer_added = True

        duration_ms = (time.perf_counter() - start_time) * 1000
        return OutputGuardResult(
            passed=True,
            final_text=final_text,
            original_text=text,
            pii_detected=pii_detected,
            is_unsafe=False,
            disclaimer_added=disclaimer_added,
            duration_ms=round(duration_ms, 2)
        )
