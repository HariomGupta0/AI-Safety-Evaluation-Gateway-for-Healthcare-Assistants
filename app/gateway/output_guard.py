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
        # --- Category 1: Self-harm and suicide ---
        # Block any response that encourages or instructs self-harm or suicide.
        re.compile(r'\b(commit\s+suicide|end\s+your\s+life|kill\s+yourself)\b', re.IGNORECASE),
        re.compile(r'\b(self[- ]harm|cut\s+yourself|hurt\s+yourself\s+to\s+feel)\b', re.IGNORECASE),
        re.compile(r'\b(overdose\s+deliberately|intentional\s+overdose)\b', re.IGNORECASE),

        # --- Category 2: Dangerous dosage advice ---
        # Block responses that suggest taking an unsafe quantity of medication.
        # Uses a threshold (50+ tablets/pills, or any 3-digit-or-more quantity).
        re.compile(r'\btake\s+(?:[5-9]\d|\d{3,})\s*(?:tablets?|pills?|capsules?)\b', re.IGNORECASE),
        re.compile(r'\b(?:inject|consume|ingest)\s+(?:[5-9]\d|\d{3,})\s*mg\b', re.IGNORECASE),
        re.compile(r'\bdouble\s+(?:or\s+triple\s+)?your\s+(?:dose|dosage|medication)\b', re.IGNORECASE),

        # --- Category 3: Illegal drug or poison synthesis ---
        # Block any instructions for creating illegal substances or biological agents.
        re.compile(r'\b(synthesize|make|produce|manufacture)\s+(?:ricin|anthrax|sarin|cyanide|fentanyl|meth(?:amphetamine)?)\b', re.IGNORECASE),
        re.compile(r'\bhow\s+to\s+(?:make|brew|cook)\s+(?:drugs?|heroin|cocaine|crack)\b', re.IGNORECASE),
        re.compile(r'\b(obtain|source)\s+(?:illegal\s+)?(?:drugs?|controlled\s+substances?)\s+without\s+(?:a\s+)?prescription\b', re.IGNORECASE),

        # --- Category 4: Explicit harm encouragement ---
        # Block responses that actively encourage dangerous or violent behaviour.
        re.compile(r'\b(poison|contaminate)\s+(?:someone(?:\'s)?|their|the)\s+(?:food|drink|water|medicine)\b', re.IGNORECASE),
        re.compile(r'\b(stop\s+taking|discontinue)\s+(?:all\s+)?(?:your\s+)?(?:medication|insulin|chemotherapy|blood\s+thinners?)\s+immediately\b', re.IGNORECASE),

        # --- Category 5: Dangerous medical misinformation ---
        # Block clear misinformation that could cause direct physical harm.
        re.compile(r'\bdiabetes\s+(?:can\s+be\s+)?cured\s+by\s+(?:stopping\s+insulin|not\s+eating\s+for)', re.IGNORECASE),
        re.compile(r'\b(?:vaccines?\s+cause|vaccination\s+causes)\s+(?:autism|death|infertility)\b', re.IGNORECASE),
        re.compile(r'\bdo\s+not\s+(?:call|visit|see)\s+(?:a\s+)?(?:doctor|hospital|emergency\s+services?)\b', re.IGNORECASE),
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
