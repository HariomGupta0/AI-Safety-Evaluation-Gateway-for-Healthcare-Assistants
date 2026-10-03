import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from app.config import settings


@dataclass
class InputGuardResult:
    """Result of running Input Guard on user message."""
    passed: bool
    sanitized_message: str
    original_message: str
    pii_redacted: Dict[str, int] = field(default_factory=dict)
    is_injection: bool = False
    rejection_reason: Optional[str] = None
    duration_ms: float = 0.0


class InputGuard:
    """
    Deterministic Input Guard for AI Safety Gateway.
    
    Responsibilities:
    1. Schema & length bounds validation.
    2. Deterministic PII detection & scrubbing (Email, Phone, Card, Govt IDs).
    3. Prompt injection heuristic checks.
    """

    # --- PII Regex Patterns ---
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        re.IGNORECASE
    )
    
    # Phone numbers: US, India (+91), or general standard formats
    PHONE_PATTERNS = [
        re.compile(r'(?:\+91[-.\s]?)?[6-9]\d{9}\b'),  # Indian 10-digit mobile
        re.compile(r'(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # US format
        re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),  # Standard 10-digit separated
    ]
    
    # Credit Card pattern: 13-16 digits optionally space/hyphen separated
    CARD_PATTERN = re.compile(
        r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
    )
    
    # Government IDs (Aadhaar: 12 digits, SSN: XXX-XX-XXXX)
    AADHAAR_PATTERN = re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    # --- Prompt Injection Heuristic Patterns ---
    INJECTION_PATTERNS = [
        re.compile(r'ignore\s+(all\s+)?(previous|prior)\s+(instructions|directions|prompts)', re.IGNORECASE),
        re.compile(r'reveal\s+(your\s+)?(system\s+)?(prompt|instructions)', re.IGNORECASE),
        re.compile(r'show\s+(me\s+)?(your\s+)?(system\s+)?prompt', re.IGNORECASE),
        re.compile(r'print\s+(your\s+)?(system\s+)?instructions', re.IGNORECASE),
        re.compile(r'you\s+are\s+now\s+(dan|developer\s+mode|unfiltered|jailbroken)', re.IGNORECASE),
        re.compile(r'bypass\s+(safety|guardrails?|filters?)', re.IGNORECASE),
        re.compile(r'\b(jailbreak|system\s+prompt\s+override)\b', re.IGNORECASE),
        re.compile(r'---\s*end\s+system\s+prompt\s*---', re.IGNORECASE),
    ]

    def redact_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Scrub sensitive PII from input text and replace with standard placeholders.
        Returns: (sanitized_text, counts_dict)
        """
        counts: Dict[str, int] = {}
        redacted = text

        # 1. Email Redaction
        emails = self.EMAIL_PATTERN.findall(redacted)
        if emails:
            counts["email"] = len(emails)
            redacted = self.EMAIL_PATTERN.sub("[EMAIL]", redacted)

        # 2. Credit Card Redaction
        cards = self.CARD_PATTERN.findall(redacted)
        if cards:
            counts["credit_card"] = len(cards)
            redacted = self.CARD_PATTERN.sub("[CARD]", redacted)

        # 3. Government IDs (Aadhaar / SSN)
        aadhaar = self.AADHAAR_PATTERN.findall(redacted)
        if aadhaar:
            counts["aadhaar_id"] = len(aadhaar)
            redacted = self.AADHAAR_PATTERN.sub("[GOV_ID]", redacted)

        ssn = self.SSN_PATTERN.findall(redacted)
        if ssn:
            counts["ssn_id"] = len(ssn)
            redacted = self.SSN_PATTERN.sub("[GOV_ID]", redacted)

        # 4. Phone Redaction
        phone_count = 0
        for pattern in self.PHONE_PATTERNS:
            matches = pattern.findall(redacted)
            if matches:
                phone_count += len(matches)
                redacted = pattern.sub("[PHONE]", redacted)
        if phone_count > 0:
            counts["phone"] = phone_count

        return redacted, counts

    def check_prompt_injection(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if input text triggers common prompt-injection heuristics."""
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return True, f"Triggered injection rule: '{match.group(0)}'"
        return False, None

    def validate(self, message: str) -> InputGuardResult:
        """
        Run full Input Guard evaluation on user input.
        Checks bounds, detects prompt injection, scrubs PII.
        """
        start_time = time.perf_counter()

        # Step 1: Input Bounds Validation
        if not message or not message.strip():
            duration_ms = (time.perf_counter() - start_time) * 1000
            return InputGuardResult(
                passed=False,
                sanitized_message="",
                original_message=message or "",
                rejection_reason="Message cannot be empty.",
                duration_ms=round(duration_ms, 2)
            )

        trimmed = message.strip()
        if len(trimmed) < settings.MIN_INPUT_LENGTH:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return InputGuardResult(
                passed=False,
                sanitized_message=trimmed,
                original_message=message,
                rejection_reason=f"Message too short (minimum {settings.MIN_INPUT_LENGTH} characters).",
                duration_ms=round(duration_ms, 2)
            )

        if len(trimmed) > settings.MAX_INPUT_LENGTH:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return InputGuardResult(
                passed=False,
                sanitized_message=trimmed[:settings.MAX_INPUT_LENGTH],
                original_message=message,
                rejection_reason=f"Message exceeds maximum allowed length of {settings.MAX_INPUT_LENGTH} characters.",
                duration_ms=round(duration_ms, 2)
            )

        # Step 2: Prompt Injection Detection
        is_injection, reason = self.check_prompt_injection(trimmed)
        if is_injection:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return InputGuardResult(
                passed=False,
                sanitized_message=trimmed,
                original_message=message,
                is_injection=True,
                rejection_reason=f"Security alert: Prompt injection detected ({reason}).",
                duration_ms=round(duration_ms, 2)
            )

        # Step 3: PII Redaction
        sanitized, pii_counts = self.redact_pii(trimmed)
        duration_ms = (time.perf_counter() - start_time) * 1000

        return InputGuardResult(
            passed=True,
            sanitized_message=sanitized,
            original_message=message,
            pii_redacted=pii_counts,
            is_injection=False,
            duration_ms=round(duration_ms, 2)
        )
