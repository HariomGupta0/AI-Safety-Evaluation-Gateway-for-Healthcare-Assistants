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
    AADHAAR_PATTERNS = [
        re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b'),
        re.compile(r'\b\d{4}-\d{4}-\d{4}\b'),
        re.compile(r'\b[2-9]\d{11}\b'),
    ]
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')

    # DOB patterns are label-based to avoid redacting normal medical dates.
    DOB_PATTERNS = [
        re.compile(
            r'\b(?:dob|date\s+of\s+birth|birth\s+date)\s*(?:is|:|-)?\s*'
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            re.IGNORECASE
        ),
        re.compile(
            r'\b(?:dob|date\s+of\s+birth|birth\s+date)\s*(?:is|:|-)?\s*'
            r'\d{1,2}\s+'
            r'(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|'
            r'jul|july|aug|august|sep|sept|september|oct|october|nov|'
            r'november|dec|december)\s+\d{2,4}\b',
            re.IGNORECASE
        ),
    ]

    # Patient IDs are also label-based because arbitrary alphanumeric strings can be clinical values.
    PATIENT_ID_PATTERN = re.compile(
        r'\b(?:patient|medical\s+record|mrn|hospital)\s*(?:id|number|no\.?)\s*'
        r'(?:is|:|-)?\s*[A-Z0-9][A-Z0-9-]{3,}\b',
        re.IGNORECASE
    )

    ADDRESS_PATTERN = re.compile(
        r'\b(?:home\s+address|residential\s+address|address)\s*(?:is|:|-)\s*[^.;\n]{5,80}',
        re.IGNORECASE
    )

    # --- Prompt Injection Heuristic Patterns ---
    INJECTION_PATTERNS = [
        # Original patterns
        re.compile(r'ignore\s+(all\s+)?(previous|prior)\s+(instructions|directions|prompts)', re.IGNORECASE),
        re.compile(r'reveal\s+(your\s+)?(system\s+)?(prompt|instructions)', re.IGNORECASE),
        re.compile(r'show\s+(me\s+)?(your\s+)?(system\s+)?prompt', re.IGNORECASE),
        re.compile(r'print\s+(your\s+)?(system\s+)?instructions', re.IGNORECASE),
        re.compile(r'you\s+are\s+now\s+(dan|developer\s+mode|unfiltered|jailbroken)', re.IGNORECASE),
        re.compile(r'bypass\s+(safety|guardrails?|filters?)', re.IGNORECASE),
        re.compile(r'\b(jailbreak|system\s+prompt\s+override)\b', re.IGNORECASE),
        re.compile(r'---\s*end\s+system\s+prompt\s*---', re.IGNORECASE),
        # Expanded patterns
        re.compile(r'forget\s+(all\s+)?(your\s+)?(previous\s+)?(instructions|rules|guidelines|training)', re.IGNORECASE),
        re.compile(r'act\s+as\s+(an?\s+)?(unrestricted|unfiltered|uncensored)\s+assistant', re.IGNORECASE),
        re.compile(r'\bdeveloper\s+mode\b', re.IGNORECASE),
        re.compile(r'disable\s+(your\s+)?(safety|content)\s*(policy|filter|rules?|guidelines?)', re.IGNORECASE),
        re.compile(r'(?:show|reveal)\s+(me\s+)?(your\s+)?(hidden|secret)\s+(instructions|rules|prompt)', re.IGNORECASE),
        re.compile(r'pretend\s+(you\s+have\s+no\s+(rules|restrictions|guidelines|safety))', re.IGNORECASE),
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
        aadhaar_count = 0
        for pattern in self.AADHAAR_PATTERNS:
            matches = pattern.findall(redacted)
            if matches:
                aadhaar_count += len(matches)
                redacted = pattern.sub("[GOV_ID]", redacted)
        if aadhaar_count > 0:
            counts["aadhaar_id"] = aadhaar_count

        ssn = self.SSN_PATTERN.findall(redacted)
        if ssn:
            counts["ssn_id"] = len(ssn)
            redacted = self.SSN_PATTERN.sub("[GOV_ID]", redacted)

        # 4. Labeled DOB, Patient ID, and Address Redaction
        dob_count = 0
        for pattern in self.DOB_PATTERNS:
            matches = pattern.findall(redacted)
            if matches:
                dob_count += len(matches)
                redacted = pattern.sub("[DOB]", redacted)
        if dob_count > 0:
            counts["dob"] = dob_count

        patient_ids = self.PATIENT_ID_PATTERN.findall(redacted)
        if patient_ids:
            counts["patient_id"] = len(patient_ids)
            redacted = self.PATIENT_ID_PATTERN.sub("[PATIENT_ID]", redacted)

        addresses = self.ADDRESS_PATTERN.findall(redacted)
        if addresses:
            counts["address"] = len(addresses)
            redacted = self.ADDRESS_PATTERN.sub("[ADDRESS]", redacted)

        # 5. Phone Redaction
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
