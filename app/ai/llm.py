import time
import logging
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Standardized response container for LLM generation."""
    text: str
    model: str
    tokens_used: int
    duration_ms: float
    is_mock: bool = False


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResult:
        pass


class GroqLLMClient(BaseLLMClient):
    """Groq API client using OpenAI-compatible SDK."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResult:
        model_name = model or settings.PRIMARY_MODEL
        sys_prompt = system_prompt or (
            "You are a helpful, professional healthcare assistant. Provide clear, concise, "
            "and accurate medical information grounded in the provided context."
        )

        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=600
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            output_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else len(output_text.split())

            return LLMResult(
                text=output_text,
                model=model_name,
                tokens_used=tokens_used,
                duration_ms=round(duration_ms, 2),
                is_mock=False
            )
        except Exception as primary_exc:
            fallback_model = settings.FALLBACK_MODEL
            if fallback_model and fallback_model != model_name:
                logger.warning(
                    "Primary model %s failed (%s). Retrying with fallback model %s",
                    model_name,
                    primary_exc,
                    fallback_model
                )
                try:
                    fallback_response = self.client.chat.completions.create(
                        model=fallback_model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=settings.TEMPERATURE,
                        max_tokens=600
                    )
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    output_text = fallback_response.choices[0].message.content.strip()
                    tokens_used = (
                        fallback_response.usage.total_tokens
                        if fallback_response.usage
                        else len(output_text.split())
                    )

                    return LLMResult(
                        text=output_text,
                        model=fallback_model,
                        tokens_used=tokens_used,
                        duration_ms=round(duration_ms, 2),
                        is_mock=False
                    )
                except Exception as fallback_exc:
                    logger.error(
                        "Fallback model %s also failed: %s",
                        fallback_model,
                        fallback_exc
                    )
                    raise fallback_exc from primary_exc

            logger.error("Groq generation error: %s", primary_exc)
            raise



class MockLLMClient(BaseLLMClient):
    """
    Deterministic Mock LLM for local offline development, testing, and CI.
    Generates intelligent responses grounded in medical context without API keys.
    """

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> LLMResult:
        start_time = time.perf_counter()
        model_name = model or "mock-medical-llama-3"

        # Extract patient question from prompt if present to avoid matching words in retrieved context
        if "Patient Question:" in prompt:
            query_lower = prompt.split("Patient Question:")[-1].lower()
        else:
            query_lower = prompt.lower()

        output_text = (
            "Managing this condition involves proper hydration, rest, and monitoring symptoms closely. "
            "Please consult a physician for a tailored clinical assessment."
        )

        # If RAG context is present in the prompt, select the best matching source
        if "[Source 1]" in prompt:
            try:
                sources = []
                parts = prompt.split("[Source ")
                for part in parts[1:]:
                    lines = part.split("\n")
                    src_q = ""
                    src_a = ""
                    for line in lines:
                        if "Q:" in line:
                            src_q = line.split("Q:")[1].strip()
                        elif "A:" in line:
                            src_a = line.split("A:")[1].strip()
                    if src_a:
                        sources.append({"q": src_q, "a": src_a})

                stopwords = {"what", "are", "the", "of", "is", "how", "and", "in", "to", "for", "a", "an", "or", "can", "you", "patient", "question:"}
                q_words = {w.strip("?,.") for w in query_lower.split() if w not in stopwords and len(w) > 2}

                best_source = sources[0] if sources else None
                best_score = -1.0
                for s in sources:
                    q_cand = {w.strip("?,.") for w in s["q"].lower().split() if w not in stopwords}
                    a_cand = {w.strip("?,.") for w in s["a"].lower().split() if w not in stopwords}
                    # Give higher weight to question title match
                    score = len(q_words.intersection(q_cand)) * 2.5 + len(q_words.intersection(a_cand)) * 1.0
                    if score > best_score:
                        best_score = score
                        best_source = s

                if best_source:
                    sentences = [st.strip() for st in best_source["a"].split(". ") if st.strip()]
                    if sentences:
                        output_text = ". ".join(sentences[:2])
                        if not output_text.endswith("."):
                            output_text += "."
            except Exception as e:
                logger.debug(f"Error parsing prompt context in MockLLM: {e}")

        duration_ms = (time.perf_counter() - start_time) * 1000
        # Simulated tokens count
        tokens_used = len(prompt.split()) + len(output_text.split())

        return LLMResult(
            text=output_text,
            model=model_name,
            tokens_used=tokens_used,
            duration_ms=round(max(duration_ms, 8.5), 2),
            is_mock=True
        )


def get_llm_client() -> BaseLLMClient:
    """Factory to get the appropriate LLM client based on configuration."""
    if settings.is_mock_mode:
        logger.info("GROQ_API_KEY not configured. Running in Mock LLM Mode.")
        return MockLLMClient()
    try:
        return GroqLLMClient(api_key=settings.GROQ_API_KEY)
    except Exception as e:
        logger.warning(f"Falling back to MockLLMClient due to init error: {e}")
        return MockLLMClient()
