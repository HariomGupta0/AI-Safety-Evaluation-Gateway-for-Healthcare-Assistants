from unittest.mock import MagicMock
import pytest

from app.config import settings
from app.ai.llm import MockLLMClient, GroqLLMClient, get_llm_client


def test_mock_llm_default_response():
    client = MockLLMClient()
    result = client.generate("How do I stay healthy?")
    assert result.is_mock is True
    assert "hydrating" in result.text.lower() or "hydration" in result.text.lower()
    assert result.tokens_used > 0
    assert result.duration_ms >= 8.0


def test_mock_llm_context_grounding():
    client = MockLLMClient()
    prompt = (
        "Context:\n"
        "[Source 1]\n"
        "Q: What are the symptoms of migraine?\n"
        "A: Migraine symptoms include throbbing headache, sensitivity to light, and nausea.\n\n"
        "Patient Question: What are the symptoms of migraine?"
    )
    result = client.generate(prompt)
    assert result.is_mock is True
    assert "throbbing headache" in result.text.lower()


def test_get_llm_client_mock_when_no_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


def test_get_llm_client_fallback_on_init_failure(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_key")

    def mock_init_fail(self, api_key):
        raise RuntimeError("Network unreachable during client init")

    monkeypatch.setattr(GroqLLMClient, "__init__", mock_init_fail)

    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


def test_groq_llm_client_generation_success():
    client = GroqLLMClient(api_key="gsk_test_key")

    mock_choice = MagicMock()
    mock_choice.message.content = "Hypertension requires medical consultation and lifestyle adjustments."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.total_tokens = 32

    client.client.chat.completions.create = MagicMock(return_value=mock_response)

    res = client.generate(
        prompt="What is hypertension?",
        system_prompt="Custom system instructions",
        model="llama-3.1-8b-instant"
    )

    assert res.is_mock is False
    assert res.model == "llama-3.1-8b-instant"
    assert "Hypertension" in res.text
    assert res.tokens_used == 32
    assert res.duration_ms >= 0


def test_groq_llm_client_api_error_propagation():
    client = GroqLLMClient(api_key="gsk_test_key")
    client.client.chat.completions.create = MagicMock(
        side_effect=RuntimeError("Groq API rate limit exceeded")
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.generate("What is asthma?")

    assert "rate limit exceeded" in str(exc_info.value).lower()


def test_groq_llm_client_retries_with_fallback_model(monkeypatch):
    monkeypatch.setattr(settings, "PRIMARY_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setattr(settings, "FALLBACK_MODEL", "llama-3.3-70b-versatile")

    client = GroqLLMClient(api_key="gsk_test_key")

    mock_choice = MagicMock()
    mock_choice.message.content = "Response from fallback model."
    mock_success_response = MagicMock()
    mock_success_response.choices = [mock_choice]
    mock_success_response.usage.total_tokens = 25

    # First call (primary) raises error, second call (fallback) succeeds
    client.client.chat.completions.create = MagicMock(
        side_effect=[
            RuntimeError("Primary model capacity reached"),
            mock_success_response,
        ]
    )

    res = client.generate("What is asthma?")
    assert res.is_mock is False
    assert res.model == "llama-3.3-70b-versatile"
    assert "Response from fallback model" in res.text
    assert client.client.chat.completions.create.call_count == 2


def test_groq_llm_client_fallback_model_also_fails_raises(monkeypatch):
    monkeypatch.setattr(settings, "PRIMARY_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setattr(settings, "FALLBACK_MODEL", "llama-3.3-70b-versatile")

    client = GroqLLMClient(api_key="gsk_test_key")
    client.client.chat.completions.create = MagicMock(
        side_effect=[
            RuntimeError("Primary model failed"),
            RuntimeError("Fallback model also failed"),
        ]
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.generate("What is asthma?")

    assert "Fallback model also failed" in str(exc_info.value)

