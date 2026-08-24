"""Unit tests for app.agent.

These tests never call the real Claude API and never need an API key. The
model is replaced with a scripted fake, so the suite runs offline, fast, and
free.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import anthropic
import httpx2
import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

import app.agent as agent
from app.agent import (
    PostGenerationError,
    PostResult,
    build_chain,
    generate_linkedin_post,
    generate_post_result,
)
from app.config import MissingAPIKeyError, get_anthropic_api_key

SAMPLE_POST = (
    "Most onboarding fails in the first week.\n\n"
    "Here is what actually helps.\n\n"
    "What worked for you?"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the Claude model with a fake returning a fixed response.

    Usage:  fake_llm("some post text")
    """

    def install(response: str) -> None:
        monkeypatch.setattr(
            agent, "get_llm", lambda: FakeListChatModel(responses=[response])
        )

    return install


@pytest.fixture
def failing_chain(monkeypatch):
    """Replace the chain with one that raises the given exception.

    Usage:  failing_chain(anthropic.RateLimitError(...))
    """

    def install(exc: Exception) -> None:
        def boom(_input: object) -> str:
            raise exc

        monkeypatch.setattr(agent, "build_chain", lambda: RunnableLambda(boom))

    return install


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=_request())


# ---------------------------------------------------------------------------
# 1. Empty topic validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topic", ["", "   ", "\n\t", None])
def test_empty_topic_is_rejected(topic):
    with pytest.raises(ValueError, match="topic"):
        generate_linkedin_post(topic, "English")


# ---------------------------------------------------------------------------
# 2. Empty language validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", ["", "   ", "\n\t", None])
def test_empty_language_is_rejected(language):
    with pytest.raises(ValueError, match="language"):
        generate_linkedin_post("Remote work", language)


def test_validation_runs_before_the_model_is_built(monkeypatch):
    """Invalid input must fail before any model or network call happens."""

    def explode() -> None:
        raise AssertionError("get_llm() must not be called for invalid input")

    monkeypatch.setattr(agent, "get_llm", explode)

    with pytest.raises(ValueError):
        generate_linkedin_post("", "English")


# ---------------------------------------------------------------------------
# 3. Basic function behavior
# ---------------------------------------------------------------------------


def test_returns_the_generated_post(fake_llm):
    fake_llm(SAMPLE_POST)

    result = generate_linkedin_post("Remote onboarding", "English")

    assert isinstance(result, str)
    assert result == SAMPLE_POST


def test_topic_and_language_reach_the_model(monkeypatch):
    """The user's topic and language must arrive in the prompt sent out."""
    seen: dict[str, str] = {}

    def recording_model(prompt_value) -> AIMessage:
        # The last message is the human turn holding topic + language.
        seen["text"] = str(prompt_value.to_messages()[-1].content)
        return AIMessage(content=SAMPLE_POST)

    monkeypatch.setattr(
        agent, "get_llm", lambda: RunnableLambda(recording_model)
    )

    result = generate_linkedin_post("  Remote onboarding  ", "  German  ")

    assert result == SAMPLE_POST
    # Stripped, not padded - proves the inputs were cleaned before use.
    assert "Topic: Remote onboarding\n" in seen["text"]
    assert "Language: German\n" in seen["text"]


def test_whitespace_is_stripped_from_the_output(fake_llm):
    fake_llm(f"\n\n  {SAMPLE_POST}  \n\n")

    result = generate_linkedin_post("Remote onboarding", "English")

    assert result == SAMPLE_POST


def test_surrounding_quotes_are_removed(fake_llm):
    fake_llm(f'"{SAMPLE_POST}"')

    result = generate_linkedin_post("Remote onboarding", "English")

    assert not result.startswith('"')
    assert not result.endswith('"')
    assert "Most onboarding fails" in result


def test_build_chain_returns_a_runnable(fake_llm):
    fake_llm(SAMPLE_POST)

    chain = build_chain()

    assert isinstance(chain, Runnable)
    assert chain.invoke({"topic": "AI", "language": "English"}) == SAMPLE_POST


# ---------------------------------------------------------------------------
# 4. Error handling
# ---------------------------------------------------------------------------


def test_empty_model_response_raises(fake_llm):
    fake_llm("   \n  ")

    with pytest.raises(PostGenerationError, match="empty"):
        generate_linkedin_post("Remote onboarding", "English")


def test_rejected_api_key_is_reported(failing_chain):
    failing_chain(
        anthropic.AuthenticationError(
            "invalid x-api-key", response=_response(401), body=None
        )
    )

    with pytest.raises(PostGenerationError, match="ANTHROPIC_API_KEY"):
        generate_linkedin_post("Remote onboarding", "English")


def test_unknown_model_is_reported(failing_chain):
    failing_chain(
        anthropic.NotFoundError(
            "model not found", response=_response(404), body=None
        )
    )

    with pytest.raises(PostGenerationError, match="ANTHROPIC_MODEL"):
        generate_linkedin_post("Remote onboarding", "English")


def test_rate_limit_is_reported(failing_chain):
    failing_chain(
        anthropic.RateLimitError(
            "slow down", response=_response(429), body=None
        )
    )

    with pytest.raises(PostGenerationError, match="Rate limit"):
        generate_linkedin_post("Remote onboarding", "English")


def test_timeout_is_reported(failing_chain):
    failing_chain(anthropic.APITimeoutError(request=_request()))

    with pytest.raises(PostGenerationError, match="timed out"):
        generate_linkedin_post("Remote onboarding", "English")


def test_connection_failure_is_reported(failing_chain):
    failing_chain(anthropic.APIConnectionError(request=_request()))

    with pytest.raises(PostGenerationError, match="internet connection"):
        generate_linkedin_post("Remote onboarding", "English")


def test_other_api_errors_are_reported_with_the_status_code(failing_chain):
    failing_chain(
        anthropic.APIStatusError(
            "overloaded", response=_response(529), body=None
        )
    )

    with pytest.raises(PostGenerationError, match="529"):
        generate_linkedin_post("Remote onboarding", "English")


def test_api_errors_keep_the_original_exception(failing_chain):
    """`raise ... from exc` must preserve the cause for debugging."""
    original = anthropic.RateLimitError(
        "slow down", response=_response(429), body=None
    )
    failing_chain(original)

    with pytest.raises(PostGenerationError) as info:
        generate_linkedin_post("Remote onboarding", "English")

    assert info.value.__cause__ is original


def test_missing_api_key_is_reported(monkeypatch):
    """No key configured must give a clear setup message, not a crash."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_anthropic_api_key.cache_clear()

    try:
        with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
            generate_linkedin_post("Remote onboarding", "English")
    finally:
        # Don't leak the cleared cache into other tests.
        get_anthropic_api_key.cache_clear()


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------


def test_result_carries_the_inputs_and_the_post(fake_llm):
    fake_llm(SAMPLE_POST)

    result = generate_post_result("  Remote onboarding  ", "  German  ")

    assert isinstance(result, PostResult)
    # Inputs are recorded stripped, exactly as they were sent to the model.
    assert result.topic == "Remote onboarding"
    assert result.language == "German"
    assert result.generated_post == SAMPLE_POST


def test_review_fields_default_to_not_reviewed(fake_llm):
    """No review step runs yet, so these must be empty rather than guessed."""
    fake_llm(SAMPLE_POST)

    result = generate_post_result("Remote onboarding", "English")

    assert result.review_summary is None
    assert result.was_improved is False


def test_result_is_immutable(fake_llm):
    fake_llm(SAMPLE_POST)

    result = generate_post_result("Remote onboarding", "English")

    with pytest.raises(FrozenInstanceError):
        result.generated_post = "something else"


def test_string_helper_matches_the_structured_result(fake_llm):
    """generate_linkedin_post() must stay a thin wrapper, not a second path."""
    fake_llm(SAMPLE_POST)
    text = generate_linkedin_post("Remote onboarding", "English")

    fake_llm(SAMPLE_POST)
    result = generate_post_result("Remote onboarding", "English")

    assert text == result.generated_post


def test_structured_entry_point_validates_inputs_too(fake_llm):
    fake_llm(SAMPLE_POST)

    with pytest.raises(ValueError, match="topic"):
        generate_post_result("", "English")
