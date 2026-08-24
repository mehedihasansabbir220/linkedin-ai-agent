"""LLM factory.

This is the single place where the Claude chat model is created. Everything
else in the app (the agent, the Streamlit UI, the tests) calls `get_llm()` and
does not need to know how the model is configured.

Environment variables are read through `app.config`, which loads the local
.env file with python-dotenv. No API key ever appears in this file.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from app.config import get_anthropic_api_key, get_effort, get_model

# ---------------------------------------------------------------------------
# Model configuration
#
# The model name and effort level come from .env (see .env.example) so they can
# be changed without touching code. The settings below are LinkedIn-post
# specific, so they live here rather than in the environment.
# ---------------------------------------------------------------------------

# Caps the response so requests stay fast (langchain-anthropic would
# otherwise default to 128,000 tokens).
#
# This budget covers thinking tokens AND the post itself. Non-Latin scripts
# cost far more tokens for the same text - a Bengali post measured 1,286
# output tokens where the English equivalent used 494 - so leave real
# headroom here or long posts get cut off mid-sentence.
MAX_TOKENS: int = 4000

# Seconds to wait for the API before giving up.
TIMEOUT: float = 60.0

# Retries on rate limits and transient server errors, handled by the SDK.
MAX_RETRIES: int = 2


def get_llm() -> ChatAnthropic:
    """Create the configured Claude chat model.

    Reads ANTHROPIC_API_KEY, ANTHROPIC_MODEL, and ANTHROPIC_EFFORT from the
    environment via `app.config`.

    Returns:
        A ready-to-use ChatAnthropic model.

    Raises:
        MissingAPIKeyError: if ANTHROPIC_API_KEY is not configured. The error
            message explains how to create the .env file.
    """
    return ChatAnthropic(
        model=get_model(),
        api_key=get_anthropic_api_key(),
        max_tokens=MAX_TOKENS,
        # `reasoning_effort` is Claude's replacement for `temperature`, which
        # Claude Opus 5 rejects. It sets how much the model thinks before
        # answering: low | medium | high | xhigh | max.
        reasoning_effort=get_effort(),
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
    )
