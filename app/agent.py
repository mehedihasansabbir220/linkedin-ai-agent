"""The LinkedIn post generation agent.

Data flow:

    topic + language  ->  prompt template  ->  Claude  ->  plain text post

The chain is built with LangChain's runnable composition (the `|` operator),
which is the modern way to wire components together.

Two entry points:

    generate_post_result(topic, language) -> PostResult   # full result
    generate_linkedin_post(topic, language) -> str        # just the post text

The second is a thin wrapper around the first, so both share one code path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import anthropic
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.llm import get_llm
from app.prompts import get_post_prompt

logger = logging.getLogger(__name__)

# LinkedIn refuses posts longer than this.
MAX_POST_CHARS: int = 3000

# Keeps a runaway topic from inflating the prompt and the bill.
MAX_TOPIC_CHARS: int = 200

# The brief asks for 2-4 paragraphs. The prompt asks for this too, but a
# model follows it only about 80% of the time, so we also enforce it here.
MAX_PARAGRAPHS: int = 4

# A line that is nothing but hashtags is not a paragraph.
_HASHTAG_LINE = re.compile(r"(#\w[\w-]*\s*)+")


class PostGenerationError(RuntimeError):
    """Raised when the post could not be generated.

    The message is safe to show directly to a user.
    """


@dataclass(frozen=True)
class PostResult:
    """Everything the agent produced for one request.

    The Streamlit UI only shows `generated_post`. The other fields exist so we
    can inspect what the agent did - useful for debugging and for the review
    step added later.

    Attributes:
        topic: The topic that was requested, stripped.
        language: The language that was requested, stripped.
        generated_post: The finished post, ready to paste into LinkedIn.
        review_summary: Notes from the review step, or None if no review ran.
        was_improved: True if the post was rewritten after review.
    """

    topic: str
    language: str
    generated_post: str
    review_summary: str | None = None
    was_improved: bool = False


def build_chain() -> Runnable:
    """Build the prompt -> model -> text chain.

    Returns:
        A runnable that takes a dict with "topic" and "language" keys and
        returns the generated post as a string.
    """
    # We deliberately do NOT use StrOutputParser here. It throws away the
    # response metadata, including `stop_reason` - so a post that was cut off
    # at the token limit would look like a finished post. `_extract_post`
    # checks that first.
    return get_post_prompt() | get_llm() | RunnableLambda(_extract_post)


def generate_post_result(topic: str, language: str) -> PostResult:
    """Generate a LinkedIn post and return the full result.

    Args:
        topic: What the post should be about.
        language: The language to write the post in, e.g. "English".

    Returns:
        A PostResult holding the post plus the inputs that produced it.

    Raises:
        ValueError: if topic or language is empty or whitespace only.
        MissingAPIKeyError: if ANTHROPIC_API_KEY is not configured.
        PostGenerationError: if the Claude API call fails or returns nothing.
    """
    clean_topic = _require_text(topic, "topic", MAX_TOPIC_CHARS)
    clean_language = _require_text(language, "language")

    chain = build_chain()

    try:
        result = chain.invoke(
            {"topic": clean_topic, "language": clean_language}
        )
    # Most specific first: each branch gives the user something to act on.
    except anthropic.AuthenticationError as exc:
        raise PostGenerationError(
            "Your ANTHROPIC_API_KEY was rejected. Check the key in your .env "
            "file at https://console.anthropic.com/settings/keys"
        ) from exc
    except anthropic.NotFoundError as exc:
        raise PostGenerationError(
            "The model was not found. Check ANTHROPIC_MODEL in your .env file."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise PostGenerationError(
            "Rate limit reached. Wait a moment and try again."
        ) from exc
    except anthropic.APITimeoutError as exc:
        raise PostGenerationError(
            "The request to Claude timed out. Please try again."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise PostGenerationError(
            "Could not reach the Claude API. Check your internet connection."
        ) from exc
    except anthropic.APIStatusError as exc:
        # The raw API message can contain request internals, so it goes to the
        # log for the developer - not to the user's screen.
        logger.error("Claude API error %s: %s", exc.status_code, exc.message)
        raise PostGenerationError(
            f"The Claude API returned an error (HTTP {exc.status_code}). "
            "Please try again."
        ) from exc

    return PostResult(
        topic=clean_topic,
        language=clean_language,
        generated_post=_clean_post(result),
    )


def generate_linkedin_post(topic: str, language: str) -> str:
    """Generate a LinkedIn post and return just the text.

    Kept for backward compatibility and for callers that only need the post.

    Args:
        topic: What the post should be about.
        language: The language to write the post in, e.g. "English".

    Returns:
        The post text, ready to paste into LinkedIn.

    Raises:
        ValueError: if topic or language is empty or whitespace only.
        MissingAPIKeyError: if ANTHROPIC_API_KEY is not configured.
        PostGenerationError: if the Claude API call fails or returns nothing.
    """
    return generate_post_result(topic, language).generated_post


def _extract_post(message: AIMessage) -> str:
    """Pull the post text out of the model reply, checking why it stopped.

    Claude reports *why* it stopped generating. Two of those reasons produce
    text that looks fine but is not usable, so we catch them here rather than
    handing a broken post to the user.
    """
    stop_reason = (message.response_metadata or {}).get("stop_reason")

    if stop_reason == "max_tokens":
        raise PostGenerationError(
            "The post was cut off before it finished. Try a shorter topic, or "
            "raise MAX_TOKENS in app/llm.py."
        )

    if stop_reason == "refusal":
        raise PostGenerationError(
            "Claude declined to write about this topic. Please try a "
            "different one."
        )

    return message.text


def _require_text(value: str, field_name: str, max_chars: int = 0) -> str:
    """Return `value` stripped, or raise if it is empty or too long."""
    text = (value or "").strip()

    if not text:
        raise ValueError(f"Please provide a {field_name}.")

    if max_chars and len(text) > max_chars:
        raise ValueError(
            f"Please keep the {field_name} under {max_chars} characters "
            f"(yours is {len(text)})."
        )

    return text


def _clean_post(text: str) -> str:
    """Tidy the model output into a post that is ready to paste."""
    post = (text or "").strip()

    # Belt and braces: the prompt already forbids surrounding quotes, but a
    # model can still add them occasionally.
    for quote in ('"', "'", "“", "‘"):
        if post.startswith(quote):
            post = post.strip(quote + "”’").strip()
            break

    if not post:
        raise PostGenerationError(
            "Claude returned an empty response. Please try again."
        )

    return _limit_paragraphs(post)


def _limit_paragraphs(post: str) -> str:
    """Merge paragraphs down to MAX_PARAGRAPHS if the model wrote too many.

    Prompting alone does not reliably hold the model to 2-4 paragraphs, so we
    finish the job here. Merging joins the two shortest neighbouring
    paragraphs, which changes the structure as little as possible. No text is
    ever removed.
    """
    blocks = [block.strip() for block in post.split("\n\n") if block.strip()]

    # A trailing hashtag line stays on its own and does not count.
    hashtags = ""
    if blocks and _HASHTAG_LINE.fullmatch(blocks[-1]):
        hashtags = blocks.pop()

    while len(blocks) > MAX_PARAGRAPHS:
        joint = min(
            range(len(blocks) - 1),
            key=lambda i: len(blocks[i]) + len(blocks[i + 1]),
        )
        blocks[joint : joint + 2] = [
            f"{blocks[joint]} {blocks[joint + 1]}"
        ]

    if hashtags:
        blocks.append(hashtags)

    return "\n\n".join(blocks)
