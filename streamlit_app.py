"""Streamlit user interface for the LinkedIn AI Agent.

Collects a topic and a language from the user, calls the LangChain agent, and
displays the generated post.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import traceback

import streamlit as st

from app.agent import PostGenerationError, PostResult, generate_post_result
from app.agent import MAX_POST_CHARS, MAX_TOPIC_CHARS
from app.config import ConfigurationError, MissingAPIKeyError

LANGUAGES: list[str] = [
    "English",
    "Bengali",
    "Spanish",
    "French",
    "German",
]

st.set_page_config(
    page_title="LinkedIn AI Post Generator",
    page_icon="📝",
    layout="centered",
)


def main() -> None:
    st.title("LinkedIn AI Post Generator")
    st.caption(
        "Generate professional LinkedIn posts with a LangChain AI Agent."
    )

    topic = st.text_input(
        "Topic",
        max_chars=MAX_TOPIC_CHARS,
        placeholder="e.g. onboarding engineers on a remote team",
        help="What should the post be about?",
    )
    language = st.selectbox(
        "Language",
        LANGUAGES,
        help="The post will be written natively in this language.",
    )

    if st.button(
        "Generate LinkedIn Post", type="primary", use_container_width=True
    ):
        _generate(topic or "", language or LANGUAGES[0])

    _show_result()
    _show_about()


def _generate(topic: str, language: str) -> None:
    """Validate the input, call the agent, and store the outcome."""
    # Clear whatever was on screen before.
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)

    if not topic.strip():
        st.session_state["error"] = "Please enter a topic first."
        return

    try:
        with st.spinner(f"Writing your post in {language}...", show_time=True):
            st.session_state["result"] = generate_post_result(topic, language)

    except ConfigurationError as exc:
        # A bad value in .env - not the user's fault, so say so.
        st.session_state["error"] = (
            f"Configuration problem in your .env file: {exc}"
        )

    except ValueError as exc:
        # Empty or over-long topic - the message is written for users.
        st.session_state["error"] = str(exc)

    except MissingAPIKeyError:
        st.session_state["error"] = (
            "The app is not configured yet. Copy `.env.example` to `.env` "
            "and add your Anthropic API key."
        )

    except PostGenerationError as exc:
        # Already a clean, user-facing message from the agent.
        st.session_state["error"] = str(exc)

    except Exception:  # noqa: BLE001 - last resort, never show a traceback
        # The full details go to the terminal for the developer. The user sees
        # a generic message: no stack trace, no API key, no internals.
        traceback.print_exc()
        st.session_state["error"] = (
            "Something went wrong while generating the post. "
            "Please try again."
        )


def _show_result() -> None:
    """Render the generated post, or the error message."""
    error = st.session_state.get("error")
    if error:
        st.error(error, icon="⚠️")
        return

    result: PostResult | None = st.session_state.get("result")
    if result is None:
        return

    st.divider()

    left, right = st.columns(2)
    left.markdown(f"**Topic**  \n{result.topic}")
    right.markdown(f"**Language**  \n{result.language}")

    st.markdown("### Your post")
    with st.container(border=True):
        st.markdown(_as_markdown(result.generated_post))

    length = len(result.generated_post)
    if length > MAX_POST_CHARS:
        st.warning(
            f"This post is {length:,} characters. LinkedIn allows "
            f"{MAX_POST_CHARS:,}, so you will need to trim it.",
            icon="✂️",
        )
    else:
        st.caption(f"{length:,} / {MAX_POST_CHARS:,} characters")

    with st.expander("Copy post text"):
        st.code(result.generated_post, language=None, wrap_lines=True)
        st.caption("Use the copy icon in the top-right of the box above.")


def _as_markdown(post: str) -> str:
    """Escape characters that Markdown would treat as formatting.

    A post often ends with hashtags, and a line starting with "#" would
    otherwise render as a huge heading.
    """
    safe_lines = []

    for line in post.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("#", ">")):
            marker = stripped[0]
            line = line.replace(marker, "\\" + marker, 1)
        safe_lines.append(line)

    return "\n".join(safe_lines)


def _show_about() -> None:
    st.divider()
    with st.expander("How this works"):
        st.markdown(
            """
            This app is a small **AI agent** built with
            [LangChain](https://python.langchain.com).

            When you click *Generate LinkedIn Post*:

            1. Your topic and language are inserted into a prompt template
               that describes what a good LinkedIn post looks like.
            2. That prompt is sent to a **large language model**
               (Anthropic's Claude) through LangChain.
            3. The model's reply is cleaned up and shown above.

            The chain is composed as `prompt | model | parser`, which is
            LangChain's runnable composition style.

            Posts are generated fresh each time, so the same topic will
            produce a different post on every run.
            """
        )


if __name__ == "__main__":
    main()
