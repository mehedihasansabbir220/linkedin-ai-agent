"""Environment variable handling.

Loads configuration from the local .env file via python-dotenv and exposes it
to the rest of the application. This is the single place where environment
variables are read, so no other module ever touches os.environ directly and no
secret is ever written into source code.

The API key is read at call time and never logged, printed, or given a default.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

# Read .env into the process environment. Real environment variables already
# set by the shell or the host take precedence (override=False), which is what
# you want for deployment.
load_dotenv(override=False)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"

VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not configured."""


class ConfigurationError(ValueError):
    """Raised when an environment variable holds an invalid value.

    Subclasses ValueError so existing callers keep working, but lets the UI
    tell "you typed something wrong" apart from "your .env is wrong".
    """


@lru_cache(maxsize=1)
def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment.

    Raises:
        MissingAPIKeyError: if the key is unset, blank, or still the
            placeholder value shipped in .env.example.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not key or key == "your_api_key_here":
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key:\n    cp .env.example .env"
        )

    return key


def get_model() -> str:
    """Return the Claude model name, falling back to the default."""
    return os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL


def get_effort() -> str:
    """Return the reasoning effort level, falling back to the default.

    Claude Opus 5 controls token spend with `effort` rather than
    `temperature`, which the model rejects.
    """
    effort = os.getenv("ANTHROPIC_EFFORT", "").strip().lower() or DEFAULT_EFFORT

    if effort not in VALID_EFFORTS:
        raise ConfigurationError(
            f"ANTHROPIC_EFFORT must be one of {', '.join(VALID_EFFORTS)}, "
            f"got {effort!r}"
        )

    return effort
