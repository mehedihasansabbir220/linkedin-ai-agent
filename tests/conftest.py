"""Shared pytest configuration.

Integration tests call the real Claude API, which is slow and costs money, so
they are skipped unless you explicitly ask for them:

    pytest                      # unit tests only (offline, free)
    pytest --run-integration    # also run the real API tests
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that make real Claude API calls (needs a valid key)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: makes a real Claude API call (slow, costs money)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests unless --run-integration was passed."""
    if config.getoption("--run-integration"):
        return

    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
