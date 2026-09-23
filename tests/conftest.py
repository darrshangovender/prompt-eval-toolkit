"""Shared fixtures.

Everything here is hermetic: no network, no API keys, no provider SDKs. The
model is always a plain callable so the tests exercise our own logic rather
than a vendor's.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    """Path to the checkout, so tests can read the shipped demo assets."""
    return REPO_ROOT


@pytest.fixture
def dataset_path(repo_root: Path) -> Path:
    return repo_root / "data" / "invoices.yml"


@pytest.fixture
def echo_model() -> Callable[[str], str]:
    """A 'model' that returns the prompt it was given.

    Lets us assert on what was actually rendered and sent, which is the part
    of run_prompt worth testing.
    """
    return lambda prompt: prompt


@pytest.fixture(autouse=True)
def no_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee provider selection is never influenced by the dev's real env.

    Without this, a developer with ANTHROPIC_API_KEY exported would get
    different behaviour locally than CI does, which is exactly the class of
    flake this suite exists to prevent.
    """
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MODEL"):
        monkeypatch.delenv(var, raising=False)
