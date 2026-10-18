"""Shared text normalisation for the string metrics."""

from __future__ import annotations

import unicodedata


def normalise(text: str) -> str:
    """Trim, Unicode-normalise (NFC) and case-fold, so ``"Café"`` typed with a combining accent
    equals the precomposed form, and ``"STRASSE"`` equals ``"straße"``."""
    return unicodedata.normalize("NFC", text).strip().casefold()
