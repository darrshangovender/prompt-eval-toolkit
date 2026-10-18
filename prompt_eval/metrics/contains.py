from ._text import normalise


def contains(output: str, expected: str) -> float:
    """1.0 if expected is a substring of output (case-insensitive, NFC-normalised); else 0.0.

    Useful when the model returns a sentence and you just want the answer
    keyword somewhere in it ("the invoice number is 12345" → 1.0 for "12345").
    An empty `expected` is a substring of everything, so it scores 1.0 for every output.
    """
    return 1.0 if normalise(expected) in normalise(output) else 0.0
