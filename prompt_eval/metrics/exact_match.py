from ._text import normalise


def exact_match(output: str, expected: str) -> float:
    """1.0 if output equals expected after trimming, Unicode NFC normalisation and case-folding."""
    return 1.0 if normalise(output) == normalise(expected) else 0.0
