def exact_match(output: str, expected: str) -> float:
    """1.0 if output (trimmed, lowered) equals expected; else 0.0."""
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0
