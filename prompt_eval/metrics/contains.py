def contains(output: str, expected: str) -> float:
    """1.0 if expected substring is in output (case-insensitive); else 0.0.

    Useful when the model returns a sentence and you just want the answer
    keyword somewhere in it ("the invoice number is 12345" → 1.0 for "12345").
    """
    return 1.0 if expected.strip().lower() in output.strip().lower() else 0.0
