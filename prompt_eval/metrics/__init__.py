"""Pluggable scoring metrics. Each is (output, expected) -> float in [0, 1]."""

from .contains import contains
from .exact_match import exact_match

# Names available via CLI --metric flag
REGISTRY = {
    "exact_match": exact_match,
    "contains": contains,
}
