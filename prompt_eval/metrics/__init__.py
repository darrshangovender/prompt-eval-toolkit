"""Pluggable scoring metrics. Each is (output, expected) -> float in [0, 1]."""

from .exact_match import exact_match
from .contains import contains

# Names available via CLI --metric flag
REGISTRY = {
    "exact_match": exact_match,
    "contains": contains,
}
