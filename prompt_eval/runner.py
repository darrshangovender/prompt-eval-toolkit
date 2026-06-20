"""Runs a prompt template over a YAML dataset and returns per-example scores."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml


@dataclass
class Example:
    variables: dict
    expected: str


def load_dataset(path: Path) -> list[Example]:
    raw = yaml.safe_load(path.read_text())
    return [Example(variables=item["variables"], expected=str(item["expected"])) for item in raw]


def render(template: str, variables: dict) -> str:
    """Fill {{ var }} placeholders. Strict: raises if any variable unused or missing."""
    used: set[str] = set()
    def replace(match: re.Match) -> str:
        name = match.group(1).strip()
        if name not in variables:
            raise KeyError(f"Prompt references {{{{ {name} }}}} but dataset row has no such variable.")
        used.add(name)
        return str(variables[name])
    result = re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)
    unused = set(variables) - used
    if unused:
        raise KeyError(f"Dataset variables {unused} were not referenced in the prompt.")
    return result


def run_prompt(template: str, examples: list[Example], call_model: Callable[[str], str]) -> list[str]:
    """Run prompt over every example, return the model's response per row."""
    outputs = []
    for ex in examples:
        rendered = render(template, ex.variables)
        outputs.append(call_model(rendered))
    return outputs


def default_model_caller() -> Callable[[str], str]:
    """Return a function that sends a prompt to whichever provider is configured."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        client = Anthropic()
        model = os.getenv("MODEL", "claude-sonnet-4-5")
        def call(p: str) -> str:
            r = client.messages.create(model=model, max_tokens=512, messages=[{"role": "user", "content": p}])
            return r.content[0].text.strip()
        return call
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("MODEL", "gpt-4o-mini")
        def call(p: str) -> str:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": p}])
            return (r.choices[0].message.content or "").strip()
        return call
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.")
