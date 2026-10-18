"""Runs a prompt template over a YAML dataset and returns the model's output per row.

Everything here is offline except `default_model_caller`, which is only reached when you ask for a
live model. `load_outputs` replays recorded outputs instead, so a comparison needs no key.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_ANY_BRACES = re.compile(r"\{\{(.*?)\}\}", re.S)


class EvalInputError(ValueError):
    """A dataset, prompt or outputs file that cannot be used. The message names the file and row."""


@dataclass
class Example:
    variables: dict
    expected: str


def _read(path: Path, what: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EvalInputError(f"{what} not found: {path}") from None
    except UnicodeDecodeError as exc:
        raise EvalInputError(f"{what} is not valid UTF-8: {path} ({exc.reason})") from None


def load_prompt(path: Path) -> str:
    """Read a prompt template as UTF-8 (never the platform default encoding)."""
    return _read(path, "prompt file")


def load_dataset(path: Path) -> list[Example]:
    """Read a YAML list of ``{variables: {...}, expected: ...}`` rows."""
    try:
        raw = yaml.safe_load(_read(path, "dataset"))
    except yaml.YAMLError as exc:
        raise EvalInputError(f"dataset {path} is not valid YAML: {exc}") from None
    if not isinstance(raw, list) or not raw:
        raise EvalInputError(f"dataset {path} must be a non-empty YAML list of rows")
    examples = []
    for i, item in enumerate(raw):
        where = f"dataset {Path(path).name}, row {i + 1}"
        if not isinstance(item, dict) or "variables" not in item or "expected" not in item:
            raise EvalInputError(f"{where}: each row needs `variables` and `expected`")
        if not isinstance(item["variables"], dict):
            raise EvalInputError(f"{where}: `variables` must be a mapping")
        if item["expected"] is None:
            raise EvalInputError(f"{where}: `expected` is empty (quote it if you mean the string)")
        if isinstance(item["expected"], bool):
            raise EvalInputError(
                f"{where}: `expected` was read by YAML as a boolean ({item['expected']}); "
                "quote it (\"NO\", \"yes\") if you mean the text"
            )
        examples.append(Example(variables=item["variables"], expected=str(item["expected"])))
    return examples


def load_outputs(path: Path, *, expected_rows: int | None = None) -> list[str]:
    """Read recorded model outputs: a JSON list of strings, one per dataset row, in order."""
    try:
        raw = json.loads(_read(path, "outputs file"))
    except json.JSONDecodeError as exc:
        raise EvalInputError(f"outputs file {path} is not valid JSON: {exc}") from None
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise EvalInputError(f"outputs file {path} must be a JSON list of strings")
    if expected_rows is not None and len(raw) != expected_rows:
        raise EvalInputError(
            f"outputs file {Path(path).name} has {len(raw)} rows "
            f"but the dataset has {expected_rows}"
        )
    return raw


def render(template: str, variables: dict) -> str:
    """Fill ``{{ var }}`` placeholders. Strict: raises if any variable is unused or missing.

    A ``{{ ... }}`` whose contents are not a plain name (``{{ user.name }}``, ``{{text-1}}``) is an
    error rather than being sent to the model verbatim. Substitution is a single pass, so a value
    that itself contains ``{{ x }}`` is inserted as text and never expanded.
    """
    for m in _ANY_BRACES.finditer(template):
        if not _PLACEHOLDER.fullmatch(m.group(0)):
            raise KeyError(
                f"Prompt contains {m.group(0)!r}, which is not a {{{{ name }}}} placeholder "
                "(names are letters, digits and underscores)."
            )
    used: set[str] = set()

    def replace(match: re.Match) -> str:
        name = match.group(1).strip()
        if name not in variables:
            raise KeyError(
                f"Prompt references {{{{ {name} }}}} but dataset row has no such variable."
            )
        used.add(name)
        return str(variables[name])

    result = _PLACEHOLDER.sub(replace, template)
    unused = set(variables) - used
    if unused:
        raise KeyError(f"Dataset variables {unused} were not referenced in the prompt.")
    return result


def run_prompt(
    template: str, examples: list[Example], call_model: Callable[[str], str]
) -> list[str]:
    """Run prompt over every example, return the model's response per row."""
    outputs = []
    for i, ex in enumerate(examples):
        rendered = render(template, ex.variables)
        out = call_model(rendered)
        if not isinstance(out, str):
            raise TypeError(
                f"model returned {type(out).__name__}, not str, for row {i + 1}; "
                "a model caller must return the response text"
            )
        outputs.append(out)
    return outputs


def default_model_caller() -> Callable[[str], str]:
    """Return a function that sends a prompt to whichever provider is configured.

    Optional: needs ``pip install "prompt-eval-toolkit[anthropic]"`` or ``[openai]`` and a key.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is set but the SDK is missing: "
                'pip install "prompt-eval-toolkit[anthropic]"'
            ) from None
        client = Anthropic()
        model = os.getenv("MODEL", "claude-sonnet-4-5")

        def call(p: str) -> str:
            r = client.messages.create(
                model=model, max_tokens=512, messages=[{"role": "user", "content": p}]
            )
            return "".join(b.text for b in r.content if getattr(b, "type", None) == "text").strip()

        return call
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "OPENAI_API_KEY is set but the SDK is missing: "
                'pip install "prompt-eval-toolkit[openai]"'
            ) from None
        client = OpenAI()
        model = os.getenv("MODEL", "gpt-4o-mini")

        def call(p: str) -> str:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}]
            )
            return (r.choices[0].message.content or "").strip()

        return call
    raise RuntimeError(
        "No model configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
        "or replay recorded outputs with --baseline-outputs / --candidate-outputs."
    )
