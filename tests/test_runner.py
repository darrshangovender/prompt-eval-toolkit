"""Tests for dataset loading, strict template rendering, and prompt execution."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from prompt_eval.runner import (
    Example,
    default_model_caller,
    load_dataset,
    render,
    run_prompt,
)


class TestRender:
    """`render` is deliberately strict — a silently-unfilled placeholder would
    corrupt an eval run without anyone noticing, so both directions of
    variable/placeholder mismatch must raise."""

    def test_substitutes_placeholder(self) -> None:
        assert render("Text: {{ text }}", {"text": "Invoice 12345"}) == "Text: Invoice 12345"

    def test_tolerates_missing_inner_whitespace(self) -> None:
        assert render("{{text}}", {"text": "a"}) == "a"

    def test_substitutes_same_variable_twice(self) -> None:
        out = render("{{ x }} and {{ x }}", {"x": "9"})
        assert out == "9 and 9"

    def test_coerces_non_string_values(self) -> None:
        assert render("n={{ n }}", {"n": 42}) == "n=42"

    def test_placeholder_with_no_matching_variable_raises(self) -> None:
        with pytest.raises(KeyError, match="no such variable"):
            render("{{ missing }}", {"text": "a"})

    def test_unused_variable_raises(self) -> None:
        # A dataset column the prompt forgot to reference is a silent scoring
        # bug, so it is an error rather than a warning.
        with pytest.raises(KeyError, match="not referenced"):
            render("static prompt", {"text": "a"})

    def test_no_variables_and_no_placeholders_is_fine(self) -> None:
        assert render("hello", {}) == "hello"


class TestLoadDataset:
    def test_loads_shipped_demo_dataset(self, dataset_path: Path) -> None:
        examples = load_dataset(dataset_path)
        assert len(examples) == 4
        assert all(isinstance(ex, Example) for ex in examples)

    def test_expected_values_are_strings(self, tmp_path: Path) -> None:
        # YAML would parse a bare 12345 as int; the metrics do string
        # comparison, so load_dataset must coerce.
        path = tmp_path / "ds.yml"
        path.write_text("- variables:\n    text: t\n  expected: 12345\n")
        (example,) = load_dataset(path)
        assert example.expected == "12345"

    def test_variables_are_preserved_verbatim(self, dataset_path: Path) -> None:
        examples = load_dataset(dataset_path)
        assert examples[0].variables == {"text": "Invoice 12345 dated 2025-01-15"}


class TestRunPrompt:
    def test_returns_one_output_per_example(self, echo_model: Callable[[str], str]) -> None:
        examples = [Example({"text": "a"}, "a"), Example({"text": "b"}, "b")]
        assert run_prompt("{{ text }}", examples, echo_model) == ["a", "b"]

    def test_sends_the_rendered_prompt_not_the_template(self) -> None:
        seen: list[str] = []

        def spy(prompt: str) -> str:
            seen.append(prompt)
            return "ok"

        run_prompt("Q: {{ text }}", [Example({"text": "42"}, "42")], spy)
        assert seen == ["Q: 42"]

    def test_propagates_render_errors(self, echo_model: Callable[[str], str]) -> None:
        with pytest.raises(KeyError):
            run_prompt("{{ nope }}", [Example({"text": "a"}, "a")], echo_model)

    def test_empty_dataset_yields_no_calls(self) -> None:
        calls = 0

        def counter(_: str) -> str:
            nonlocal calls
            calls += 1
            return ""

        assert run_prompt("{{ x }}", [], counter) == []
        assert calls == 0


class TestDefaultModelCaller:
    def test_raises_when_no_provider_configured(self) -> None:
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY or OPENAI_API_KEY"):
            default_model_caller()

    def test_provider_sdks_are_not_imported_at_module_load(self) -> None:
        # The SDKs are heavy and only needed when a key is actually set.
        # Checked in a clean subprocess rather than against sys.modules so the
        # result cannot be perturbed by test ordering or pytest plugins.
        code = (
            "import sys; import prompt_eval.runner, prompt_eval.cli; "
            "print(int('anthropic' in sys.modules or 'openai' in sys.modules))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert proc.stdout.strip() == "0", "provider SDK imported eagerly"
