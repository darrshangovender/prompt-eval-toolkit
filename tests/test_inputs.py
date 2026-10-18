"""Dataset / outputs / prompt loading, placeholder validation, model callers and the string metrics.

Complements test_runner.py and test_metrics.py with the failure paths: every unusable input should
be refused with a message that names the file and the row, never a bare traceback.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from prompt_eval.metrics import REGISTRY
from prompt_eval.metrics.contains import contains
from prompt_eval.metrics.exact_match import exact_match
from prompt_eval.runner import (
    EvalInputError,
    Example,
    default_model_caller,
    load_dataset,
    load_outputs,
    load_prompt,
    render,
    run_prompt,
)


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return p


class TestLoadDataset:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(EvalInputError, match="dataset not found"):
            load_dataset(tmp_path / "nope.yml")

    @pytest.mark.parametrize("text", ["", "[]", "{}", "just a string", "42"])
    def test_must_be_a_non_empty_list(self, tmp_path: Path, text: str) -> None:
        with pytest.raises(EvalInputError, match="non-empty YAML list"):
            load_dataset(write(tmp_path, "d.yml", text))

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(EvalInputError, match="not valid YAML"):
            load_dataset(write(tmp_path, "d.yml", "- variables: {a: 1\n  expected: x"))

    def test_row_missing_a_key_names_the_row(self, tmp_path: Path) -> None:
        text = "- variables: {a: 1}\n  expected: x\n- variables: {a: 2}\n"
        with pytest.raises(EvalInputError, match=r"d\.yml, row 2: each row needs"):
            load_dataset(write(tmp_path, "d.yml", text))

    def test_row_that_is_not_a_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(EvalInputError, match="row 1"):
            load_dataset(write(tmp_path, "d.yml", "- just text\n"))

    def test_variables_must_be_a_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(EvalInputError, match="`variables` must be a mapping"):
            load_dataset(write(tmp_path, "d.yml", "- variables: [a, b]\n  expected: x\n"))

    def test_null_expected_is_refused_rather_than_becoming_the_string_None(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(EvalInputError, match="`expected` is empty"):
            load_dataset(write(tmp_path, "d.yml", "- variables: {a: 1}\n  expected:\n"))

    @pytest.mark.parametrize("word", ["NO", "yes", "off", "true"])
    def test_yaml_booleans_in_expected_are_refused(self, tmp_path: Path, word: str) -> None:
        """Unquoted NO / yes / off are booleans in YAML; str(False) would silently become 'False'."""
        with pytest.raises(EvalInputError, match="read by YAML as a boolean"):
            load_dataset(write(tmp_path, "d.yml", f"- variables: {{a: 1}}\n  expected: {word}\n"))

    def test_quoted_booleans_are_text(self, tmp_path: Path) -> None:
        rows = load_dataset(write(tmp_path, "d.yml", '- variables: {a: 1}\n  expected: "NO"\n'))
        assert rows[0].expected == "NO"

    def test_numbers_become_strings(self, tmp_path: Path) -> None:
        rows = load_dataset(write(tmp_path, "d.yml", "- variables: {a: 1}\n  expected: 12345\n"))
        assert rows[0].expected == "12345"

    def test_non_ascii_is_read_as_utf8(self, tmp_path: Path) -> None:
        """The platform default (cp1252 on Windows) turns UTF-8 into mojibake."""
        text = '- variables: {text: "Faktura — nr 12 ‹Δ›"}\n  expected: "مرحبا"\n'
        rows = load_dataset(write(tmp_path, "d.yml", text))
        assert rows[0].variables["text"] == "Faktura — nr 12 ‹Δ›"
        assert rows[0].expected == "مرحبا"

    def test_file_that_is_not_utf8(self, tmp_path: Path) -> None:
        p = tmp_path / "d.yml"
        p.write_bytes("- variables: {a: é}\n  expected: x\n".encode("latin-1"))
        with pytest.raises(EvalInputError, match="not valid UTF-8"):
            load_dataset(p)


class TestLoadOutputsAndPrompt:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = write(tmp_path, "o.json", '["a", "b", "ç"]')
        assert load_outputs(p, expected_rows=3) == ["a", "b", "ç"]

    def test_row_count_mismatch_names_both_numbers(self, tmp_path: Path) -> None:
        p = write(tmp_path, "o.json", '["a", "b"]')
        with pytest.raises(EvalInputError, match=r"o\.json has 2 rows but the dataset has 3"):
            load_outputs(p, expected_rows=3)

    @pytest.mark.parametrize("text", ['{"a": 1}', '"a string"', "[1, 2]", '["a", null]'])
    def test_must_be_a_list_of_strings(self, tmp_path: Path, text: str) -> None:
        with pytest.raises(EvalInputError, match="JSON list of strings"):
            load_outputs(write(tmp_path, "o.json", text))

    def test_invalid_json(self, tmp_path: Path) -> None:
        with pytest.raises(EvalInputError, match="not valid JSON"):
            load_outputs(write(tmp_path, "o.json", "[1,"))

    def test_missing_outputs_and_prompt_files(self, tmp_path: Path) -> None:
        with pytest.raises(EvalInputError, match="outputs file not found"):
            load_outputs(tmp_path / "x.json")
        with pytest.raises(EvalInputError, match="prompt file not found"):
            load_prompt(tmp_path / "x.txt")

    def test_prompt_is_read_as_utf8(self, tmp_path: Path) -> None:
        assert load_prompt(write(tmp_path, "p.txt", "Résumé — {{ x }}")) == "Résumé — {{ x }}"


class TestRenderPlaceholders:
    @pytest.mark.parametrize("bad", ["{{ user.name }}", "{{text-1}}", "{{ }}", "{{ a b }}", "{{#if x}}"])
    def test_a_brace_pair_that_is_not_a_plain_name_is_an_error_not_silent_text(self, bad: str) -> None:
        """It used to be sent to the model verbatim, so a typo scored 0 on every row."""
        with pytest.raises(KeyError, match="not a"):
            render(f"Extract from {bad}", {"x": "1"})

    def test_values_are_inserted_as_text_and_never_re_expanded(self) -> None:
        out = render("A: {{ a }} B: {{ b }}", {"a": "{{ b }}", "b": "real"})
        assert out == "A: {{ b }} B: real"

    def test_single_braces_and_json_in_the_prompt_are_untouched(self) -> None:
        out = render('Reply as {"id": "..."} for {{ x }}', {"x": "7"})
        assert out == 'Reply as {"id": "..."} for 7'

    def test_regex_metacharacters_in_values_are_inserted_literally(self) -> None:
        assert render("{{ x }}", {"x": r"\1 \g<0> $& (.*)"}) == r"\1 \g<0> $& (.*)"

    def test_multiline_template_and_value(self) -> None:
        assert render("a\n{{ x }}\nb", {"x": "1\n2"}) == "a\n1\n2\nb"


class TestRunPrompt:
    def test_a_model_returning_a_non_string_names_the_row(self) -> None:
        rows = [Example({"x": "1"}, "a"), Example({"x": "2"}, "b")]

        def model(prompt: str):
            return None if "2" in prompt else "ok"

        with pytest.raises(TypeError, match="NoneType, not str, for row 2"):
            run_prompt("{{ x }}", rows, model)

    def test_a_model_exception_propagates_unchanged(self) -> None:
        def model(prompt: str) -> str:
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError, match="boom"):
            run_prompt("{{ x }}", [Example({"x": "1"}, "a")], model)

    def test_rows_are_run_in_order_exactly_once(self) -> None:
        seen: list[str] = []
        rows = [Example({"x": str(i)}, "") for i in range(5)]
        out = run_prompt("q{{ x }}", rows, lambda p: seen.append(p) or p.upper())
        assert seen == [f"q{i}" for i in range(5)] and out == [f"Q{i}" for i in range(5)]


def _install_fake(monkeypatch: pytest.MonkeyPatch, module: str, cls: str, client) -> None:
    mod = types.ModuleType(module)
    setattr(mod, cls, lambda: client)
    monkeypatch.setitem(sys.modules, module, mod)


class TestModelCallers:
    def test_anthropic_text_blocks_are_joined_and_non_text_blocks_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = {}
        blocks = [types.SimpleNamespace(type="thinking", thinking="hmm"),
                  types.SimpleNamespace(type="text", text=" 12345 "),
                  types.SimpleNamespace(type="text", text="\n")]

        def create(**kw):
            sent.update(kw)
            return types.SimpleNamespace(content=blocks)

        client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
        _install_fake(monkeypatch, "anthropic", "Anthropic", client)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
        monkeypatch.setenv("MODEL", "some-model")
        assert default_model_caller()("hello") == "12345"
        assert sent["model"] == "some-model"
        assert sent["messages"] == [{"role": "user", "content": "hello"}]

    def test_openai_none_content_becomes_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        msg = types.SimpleNamespace(content=None)
        resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
        completions = types.SimpleNamespace(create=lambda **kw: resp)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
        _install_fake(monkeypatch, "openai", "OpenAI", client)
        monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
        assert default_model_caller()("hi") == ""

    def test_anthropic_is_preferred_when_both_keys_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        used = []
        a = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **kw: (
            used.append("anthropic"), types.SimpleNamespace(content=[
                types.SimpleNamespace(type="text", text="a")]))[1]))
        _install_fake(monkeypatch, "anthropic", "Anthropic", a)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("OPENAI_API_KEY", "y")
        assert default_model_caller()("p") == "a" and used == ["anthropic"]

    @pytest.mark.parametrize("var,extra", [("ANTHROPIC_API_KEY", "anthropic"), ("OPENAI_API_KEY", "openai")])
    def test_a_missing_sdk_points_at_the_extra(
        self, monkeypatch: pytest.MonkeyPatch, var: str, extra: str
    ) -> None:
        monkeypatch.setitem(sys.modules, extra, None)  # makes `import <extra>` raise ImportError
        monkeypatch.setenv(var, "x")
        with pytest.raises(RuntimeError, match=rf"prompt-eval-toolkit\[{extra}\]"):
            default_model_caller()

    def test_the_no_provider_message_mentions_the_offline_alternative(self) -> None:
        with pytest.raises(RuntimeError, match="--baseline-outputs"):
            default_model_caller()


class TestMetrics:
    def test_decomposed_and_precomposed_accents_are_equal(self) -> None:
        assert exact_match("Café", "Café") == 1.0
        assert contains("The word is Café.", "café") == 1.0

    def test_case_folding_is_not_just_lowercase(self) -> None:
        assert exact_match("STRASSE", "straße") == 1.0

    def test_surrounding_whitespace_including_newlines_is_ignored(self) -> None:
        assert exact_match("  12345\n", "12345") == 1.0
        assert exact_match("12 345", "12345") == 0.0

    def test_exact_match_is_not_contains(self) -> None:
        assert exact_match("the answer is 4", "4") == 0.0
        assert contains("the answer is 4", "4") == 1.0

    def test_empty_output(self) -> None:
        assert exact_match("", "x") == 0.0
        assert contains("", "x") == 0.0
        assert exact_match("", "") == 1.0

    def test_non_latin_scripts(self) -> None:
        assert exact_match("فاتورة 1234", "فاتورة 1234") == 1.0
        assert contains("رقم الفاتورة: 1234", "1234") == 1.0

    @pytest.mark.parametrize("name", list(REGISTRY))
    def test_metrics_are_pure_and_symmetric_in_case(self, name: str) -> None:
        f = REGISTRY[name]
        assert f("Abc", "abc") == f("ABC", "AbC") == f("abc", "ABC")
