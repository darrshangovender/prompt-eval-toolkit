"""The bundled example is reproducible, and every runnable claim in the README is run."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from benchmarks import make_example_data, verify_stats
from prompt_eval import cli
from prompt_eval.runner import load_dataset, load_outputs

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


class TestBundledExample:
    def test_committed_files_are_exactly_what_the_generator_writes(self) -> None:
        for path, text in make_example_data.files().items():
            assert path.read_bytes() == text.encode("utf-8"), f"{path.name} is stale: rerun the generator"

    def test_generation_is_deterministic(self) -> None:
        assert make_example_data.generate() == make_example_data.generate()

    def test_the_dataset_is_labelled_synthetic_in_its_first_line(self) -> None:
        first = (ROOT / "data" / "example_dataset.yml").read_text(encoding="utf-8").splitlines()[0]
        assert "SYNTHETIC" in first

    def test_shape_60_rows_ten_without_an_invoice_number(self) -> None:
        rows = load_dataset(ROOT / "data" / "example_dataset.yml")
        assert len(rows) == 60
        assert sum(r.expected == "NONE" for r in rows) == 10
        assert all(set(r.variables) == {"text"} for r in rows)

    def test_expected_values_really_occur_in_the_text(self) -> None:
        for r in load_dataset(ROOT / "data" / "example_dataset.yml"):
            if r.expected != "NONE":
                assert r.expected in r.variables["text"]

    def test_both_output_files_line_up_with_the_dataset(self) -> None:
        rows = load_dataset(ROOT / "data" / "example_dataset.yml")
        for who in ("baseline", "candidate"):
            outs = load_outputs(ROOT / "data" / f"example_outputs_{who}.json", expected_rows=len(rows))
            assert all(isinstance(o, str) and o for o in outs)

    def test_the_two_simulated_prompts_differ_in_the_stated_ways(self) -> None:
        rows = load_dataset(ROOT / "data" / "example_dataset.yml")
        base = load_outputs(ROOT / "data" / "example_outputs_baseline.json")
        cand = load_outputs(ROOT / "data" / "example_outputs_candidate.json")
        wrapped = lambda outs: sum(o.startswith("The invoice number is") for o in outs)  # noqa: E731
        assert wrapped(base) > wrapped(cand)
        none_rows = [i for i, r in enumerate(rows) if r.expected == "NONE"]
        assert sum(cand[i] == "NONE" for i in none_rows) > sum(base[i] == "NONE" for i in none_rows)

    def test_the_prompts_and_the_example_dataset_use_the_same_variable(self) -> None:
        for name in ("extractor_v1.txt", "extractor_v2.txt"):
            assert "{{ text }}" in (ROOT / "prompts" / name).read_text(encoding="utf-8")

    def test_the_check_flag_reports_a_stale_file(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(make_example_data, "DATA", tmp_path)
        monkeypatch.setattr(sys, "argv", ["make_example_data.py", "--check"])
        assert make_example_data.main() == 1  # nothing in tmp_path yet
        for path, text in make_example_data.files().items():
            path.write_bytes(text.encode("utf-8"))
        assert make_example_data.main() == 0

    def test_the_small_smoke_dataset_still_loads(self) -> None:
        assert len(load_dataset(ROOT / "data" / "invoices.yml")) == 4


def _blocks(lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", README, re.S)


def _run_cli(capsys, monkeypatch, argv: list[str]) -> tuple[int, str]:
    monkeypatch.chdir(ROOT)
    code = cli.main(argv)
    return code, capsys.readouterr().out


class TestReadmeRunsAsWritten:
    def test_the_quick_start_command_runs_verbatim_and_prints_the_block_shown(
        self, capsys, monkeypatch
    ) -> None:
        bash = next(b for b in _blocks("bash") if "prompt-eval compare" in b and "--baseline-outputs" in b)
        command = bash.split("prompt-eval compare", 1)[1].replace("\\\n", " ")
        code, out = _run_cli(capsys, monkeypatch, ["compare", *shlex.split(command)])
        assert code == 0
        shown = next(b for b in _blocks("text") if b.startswith("Comparing"))
        assert out == shown

    def test_every_python_block_that_needs_no_key_runs_and_prints_what_is_shown(
        self, capsys, monkeypatch
    ) -> None:
        monkeypatch.chdir(ROOT)
        blocks = _blocks("python")
        assert blocks, "the README must show an offline python example"
        for code in blocks:
            exec(compile(code, "<README>", "exec"), {"__name__": "__readme__"})
        printed = capsys.readouterr().out
        shown = next(b for b in _blocks("text") if b.startswith("delta="))
        assert printed == shown

    def test_the_live_model_example_is_the_only_command_that_needs_a_key(self) -> None:
        bash = [b for b in _blocks("bash") if "prompt-eval compare" in b]
        assert len(bash) == 2
        assert sum("API_KEY" in b for b in bash) == 1

    def test_known_cases_and_example_tables_are_what_the_script_computes(self) -> None:
        """Columns computed by this package must match the README exactly. Columns computed by scipy
        are compared with a tolerance, so a scipy release that changes its resampling stream (the
        numbers move in the 3rd decimal) does not break the suite; their meaning is tested in
        test_stats_scipy.py."""
        known = verify_stats.known_lines(verify_stats.known_cases())
        _match_table(known, key=0, exact={0, 1, 2, 3}, approx={4})
        example = verify_stats.example_lines(verify_stats.bundled_example())
        table, ci = example[:4], example[5:]
        _match_table(table, key=0, exact={0, 1, 2, 3, 4, 5, 10}, approx={6, 7, 8, 9})
        _match_table(ci, key=0, exact={1}, approx={2})


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _numbers(cell: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", cell)]


def _match_table(lines: list[str], *, key: int, exact: set[int], approx: set[int]) -> None:
    # A table is identified by its first cell and its width (two tables list the same metric names).
    by_key = {(_cells(ln)[key], len(_cells(ln))): _cells(ln)
              for ln in README.splitlines() if ln.startswith("|")}
    for line in lines[2:]:
        got = _cells(line)
        want = by_key.get((got[key], len(got)))
        assert want is not None, f"README has no row for {got[key]!r} with {len(got)} columns"
        for i in exact:
            assert got[i] == want[i], (got[key], i, got[i], want[i])
        for i in approx:
            a, b = _numbers(got[i]), _numbers(want[i])
            if not a:  # "n/a"
                assert got[i] == want[i]
                continue
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                assert x == pytest.approx(y, rel=0.5, abs=0.006), (got[key], i, got[i], want[i])


class TestReadmeShape:
    def test_one_h1_and_four_badges_ci_first(self) -> None:
        outside = re.sub(r"```.*?```", "", README, flags=re.S)
        assert len(re.findall(r"^# ", outside, re.M)) == 1
        badges = re.findall(r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", README, re.M)
        assert len(badges) == 4 and "actions/workflows/tests.yml" in badges[0]

    @pytest.mark.parametrize(
        "heading",
        ["Why this exists", "Quick start", "How it works", "Results", "Design decisions",
         "Limitations", "Project layout", "Tests"],
    )
    def test_required_sections(self, heading: str) -> None:
        assert re.search(rf"^(#+ {re.escape(heading)}|\*\*{re.escape(heading)}\.?\*\*)", README, re.M)

    def test_blockquote_mermaid_and_footer(self) -> None:
        assert re.search(r"^> .{40,}", README, re.M)
        assert "```mermaid" in README
        assert ("Built by [Darrshan Govender](https://github.com/darrshangovender) · "
                "[Agulhas Code](https://agulhascode.co.za)") in README

    def test_no_unfilled_placeholders_and_synthetic_is_stated(self) -> None:
        assert "TESTCOUNT" not in README
        assert "SYNTHETIC" in README

    def test_the_stated_test_count_is_the_real_one(self, request) -> None:
        items = request.session.items
        files = {Path(str(i.fspath)).name for i in items}
        every_file = {p.name for p in (ROOT / "tests").glob("test_*.py")}
        if files != every_file:
            pytest.skip("only part of the suite was selected, so its size cannot be compared")
        stated = {int(n) for n in re.findall(r"(\d+) (?:tests|passed)", README)}
        assert stated == {len(items)}, (stated, len(items))

    def test_relative_links_point_at_files_that_exist(self) -> None:
        for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s#]+)", README):
            assert (ROOT / target).exists(), target


class TestProjectFiles:
    @pytest.mark.parametrize("name", ["LICENSE", "Makefile", "CHANGELOG.md", ".gitignore",
                                      ".github/workflows/tests.yml", "pyproject.toml"])
    def test_present(self, name: str) -> None:
        assert (ROOT / name).is_file()

    def test_changelog_has_the_initial_release_and_an_unreleased_section(self) -> None:
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [0.1.0] - 2026-10-09" in text and "## [Unreleased]" in text

    def test_ci_lints_for_real_and_runs_both_python_versions_and_the_benchmark(self) -> None:
        wf = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8"))
        steps = wf["jobs"]["test"]["steps"]
        lint = next(s for s in steps if "ruff" in s.get("run", ""))
        assert "continue-on-error" not in lint
        assert wf["jobs"]["test"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
        assert any("verify_stats" in s.get("run", "") for s in steps)

    def test_the_sdks_are_optional_extras_not_hard_dependencies(self) -> None:
        import tomllib

        meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        runtime = " ".join(meta["dependencies"]).lower()
        assert "anthropic" not in runtime and "openai" not in runtime and "scipy" not in runtime
        assert {"anthropic", "openai", "dev"} <= set(meta["optional-dependencies"])
        assert any(d.startswith("scipy") for d in meta["optional-dependencies"]["dev"])

    def test_data_files_are_not_swallowed_by_gitignore_rules(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "!data/*.yml" in ignore and "!data/*.json" in ignore
        assert ".hypothesis/" in ignore
