"""The command line: exit codes, offline replay, failure messages, encoding."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prompt_eval import cli

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"


def replay_args(*extra: str, swap: bool = False) -> list[str]:
    a, b = ("candidate", "baseline") if swap else ("baseline", "candidate")
    return [
        "compare",
        "--baseline", str(PROMPTS / "extractor_v1.txt"),
        "--candidate", str(PROMPTS / "extractor_v2.txt"),
        "--dataset", str(DATA / "example_dataset.yml"),
        "--baseline-outputs", str(DATA / f"example_outputs_{a}.json"),
        "--candidate-outputs", str(DATA / f"example_outputs_{b}.json"),
        *extra,
    ]


def run(capsys, argv: list[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


class TestVerdictsAndExitCodes:
    def test_a_clear_win_exits_zero(self, capsys) -> None:
        code, out, _ = run(capsys, replay_args())
        assert code == 0
        assert "Candidate wins. Promote." in out
        assert "no model called" in out

    def test_the_same_data_with_the_prompts_swapped_is_a_regression_and_exits_one(self, capsys) -> None:
        code, out, _ = run(capsys, replay_args(swap=True))
        assert code == 1
        assert "Candidate regresses. Do not merge." in out
        assert "-" in out.split("Δ")[1].splitlines()[0]

    def test_too_little_evidence_exits_two(self, tmp_path: Path, capsys) -> None:
        ds = tmp_path / "d.yml"
        ds.write_text('- variables: {text: "a"}\n  expected: "1"\n'
                      '- variables: {text: "b"}\n  expected: "2"\n', encoding="utf-8")
        for name, outs in (("b.json", ["x", "x"]), ("c.json", ["1", "2"])):
            (tmp_path / name).write_text(json.dumps(outs), encoding="utf-8")
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"),
                "--candidate", str(PROMPTS / "extractor_v2.txt"), "--dataset", str(ds),
                "--baseline-outputs", str(tmp_path / "b.json"),
                "--candidate-outputs", str(tmp_path / "c.json")]
        code, out, _ = run(capsys, argv)
        assert code == 2
        assert "Inconclusive" in out

    def test_identical_outputs_are_inconclusive_not_a_crash(self, capsys) -> None:
        argv = replay_args()
        argv[argv.index("--candidate-outputs") + 1] = argv[argv.index("--baseline-outputs") + 1]
        code, out, _ = run(capsys, argv)
        assert code == 2 and "worse on 0" in out

    def test_the_contains_metric_scores_wrapped_answers_as_correct(self, capsys) -> None:
        _, exact, _ = run(capsys, replay_args("--metric", "exact_match"))
        _, loose, _ = run(capsys, replay_args("--metric", "contains"))
        assert "Baseline:    0.400" in exact
        assert "Baseline:    0.733" in loose


class TestJsonOutput:
    def test_json_is_one_parsable_object_with_the_counts(self, capsys) -> None:
        code, out, _ = run(capsys, replay_args("--json"))
        payload = json.loads(out)
        assert code == payload["exit_code"] == 0
        assert (payload["wins"], payload["losses"], payload["ties"]) == (32, 1, 27)
        assert payload["n"] == 60 and payload["significant"] is True
        assert payload["seed"] == 42 and payload["n_bootstrap"] == 5000

    def test_json_has_no_infinity_which_is_not_valid_json(self, tmp_path: Path, capsys) -> None:
        ds = tmp_path / "d.yml"
        ds.write_text("".join(f'- variables: {{text: "r{i}"}}\n  expected: "ok"\n' for i in range(8)),
                      encoding="utf-8")
        (tmp_path / "b.json").write_text(json.dumps(["no"] * 8), encoding="utf-8")
        (tmp_path / "c.json").write_text(json.dumps(["ok"] * 8), encoding="utf-8")
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"),
                "--candidate", str(PROMPTS / "extractor_v2.txt"), "--dataset", str(ds),
                "--baseline-outputs", str(tmp_path / "b.json"),
                "--candidate-outputs", str(tmp_path / "c.json"), "--json"]
        code, out, _ = run(capsys, argv)
        assert json.loads(out, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))[
            "relative_pct"] is None
        assert code == 0

    def test_json_mode_prints_only_the_object(self, capsys) -> None:
        _, out, _ = run(capsys, replay_args("--json"))
        assert out.strip().startswith("{") and out.strip().endswith("}") and "\n" not in out.strip()

    def test_a_baseline_of_zero_says_relative_change_is_undefined(self, tmp_path: Path, capsys) -> None:
        ds = tmp_path / "d.yml"
        ds.write_text("".join(f'- variables: {{text: "r{i}"}}\n  expected: "ok"\n' for i in range(8)),
                      encoding="utf-8")
        (tmp_path / "b.json").write_text(json.dumps(["no"] * 8), encoding="utf-8")
        (tmp_path / "c.json").write_text(json.dumps(["ok"] * 8), encoding="utf-8")
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"),
                "--candidate", str(PROMPTS / "extractor_v2.txt"), "--dataset", str(ds),
                "--baseline-outputs", str(tmp_path / "b.json"),
                "--candidate-outputs", str(tmp_path / "c.json")]
        _, out, _ = run(capsys, argv)
        assert "inf" not in out and "relative change undefined" in out


class TestReproducibility:
    def test_same_seed_gives_byte_identical_output(self, capsys) -> None:
        assert run(capsys, replay_args("--seed", "5"))[1] == run(capsys, replay_args("--seed", "5"))[1]

    def test_a_different_seed_is_reported_and_keeps_the_verdict(self, capsys) -> None:
        _, a, _ = run(capsys, replay_args("--seed", "1"))
        code, b, _ = run(capsys, replay_args("--seed", "2"))
        assert "seed 1" in a and "seed 2" in b and code == 0

    def test_n_bootstrap_is_reported(self, capsys) -> None:
        _, out, _ = run(capsys, replay_args("--n-bootstrap", "1234"))
        assert "B=1234" in out


class TestFailuresExitThreeNotAnInterpretableVerdict:
    def test_no_model_and_no_replay_files(self, capsys, monkeypatch) -> None:
        # No replay flags -> live mode, and conftest has removed every provider key.
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"),
                "--candidate", str(PROMPTS / "extractor_v2.txt"),
                "--dataset", str(DATA / "example_dataset.yml")]
        code, out, err = run(capsys, argv)
        assert code == 3
        assert "No model configured" in err and "Traceback" not in err
        assert "VERDICT" not in out

    def test_a_usage_error_is_exit_three_not_argparses_two(self, capsys) -> None:
        with pytest.raises(SystemExit) as e:
            cli.main(["compare", "--baseline", "x"])
        assert e.value.code == 3
        assert "required" in capsys.readouterr().err

    def test_an_unknown_metric_is_a_usage_error(self, capsys) -> None:
        with pytest.raises(SystemExit) as e:
            cli.main(replay_args("--metric", "bleu"))
        assert e.value.code == 3

    def test_one_replay_file_without_the_other(self, capsys) -> None:
        argv = replay_args()
        i = argv.index("--candidate-outputs")
        del argv[i:i + 2]
        code, _, err = run(capsys, argv)
        assert code == 3 and "both --baseline-outputs and --candidate-outputs" in err

    def test_outputs_of_the_wrong_length(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "short.json").write_text('["a"]', encoding="utf-8")
        argv = replay_args()
        argv[argv.index("--baseline-outputs") + 1] = str(tmp_path / "short.json")
        code, _, err = run(capsys, argv)
        assert code == 3 and "short.json has 1 rows but the dataset has 60" in err

    def test_missing_dataset(self, tmp_path: Path, capsys) -> None:
        argv = replay_args()
        argv[argv.index("--dataset") + 1] = str(tmp_path / "gone.yml")
        code, _, err = run(capsys, argv)
        assert code == 3 and "dataset not found" in err

    def test_a_prompt_with_a_variable_the_dataset_lacks_fails_before_any_model_call(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        calls = []
        monkeypatch.setattr(cli, "default_model_caller", lambda: calls.append(1) or (lambda p: "x"))
        bad = tmp_path / "bad.txt"
        bad.write_text("Read {{ invoice_text }}", encoding="utf-8")
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"), "--candidate", str(bad),
                "--dataset", str(DATA / "example_dataset.yml")]
        code, _, err = run(capsys, argv)
        assert code == 3 and "invoice_text" in err
        assert len(calls) == 1, "the caller is built, but must never be invoked"

    def test_the_live_model_is_not_invoked_when_a_later_prompt_is_malformed(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        invoked = []
        monkeypatch.setattr(cli, "default_model_caller", lambda: (lambda p: invoked.append(p) or "x"))
        bad = tmp_path / "bad.txt"
        bad.write_text("Read {{ user.text }}", encoding="utf-8")
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"), "--candidate", str(bad),
                "--dataset", str(DATA / "example_dataset.yml")]
        code, _, _ = run(capsys, argv)
        assert code == 3 and invoked == [], "baseline must not be paid for when the candidate is broken"

    @pytest.mark.parametrize("n", ["0", "99", "-1"])
    def test_n_bootstrap_below_the_minimum(self, capsys, n: str) -> None:
        code, _, err = run(capsys, replay_args("--n-bootstrap", n))
        assert code == 3 and "--n-bootstrap must be at least 100" in err


class TestLiveModelPath:
    def test_a_live_run_calls_the_model_once_per_row_per_prompt(self, capsys, monkeypatch) -> None:
        prompts = []

        def model(prompt: str) -> str:
            prompts.append(prompt)
            return "12345"

        monkeypatch.setattr(cli, "default_model_caller", lambda: model)
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"),
                "--candidate", str(PROMPTS / "extractor_v2.txt"),
                "--dataset", str(DATA / "invoices.yml")]
        code, out, _ = run(capsys, argv)
        assert len(prompts) == 8  # 4 rows x 2 prompts
        assert "live model" in out and code == 2  # identical outputs: nothing to detect

    def test_a_model_failure_midway_is_reported_not_a_traceback(self, capsys, monkeypatch) -> None:
        def model(prompt: str) -> str:
            raise RuntimeError("rate limited")

        monkeypatch.setattr(cli, "default_model_caller", lambda: model)
        argv = ["compare", "--baseline", str(PROMPTS / "extractor_v1.txt"),
                "--candidate", str(PROMPTS / "extractor_v2.txt"),
                "--dataset", str(DATA / "invoices.yml")]
        code, _, err = run(capsys, argv)
        assert code == 3 and "rate limited" in err


class TestOutputHandling:
    def test_rich_markup_in_a_file_name_is_printed_literally(self, tmp_path: Path, capsys) -> None:
        odd = tmp_path / "[bold]v1.txt"
        odd.write_text((PROMPTS / "extractor_v1.txt").read_text(encoding="utf-8"), encoding="utf-8")
        argv = replay_args()
        argv[argv.index("--baseline") + 1] = str(odd)
        code, out, _ = run(capsys, argv)
        assert code == 0 and "[bold]v1.txt" in out

    def test_a_long_line_is_not_wrapped_or_cut(self, capsys) -> None:
        _, out, _ = run(capsys, replay_args())
        line = next(ln for ln in out.splitlines() if ln.startswith("p-value"))
        assert line.endswith("both must be <= 0.05)")


def test_the_real_entry_point_survives_a_non_utf8_console(tmp_path: Path) -> None:
    """Piped output on Windows defaults to cp1252, which cannot encode the Δ the report prints.
    A StringIO-based test never sees this, so run the module in a real subprocess."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONUTF8"] = "0"
    env["PYTHONIOENCODING"] = "cp1252:strict"
    proc = subprocess.run(
        [sys.executable, "-m", "prompt_eval.cli", *replay_args()],
        capture_output=True, cwd=ROOT, env=env, timeout=120,
    )
    out = proc.stdout.decode("utf-8")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "Δ" in out and "Candidate wins" in out


def test_utf8_prompt_and_dataset_reach_the_model_intact(tmp_path: Path, capsys, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(cli, "default_model_caller", lambda: (lambda p: seen.append(p) or "ok"))
    prompt = tmp_path / "p.txt"
    prompt.write_bytes("Fatura — çıkar: {{ text }}".encode())
    ds = tmp_path / "d.yml"
    ds.write_bytes('- variables: {text: "No. ۱۲۳ — ✓"}\n  expected: "ok"\n'.encode())
    code, _, _ = run(capsys, ["compare", "--baseline", str(prompt), "--candidate", str(prompt),
                              "--dataset", str(ds)])
    assert seen == ["Fatura — çıkar: No. ۱۲۳ — ✓"] * 2 and code == 2
