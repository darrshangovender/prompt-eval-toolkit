"""CLI: ``prompt-eval compare --baseline ... --candidate ... --dataset ... --metric ...``.

Exit codes (so a CI step can tell them apart):

    0  candidate is significantly better: promote
    1  candidate is significantly worse: do not merge
    2  inconclusive: not enough evidence either way
    3  could not run: bad usage, unreadable input, no model configured, a model error
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from .metrics import REGISTRY
from .runner import (
    EvalInputError,
    default_model_caller,
    load_dataset,
    load_outputs,
    load_prompt,
    render,
    run_prompt,
)
from .stats import ALPHA, MIN_BOOTSTRAP, CompareResult, compare

EXIT_PROMOTE, EXIT_REGRESS, EXIT_INCONCLUSIVE, EXIT_ERROR = 0, 1, 2, 3


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse exits 2 here, which would read as "inconclusive"
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="prompt-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmp = sub.add_parser(
        "compare",
        help="Compare two prompt versions on the same dataset",
        description="Run (or replay) two prompts over one dataset and test the paired difference.",
    )
    cmp.add_argument("--baseline", required=True, type=Path, help="baseline prompt file")
    cmp.add_argument("--candidate", required=True, type=Path, help="candidate prompt file")
    cmp.add_argument("--dataset", required=True, type=Path, help="YAML dataset")
    cmp.add_argument("--metric", default="exact_match", choices=list(REGISTRY))
    cmp.add_argument(
        "--baseline-outputs", type=Path,
        help="JSON list of recorded baseline outputs (one per row): no model is called",
    )
    cmp.add_argument("--candidate-outputs", type=Path, help="recorded candidate outputs, as above")
    cmp.add_argument("--n-bootstrap", type=int, default=5000, help="resamples (default 5000)")
    cmp.add_argument("--seed", type=int, default=42, help="bootstrap seed (default 42)")
    cmp.add_argument("--json", action="store_true", dest="as_json", help="print one JSON object")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    try:
        return _compare(args)
    except (EvalInputError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"prompt-eval: error: {message}", file=sys.stderr)
        return EXIT_ERROR


def _score(metric, outputs: list[str], examples) -> list[float]:
    return [metric(o, e.expected) for o, e in zip(outputs, examples, strict=True)]


def _outputs_for(template: str, replay: Path | None, examples, call) -> list[str]:
    if replay is not None:
        return load_outputs(replay, expected_rows=len(examples))
    return run_prompt(template, examples, call)


def _compare(args: argparse.Namespace) -> int:
    if (args.baseline_outputs is None) != (args.candidate_outputs is None):
        raise EvalInputError("give both --baseline-outputs and --candidate-outputs, or neither")
    if args.n_bootstrap < MIN_BOOTSTRAP:
        raise EvalInputError(f"--n-bootstrap must be at least {MIN_BOOTSTRAP}")
    replay = args.baseline_outputs is not None
    console = Console(highlight=False, soft_wrap=True, quiet=args.as_json)

    examples = load_dataset(args.dataset)
    metric = REGISTRY[args.metric]
    baseline_tmpl = load_prompt(args.baseline)
    candidate_tmpl = load_prompt(args.candidate)
    call = None if replay else default_model_caller()

    console.print(
        f"Comparing  baseline={escape(args.baseline.name)}   "
        f"candidate={escape(args.candidate.name)}"
    )
    console.print(f"Dataset    {escape(args.dataset.name)}  ({len(examples)} examples)")
    console.print(f"Metric     {args.metric}")
    source = "replayed from files (no model called)" if replay else "live model"
    console.print(f"Outputs    {source}\n")

    # Render every row of BOTH prompts first, in both modes: a bad placeholder in either prompt
    # fails before any model call is paid for.
    for template in (baseline_tmpl, candidate_tmpl):
        for ex in examples:
            render(template, ex.variables)
    baseline_outputs = _outputs_for(baseline_tmpl, args.baseline_outputs, examples, call)
    candidate_outputs = _outputs_for(candidate_tmpl, args.candidate_outputs, examples, call)
    baseline_scores = _score(metric, baseline_outputs, examples)
    candidate_scores = _score(metric, candidate_outputs, examples)

    result = compare(
        baseline_scores, candidate_scores, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    code = _verdict_code(result)

    if args.as_json:
        payload = {
            "baseline_mean": result.baseline_mean, "candidate_mean": result.candidate_mean,
            "delta": result.delta,
            "relative_pct": result.relative_pct if math.isfinite(result.relative_pct) else None,
            "p_value": result.p_value, "sign_p_value": result.sign_p_value,
            "ci_low": result.ci_low, "ci_high": result.ci_high, "n": result.n,
            "wins": result.wins, "losses": result.losses, "ties": result.ties,
            "n_bootstrap": result.n_bootstrap, "seed": result.seed,
            "significant": result.significant, "exit_code": code,
        }
        print(json.dumps(payload))
        return code

    n = len(examples)
    console.print(f"Baseline:    {result.baseline_mean:.3f}  ({sum(baseline_scores):.0f} / {n})")
    console.print(f"Candidate:   {result.candidate_mean:.3f}  ({sum(candidate_scores):.0f} / {n})")
    sign = "+" if result.delta >= 0 else ""
    if not math.isfinite(result.relative_pct):
        rel = "relative change undefined: baseline scored 0"
    else:
        rel = f"{sign}{result.relative_pct:.1f}% relative"
    console.print(f"Δ            {sign}{result.delta:.3f}  ({rel})")
    console.print(
        f"Pairs        candidate better on {result.wins}, worse on {result.losses}, "
        f"same on {result.ties}"
    )
    console.print(
        f"p-value      {_fmt_p(result.p_value)}  (paired bootstrap, B={result.n_bootstrap}, "
        f"seed {result.seed}; exact sign test p{_cmp_p(result.sign_p_value)}; "
        f"both must be <= {ALPHA})"
    )
    console.print(f"95% CI       {result.ci_low:+.3f}..{result.ci_high:+.3f}\n")

    if code == EXIT_INCONCLUSIVE:
        console.print("[yellow]VERDICT: Inconclusive — collect more examples.[/yellow]")
    elif code == EXIT_PROMOTE:
        console.print("[bold green]VERDICT: Candidate wins. Promote.[/bold green]")
    else:
        console.print("[bold red]VERDICT: Candidate regresses. Do not merge.[/bold red]")
    return code


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _cmp_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"={p:.4f}"


def _verdict_code(result: CompareResult) -> int:
    if not result.significant:
        return EXIT_INCONCLUSIVE
    return EXIT_PROMOTE if result.delta > 0 else EXIT_REGRESS


if __name__ == "__main__":
    sys.exit(main())
