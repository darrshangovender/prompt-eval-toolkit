"""CLI: `prompt-eval compare --baseline ... --candidate ... --dataset ... --metric ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .metrics import REGISTRY
from .runner import default_model_caller, load_dataset, run_prompt
from .stats import compare


def main() -> int:
    parser = argparse.ArgumentParser(prog="prompt-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cmp = sub.add_parser("compare", help="Compare two prompt versions on the same dataset")
    cmp.add_argument("--baseline", required=True, type=Path)
    cmp.add_argument("--candidate", required=True, type=Path)
    cmp.add_argument("--dataset", required=True, type=Path)
    cmp.add_argument("--metric", default="exact_match", choices=list(REGISTRY))

    args = parser.parse_args()

    if args.cmd == "compare":
        return _compare(args)
    return 1


def _compare(args) -> int:
    console = Console()
    examples = load_dataset(args.dataset)
    metric = REGISTRY[args.metric]
    call = default_model_caller()

    baseline_tmpl = args.baseline.read_text()
    candidate_tmpl = args.candidate.read_text()

    console.print(f"Comparing  [bold]baseline=[/bold]{args.baseline.name}   [bold]candidate=[/bold]{args.candidate.name}")
    console.print(f"Dataset    {args.dataset.name}  ({len(examples)} examples)")
    console.print(f"Metric     {args.metric}\n")

    console.print("Running baseline...", end=" ")
    baseline_outputs = run_prompt(baseline_tmpl, examples, call)
    console.print("done")
    console.print("Running candidate...", end=" ")
    candidate_outputs = run_prompt(candidate_tmpl, examples, call)
    console.print("done\n")

    baseline_scores = [metric(o, e.expected) for o, e in zip(baseline_outputs, examples)]
    candidate_scores = [metric(o, e.expected) for o, e in zip(candidate_outputs, examples)]

    result = compare(baseline_scores, candidate_scores)

    console.print(f"Baseline:    {result.baseline_mean:.3f}  ({int(sum(baseline_scores))} / {len(examples)})")
    console.print(f"Candidate:   {result.candidate_mean:.3f}  ({int(sum(candidate_scores))} / {len(examples)})")
    sign = "+" if result.delta >= 0 else ""
    console.print(f"Δ            {sign}{result.delta:.3f}  ({sign}{result.relative_pct:.1f}% relative)")
    console.print(f"p-value      {result.p_value:.3f}  (95% CI: {result.ci_low:+.3f}..{result.ci_high:+.3f})\n")

    if result.p_value > 0.05:
        console.print("[yellow]VERDICT: Inconclusive — collect more examples.[/yellow]")
        return 2
    if result.delta > 0:
        console.print("[bold green]VERDICT: Candidate wins. Promote.[/bold green]")
        return 0
    console.print("[bold red]VERDICT: Candidate regresses. Do not merge.[/bold red]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
