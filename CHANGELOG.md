# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking
- **The p-value is now a bootstrap hypothesis test.** It used to be twice the fraction of resampled
  deltas on the far side of zero, which is a confidence-interval heuristic. The differences are now
  centred before resampling and `p = (1 + extreme) / (B + 1)`, so it is never exactly 0. A unanimous
  20-row win now reports `1.9e-06` (the exact sign-flip value `2^(1-n)`), not `0.0`. Verdicts on
  borderline datasets can change.
- `anthropic` and `openai` are optional extras (`pip install "prompt-eval-toolkit[anthropic]"`), no
  longer hard dependencies. The statistics are standard library only.
- Usage errors and every "could not run" condition (unreadable file, no model configured, a model
  error) now exit **3**. argparse's own usage-error code is 2, which a CI job read as "inconclusive",
  and an unhandled error exited 1, which it read as "candidate regresses".
- `{{ user.name }}` and any other brace pair that is not a plain `{{ name }}` is now an error in
  `render` (it used to be sent to the model as literal text, scoring 0 on every row).
- `contains` and `exact_match` Unicode-normalise (NFC) and case-fold instead of `.lower()`.
- A dataset `expected` that is empty, or that YAML reads as a boolean (unquoted `NO`, `yes`, `off`),
  is rejected instead of becoming the string `None` / `False`.

### Added
- **Offline comparison from recorded outputs**: `--baseline-outputs` / `--candidate-outputs` (JSON
  lists of strings). With them, no model and no key are needed.
- A bundled **SYNTHETIC** example: 60 templated invoice texts (`data/example_dataset.yml`) and two sets
  of simulated outputs, regenerated and checked by `benchmarks/make_example_data.py`.
- `benchmarks/verify_stats.py`: closed-form known cases (an exact binomial tail, the unanimous case,
  the exact enumerated bootstrap distribution of a five-row example), cross-checks against scipy
  (`scipy.stats.bootstrap`, paired t, sign-flip permutation, exact binomial), a calibration table
  (false-alarm rate when the prompts are equal) and a power table.
- `CompareResult` now carries `wins`, `losses`, `ties`, `n_bootstrap` and `seed`; `compare` takes
  `seed`. The CLI prints the pair counts, the resampling settings and both p-values, and has
  `--seed`, `--n-bootstrap` and `--json`.
- `EvalInputError` for dataset, prompt and outputs problems, naming the file and row.
- Hypothesis property tests (prompt swap, constant shift, rescale, determinism) and tests for every
  failure path, every exit code, UTF-8 handling and the README's runnable blocks.
- `CHANGELOG.md`, a real `Makefile` (`example`, `data`, `bench`), `[project.urls]`, classifiers and an
  explicit ruff rule set.

### Fixed
- **A piped run crashed on Windows**: the report prints `Δ`, which cp1252 cannot encode. Output is now
  UTF-8 regardless of the console code page. Prompt, dataset and outputs files are read as UTF-8
  (they used the platform default, which turns UTF-8 into mojibake on Windows).
- A missing API key ended in a traceback with exit code 1, indistinguishable from "candidate
  regresses". It now prints one line and exits 3.
- `relative_pct` was `inf` when both means were 0 (no change); it is now `0.0`. The CLI says
  "relative change undefined" instead of printing `+inf%`, and `--json` emits `null`, not the
  non-JSON `Infinity`.
- A bad placeholder in the *candidate* prompt was found only after the baseline had been run against
  the live model (paid for). Both prompts are now rendered against every row first.
- Rich markup in a file name (`[bold]v1.txt`) was interpreted when printed.
- The Anthropic caller took `content[0].text`; it now joins the `text` blocks (a leading non-text
  block no longer raises).
- A model caller that returns a non-string now fails with the row number instead of an
  `AttributeError` inside the metric.
- `compare` rejects `NaN`/`inf` scores and `n_bootstrap` below 100 (it divided by zero at 0).
- Lint is a required CI step (it was `continue-on-error`).

### Removed
- The README's illustrative numbers (`0.78 / 0.88 / p = 0.038`). They were not produced by any script;
  the Quick start now shows a real run on the bundled example.
- The claim that pairing "is far more sensitive on 40-200 row datasets": not measured. The README now
  reports calibration and power tables, including where the tool is weak.

## [0.1.0] - 2026-10-09

### Added
- `prompt-eval compare`: runs two prompt files over a YAML dataset, scores each row with
  `exact_match` or `contains`, and reports means, delta, a paired-bootstrap p-value and interval,
  with exit codes 0 (promote), 1 (regression), 2 (inconclusive).
- An exact two-sided sign test as a floor on `CompareResult.significant`.
