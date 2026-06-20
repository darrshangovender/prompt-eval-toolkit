<div align="center">

# prompt-eval-toolkit — A/B test prompts before merging

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![CLI](https://img.shields.io/badge/Interface-CLI-555)](#)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude-CC785C)](https://anthropic.com)
[![Status](https://img.shields.io/badge/Status-Working%20code-blue)](#)
[![Statistics](https://img.shields.io/badge/Stats-Paired%20bootstrap-059669)](#why-paired-bootstrap-and-not-a-paired-t-test)

</div>

---

> A tiny CLI that runs two versions of a prompt against the same dataset and tells you which one wins, on which metrics, with what confidence.

**The problem.** Prompts get tweaked weekly. Engineers eyeball "this output looks better" and ship the change. Two weeks later, a customer reports the bot suddenly hallucinates.

**The fix.** Treat every prompt change like a code change: run an eval, look at numbers, only merge if it wins.

---

## What it does

```bash
prompt-eval compare \
    --baseline prompts/extractor_v1.txt \
    --candidate prompts/extractor_v2.txt \
    --dataset data/invoices.yml \
    --metric exact_match
```

Output:

```
Comparing  baseline=extractor_v1.txt   candidate=extractor_v2.txt
Dataset    invoices.yml  (40 examples)
Metric     exact_match

Baseline:    0.78  (31 / 40)
Candidate:   0.88  (35 / 40)
Δ            +0.10  (+12.8% relative)
p-value      0.038  (95% CI: +0.02..+0.18)

VERDICT: Candidate wins. Promote.
```

If the p-value is > 0.05, the verdict is "inconclusive — collect more data". If the candidate loses, it says so flatly.

---

## Why this is worth shipping as its own tool

Most RAG / agent / prompt-tweak repos do this evaluation inside a Jupyter notebook nobody runs twice. A CLI that lives in CI is the difference between "we measure prompt changes" and "we say we measure prompt changes."

## Repo structure

```
.
├── prompt_eval/
│   ├── cli.py          # argparse + the `compare` subcommand
│   ├── runner.py       # runs a prompt over a dataset
│   ├── metrics/
│   │   ├── exact_match.py
│   │   ├── contains.py     # answer must contain expected substring
│   │   └── judge.py        # LLM-as-judge, 0..1 score
│   └── stats.py        # bootstrap CI + p-value
├── prompts/
│   ├── extractor_v1.txt
│   └── extractor_v2.txt
├── data/
│   └── invoices.yml
└── pyproject.toml
```

## The contract

A prompt file is just text with `{{ variable }}` placeholders. A dataset is YAML:

```yaml
- variables:
    text: "Invoice 12345 dated 2025-01-15"
  expected: "12345"
```

Run the same prompt, fill in `{{ text }}`, send to the model, compare against `expected` using your chosen metric.

## Status

- [x] Exact-match and substring metrics
- [x] LLM-as-judge metric
- [x] Bootstrap CI + p-value
- [x] CLI: `compare`, `run`, `list`
- [ ] Multi-candidate (3+ way compare)
- [ ] HTML diff renderer

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za)
