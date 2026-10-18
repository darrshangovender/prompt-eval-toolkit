.PHONY: install test lint example data bench bench-quick clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check .

# The bundled SYNTHETIC example, compared from recorded outputs: offline, no key.
example:
	prompt-eval compare --baseline prompts/extractor_v1.txt --candidate prompts/extractor_v2.txt \
		--dataset data/example_dataset.yml \
		--baseline-outputs data/example_outputs_baseline.json \
		--candidate-outputs data/example_outputs_candidate.json

# Rewrite the example data (fixed seed); `python benchmarks/make_example_data.py --check` only checks.
data:
	python benchmarks/make_example_data.py

# Known cases, scipy cross-checks, calibration and power tables (about 2.5 minutes).
bench:
	python benchmarks/verify_stats.py

# The same with 200 simulated trials per cell instead of 1,000: a noisier smoke run (about 40 s).
bench-quick:
	python benchmarks/verify_stats.py --quick

clean:
	rm -rf .pytest_cache .ruff_cache .hypothesis *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
