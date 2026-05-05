# mldebug

[![CI](https://github.com/anpenta/mldebug/actions/workflows/ci.yml/badge.svg)](https://github.com/anpenta/mldebug/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anpenta/mldebug/graph/badge.svg?token=XPN447G4S9)](https://codecov.io/gh/anpenta/mldebug)

[![PyPI](https://img.shields.io/pypi/v/mldebug.svg)](https://pypi.org/project/mldebug/)
[![Python](https://img.shields.io/pypi/pyversions/mldebug.svg)](https://pypi.org/project/mldebug/)
[![License](https://img.shields.io/github/license/anpenta/mldebug)](https://github.com/anpenta/mldebug/blob/main/LICENSE)

> A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

## Why mldebug

Machine learning systems often degrade silently when input data changes, even when models and code remain unchanged.

These issues are typically caused by changes in input data such as:

- feature distribution drift
- increasing missing values
- unseen categorical values
- mismatch between training and production data

mldebug makes these issues visible early by comparing datasets in a lightweight, schema-driven way and detecting unexpected changes before they impact model performance.

## When To Use mldebug

Use mldebug for fast validation of ML datasets, especially in CI or pre-deployment checks.

It is a good fit for:

- CI/CD validation pipelines
- pre-deployment data checks
- schema-based comparison between training and production data
- lightweight integration into existing ML workflows

Not intended for:

- full ML observability platforms
- real-time production monitoring
- long-term dashboards or alerting infrastructure

## What It Does

mldebug compares:

- a **reference dataset** (e.g. training data)
- a **current dataset** (e.g. production data)

It runs a suite of checks and returns a structured report of detected issues.

## Installation

```bash
pip install mldebug
```

## Quick Start

### Example Usage

```python
from mldebug import run_checks
import numpy as np

reference = {
    "age": np.array([20, 21, 22]),
    "income": np.array([1000, 1200, 1100]),
    "country": np.array(["ES", "ES", "FR"]),
}

current = {
    "age": np.array([30, 35, 40]),
    "income": np.array([900, 800, 850]),
    "country": np.array(["ES", "DE", "DE"]),
}

schema = {
    "age": "numeric",
    "income": "numeric",
    "country": "categorical",
}

report = run_checks(reference=reference, current=current, schema=schema)
```

### Output Inspection

#### Inspect Results

```python
for issue in report.issues:
    print(issue)
```

```text
[WARNING] psi_drift - country: PSI drift detected (18.0152)
```

#### Summary

```python
print(report.summary())
```

```python
{
  "total": 1,
  "by_severity": {
    "info": 0,
    "warning": 1,
    "critical": 0
  },
  "status": "issues_detected"
}
```

#### Structured Output

```python
print(report.to_dict())
```

```json
{
  "issues": [
    {
      "name": "psi_drift",
      "metric": "psi",
      "severity": "warning",
      "message": "country: PSI drift detected (18.0152)",
      "feature": "country",
      "value": 18.01521528247136,
      "threshold": 0.2
    }
  ]
}
```

### Available Checks

```python
from mldebug import list_checks

checks = list_checks()
print(checks)
```

```text
{
  "numeric": [
    "run_numeric_missing_value_check",
    "run_numeric_ks_test_check"
  ],
  "categorical": [
    "run_categorical_psi_drift_check"
  ]
}
```

## Documentation

See [documentation pages](https://anpenta.github.io/mldebug).

## Status

Active development (v0.x). APIs may evolve before v1.0.0.

See [CHANGELOG.md](https://github.com/anpenta/mldebug/blob/main/CHANGELOG.md) for version history.

## Development

### Requirements

- [git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/anpenta/mldebug
cd mldebug
uv sync
```

### Workflow

All tasks are managed via [poe](https://poethepoet.natn.io/index.html).

#### Run Tests

```bash
uv run poe test
```

#### Run Linting

```bash
uv run poe lint
```

#### Check Linting

```bash
uv run poe lint-check
```

### Dependency Management

Dependencies are [managed using uv](https://docs.astral.sh/uv/concepts/projects/dependencies/) and defined in [pyproject.toml](pyproject.toml).

For local development:

```bash
uv sync
```

This installs dependencies and updates the environment as needed.

For CI and reproducible environments:

```bash
uv sync --frozen
```

This ensures the environment exactly matches the lock file without modifying it.

### CI

This project uses CI to ensure:

- code quality (linting and type checking)
- correctness across supported Python versions
- test coverage thresholds
- reproducible builds
- automated publishing on release tags

Local development runs against the active Python environment only.

See [CI workflow](https://github.com/anpenta/mldebug/blob/main/.github/workflows/ci.yml) for details.

## Contributing

We welcome contributions.

1. Clone the repository
2. Create a feature branch
3. Make your changes
4. Ensure all CI checks pass
5. Open a pull request

## Citation

If you use mldebug in your work, please cite this software.

Preferred citation format is available in [CITATION.cff](https://github.com/anpenta/mldebug/blob/main/CITATION.cff) or via GitHub's “Cite this repository” button.

## License

See [LICENSE](https://github.com/anpenta/mldebug/blob/main/LICENSE).
