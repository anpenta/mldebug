# mldebug

[![CI](https://github.com/anpenta/mldebug/actions/workflows/ci.yml/badge.svg)](https://github.com/anpenta/mldebug/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anpenta/mldebug/graph/badge.svg?token=XPN447G4S9)](https://codecov.io/gh/anpenta/mldebug)

[![PyPI](https://img.shields.io/pypi/v/mldebug.svg)](https://pypi.org/project/mldebug/)
[![Python](https://img.shields.io/pypi/pyversions/mldebug.svg)](https://pypi.org/project/mldebug/)
[![License](https://img.shields.io/github/license/anpenta/mldebug)](https://github.com/anpenta/mldebug/blob/main/LICENSE)

> A lightweight Python package for comparing datasets and detecting unexpected changes in machine learning systems.

## Why mldebug

Machine learning systems fail silently when data changes.

Common production issues include:

* feature distribution drift
* increasing missing values
* unseen categorical values
* training vs production mismatch

**mldebug provides a unified way to detect these issues before they become model failures.**

## What it does

mldebug compares:

* a **reference dataset** (e.g. training data)
* a **current dataset** (e.g. production data)

It runs a suite of checks and returns a structured report of detected issues.

## Installation

```bash
pip install mldebug
```

## Quick Start

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

## Inspect detected issues

```python
for issue in report.issues:
    print(issue)
```

```text
[WARNING] psi_drift - country: PSI drift detected (18.0152)
```

## Summary

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

## Structured output

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

## Available checks

mldebug provides runtime introspection of all available checks.

You can view the checks available in your installed version:

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

See [CHANGELOG.md](https://github.com/anpenta/mldebug/blob/main/CHANGELOG.md) for version history and updates.

## Development Setup

### Requirements

- [Ubuntu 24.04.4](https://releases.ubuntu.com/noble/) (recommended) or [WSL](https://ubuntu.com/desktop/wsl)
- [git](https://git-scm.com/)
- [direnv](https://direnv.net/)

### Environment Setup

```bash
git clone https://github.com/anpenta/mldebug
cd mldebug
direnv allow
```

## Development Workflow

Tasks are managed via [poe](https://poethepoet.natn.io/index.html) (available in the project environment via direnv).

### Run tests

```bash
poe test
```

### Run linting

```bash
poe lint
```

### Check linting

```bash
poe lint-check
```

### Run full CI parity checks

```bash
poe test-all
```

```bash
poe lint-check-all
```

### CI/CD

CI runs multi-Python version testing and linting. All pull requests must pass the checks before merging.

See [CI workflow](https://github.com/anpenta/mldebug/blob/main/.github/workflows/ci.yml) for details.

## Contributing

We welcome contributions.

1. Clone the repository
2. Create a feature branch
3. Make your changes
4. Ensure all CI checks pass
5. Open a pull request

## Dependency Management

Dependencies are [managed using uv](https://docs.astral.sh/uv/concepts/projects/dependencies/) and defined in [pyproject.toml](pyproject.toml).

## Citation

If you use mldebug in your work, please cite this software.

Preferred citation format is available in [CITATION.cff](https://github.com/anpenta/mldebug/blob/main/CITATION.cff) or via GitHub's “Cite this repository” button.

## License

See [LICENSE](https://github.com/anpenta/mldebug/blob/main/LICENSE).
