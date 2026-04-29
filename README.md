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

### Human-readable output

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

## Logs

```python
for line in report.to_logs():
    print(line)
```

```text
[WARNING] psi_drift - country: PSI drift detected (18.0152)
```

## Supported Checks

mldebug runs a combination of:

### Numeric features

* [Kolmogorov–Smirnov test (KS test)](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
* missing value rate changes

### Categorical features

* [Population Stability Index (PSI)](https://www.geeksforgeeks.org/data-science/population-stability-index-psi/)
* category distribution changes

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

## License

See [LICENSE](https://github.com/anpenta/mldebug/blob/main/LICENSE).
