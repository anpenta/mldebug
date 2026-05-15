# mldebug

[![CI](https://github.com/anpenta/mldebug/actions/workflows/ci.yml/badge.svg)](https://github.com/anpenta/mldebug/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anpenta/mldebug/graph/badge.svg?token=XPN447G4S9)](https://codecov.io/gh/anpenta/mldebug)
[![PyPI](https://img.shields.io/pypi/v/mldebug.svg)](https://pypi.org/project/mldebug/)
[![Python](https://img.shields.io/pypi/pyversions/mldebug.svg)](https://pypi.org/project/mldebug/)
[![License](https://img.shields.io/github/license/anpenta/mldebug)](https://github.com/anpenta/mldebug/blob/main/LICENSE)

> A lightweight Python package for validating and comparing datasets in machine learning pipelines.

## Why mldebug

Machine learning systems often degrade silently when input data changes, even when models and code remain unchanged.

This happens due to changes in input data, such as feature distribution drifts, increasing missing values, unseen categories, or mismatches between training and production data.

These issues are often hard to detect early and can silently degrade model performance.

## What it does

mldebug compares datasets in a schema-driven way and detects unexpected changes before they reach production.

It is designed for fast validation in CI or pre-deployment checks and integrates easily into existing ML workflows.

It is not intended for full ML observability, real-time monitoring, or long-term dashboards.

## Installation

```bash
pip install mldebug
```

## Quick start

```python
from mldebug import validate, FeatureType
import numpy as np

reference = {
    "age": np.array([20, 21, 22]),
    "country": np.array(["ES", "ES", "FR"]),
}

current = {
    "age": np.array([21, 22, 23]),
    "country": np.array(["ES", "DE", "DE"]),
}

schema = {
    "age": FeatureType.NUMERIC,
    "country": FeatureType.CATEGORICAL,
}

report = validate(reference=reference, current=current, schema=schema)

report.score()
```

## Understanding the output

mldebug returns a report object containing detected issues and inspection methods.

Start by checking overall data quality with `report.score()`, and explore `report.issues`, `report.summary()`, or `report.to_dict()` depending on what you need.

### Issues

```python
for issue in report.issues:
    print(issue)
```

```text
[WARNING] range_anomaly - age: 1 values outside [20.0000, 22.0000]
[WARNING] psi_drift - country: PSI drift detected (18.0152)
[WARNING] unseen_categories - country: 1 unseen categories detected (e.g. ['DE'])
```

### Summary

```python
print(report.summary())
```

```json
{
  "total": 3,
  "by_severity": {
    "info": 0,
    "warning": 3,
    "critical": 0
  },
  "status": "issues_detected"
}
```

### Structured output

```python
print(report.to_dict())
```

```json
{
  "issues": [
    {
      "name": "range_anomaly",
      "metric": "out_of_range_count",
      "severity": "warning",
      "message": "age: 1 values outside [20.0000, 22.0000]",
      "feature": "age",
      "value": 1.0,
      "threshold": 0.0
    },
    {
      "name": "psi_drift",
      "metric": "psi",
      "severity": "warning",
      "message": "country: PSI drift detected (18.0152)",
      "feature": "country",
      "value": 18.01521528247136,
      "threshold": 0.2
    },
    {
      "name": "unseen_categories",
      "metric": "unseen_category_count",
      "severity": "warning",
      "message": "country: 1 unseen categories detected (e.g. ['DE'])",
      "feature": "country",
      "value": 1.0,
      "threshold": 0.0
    }
  ]
}
```

### Score

```python
print(report.score())
```

```json
{
  "overall_score": 77.5,
  "feature_scores": {
    "age": 85.0,
    "country": 70.0
  },
  "status": "warning",
  "system_issue_count": 0
}
```

Interpretation:

- 100 = clean data
- 80-99 = minor issues
- 50-79 = degraded data quality
- < 50 = severe issues

## Dataset validation in CI

```python
from mldebug import validate

report = validate(reference=train_df, current=prod_df)

score = report.score()["overall_score"]

if score < 80:
    raise SystemExit(report.summary())
```

```yaml
- name: Install mldebug
  run: pip install mldebug

- name: Run validation
  run: python validate_data.py
```

## Documentation

See [documentation pages](https://anpenta.github.io/mldebug).

## Status

Active development (v0.x). Core API is stable but may still evolve before v1.0.0.

See [CHANGELOG.md](https://github.com/anpenta/mldebug/blob/main/CHANGELOG.md) for version history.

## Development

### Setup

```bash
git clone https://github.com/anpenta/mldebug
cd mldebug
uv sync
```

### Commands

```bash
uv run poe test
uv run poe lint
uv run poe lint-check
```

## Contributing

We welcome contributions.

1. Clone the repository
2. Create a feature branch
3. Make your changes
4. Ensure all CI checks pass
5. Open a pull request

## Citation

If you use mldebug, please cite this project.

See [CITATION.cff](https://github.com/anpenta/mldebug/blob/main/CITATION.cff) or use GitHub's “Cite this repository” button.

## License

See [LICENSE](https://github.com/anpenta/mldebug/blob/main/LICENSE).
