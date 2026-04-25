# mldebug

[![CI](https://github.com/anpenta/mldebug/actions/workflows/ci.yml/badge.svg)](https://github.com/anpenta/mldebug/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/anpenta/mldebug/graph/badge.svg?token=XPN447G4S9)](https://codecov.io/gh/anpenta/mldebug)

mldebug is a lightweight Python toolkit for debugging machine learning systems in production.

## Quick Start

```bash
git clone https://github.com/anpenta/mldebug
cd mldebug
direnv allow
poe test
```

## Status

Active development (v0.x, not yet stable).

## Features

- Debug ML systems in production environments
- Lightweight diagnostic utilities
- Designed for integration into modern ML pipelines
- Focus on observability and failure analysis

## Installation

```bash
pip install mldebug
```

## Example Usage

See [tests/test_public_api.py](tests/test_public_api.py) for real usage examples.

## Documentation

See [github pages](https://anpenta.github.io/mldebug).

## Development Setup

### Requirements

- [Ubuntu 24.04.4](https://releases.ubuntu.com/noble/) (recommended) or [WSL](https://ubuntu.com/desktop/wsl)
- [Git](https://git-scm.com/)
- [direnv](https://direnv.net/)

### Environment Setup

If not already done via [Quick Start](#quick-start):

```bash
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

CI runs multi-Python version testing and linting. All pull requests must pass the checks. Merging is blocked unless all checks pass.

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all CI checks pass
5. Open a pull request

## Dependency Management

Dependencies are [managed using uv](https://docs.astral.sh/uv/concepts/projects/dependencies/) and defined in [pyproject.toml](pyproject.toml).

## License

See [LICENSE](LICENSE).
