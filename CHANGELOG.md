# Changelog

All notable changes to this project will be documented in this file.

---

## [0.8.0] - 2026-05-30

### Added
- Add `highest_severity()`, `has_critical()`, and `is_clean()` methods to `Report`
- Add schema inference convenience function
- Add 1d feature value array validation
- Expose version through `__version__` dunder

## [0.7.0] - 2026-05-15

### Added
- Modification of public api to `validate()` instead of `run_checks()`
- Deterministic sorting of detected issues in report

## [0.6.0] - 2026-05-11

### Added
- Scoring functionality exposed as Report.score()
- Improved and unified feature type inference
- Input validation for strict structure and type checks of input args

## [0.5.0] - 2026-05-07

### Added
- FeatureType enum exposed to the user to replace Literal
- Drop support for python 3.10
- Range value check for numeric features

## [0.4.0] - 2026-05-05

### Added
- Type checks based on schema and actual values found

## [0.3.0] - 2026-05-02

### Added
- Variance drift check for numeric features
- Unseen value check for categorical features
- Small scale performance testing (memory and time)

## [0.2.0] - 2026-04-30

### Added
- Standardized message format
- Citation functionality
- Introspection function for available checks
- Categorical missing value check

## [0.1.0] - 2026-04-29

### Added
- Initial public release
- KS test for numeric drift detection
- PSI for categorical drift detection
- Missing value checks for numeric features
- Schema validation and dataset mismatching system
- Structured `Issue` and `Report` outputs
