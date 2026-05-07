# Changelog

All notable changes to this project will be documented in this file.

---

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
