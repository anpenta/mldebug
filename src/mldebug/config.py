from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NumericCheckConfig:
    ks_alpha: float = 0.05
    missing_threshold: float = 0.1
    variance_drift_threshold: float = 2.0


@dataclass(frozen=True, slots=True)
class CategoricalCheckConfig:
    psi_threshold: float = 0.2
    missing_threshold: float = 0.1
