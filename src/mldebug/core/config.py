from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NumericCheckConfig:
    """Configuration for numeric feature validation checks.

    Attributes
    ----------
    ks_alpha : float
        Significance level for KS test (numeric drift detection).

    missing_threshold : float
        Maximum allowed increase in fraction of missing values between reference and current numeric data.

    variance_drift_threshold : float
        Allowed variance change ratio between reference and current data. Drift if ratio > threshold or < 1/threshold.

    """

    ks_alpha: float = 0.05
    missing_threshold: float = 0.1
    variance_drift_threshold: float = 2.0


@dataclass(frozen=True, slots=True)
class CategoricalCheckConfig:
    """Configuration for categorical feature validation checks.

    Attributes
    ----------
    psi_threshold : float
        PSI threshold for categorical distribution drift detection.

    missing_threshold : float
        Maximum allowed increase in fraction of missing values between reference and current categorical data.

    """

    psi_threshold: float = 0.2
    missing_threshold: float = 0.1
