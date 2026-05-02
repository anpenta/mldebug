from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckConfig:
    """Configuration for validation checks.

    Controls sensitivity thresholds used in feature-level validation.
    """

    ks_alpha: float = 0.05
    missing_threshold: float = 0.1
    psi_threshold: float = 0.2
