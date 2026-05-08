from enum import Enum


class Severity(Enum):
    """Severity level of a detected issue.

    INFO:
        Informational issue with no immediate impact.

    WARNING:
        Potential problem that should be reviewed.

    CRITICAL:
        Serious issue likely to affect model performance or reliability.

    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
