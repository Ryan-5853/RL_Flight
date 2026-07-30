class SimulationError(RuntimeError):
    """Base exception for simulation environment failures."""


class ConfigurationError(SimulationError, ValueError):
    """Raised when an environment configuration violates the interface contract."""


class EnvironmentClosedError(SimulationError):
    """Raised when a closed environment is accessed."""


class LoggingError(SimulationError):
    """Raised when the lossless timeline logger cannot persist data."""


class InsufficientDiskSpaceError(LoggingError):
    """Raised before a log write would consume the configured safety reserve."""

    def __init__(
        self,
        path: str,
        *,
        available_bytes: int,
        required_bytes: int,
        reserve_bytes: int,
        operation: str,
    ) -> None:
        self.path = path
        self.available_bytes = available_bytes
        self.required_bytes = required_bytes
        self.reserve_bytes = reserve_bytes
        self.operation = operation
        super().__init__(
            f"insufficient disk space for {operation}: path={path}, "
            f"available={available_bytes} bytes, required={required_bytes} bytes "
            f"(including reserve={reserve_bytes} bytes)"
        )
