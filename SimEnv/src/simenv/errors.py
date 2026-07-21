class SimulationError(RuntimeError):
    """Base exception for simulation environment failures."""


class ConfigurationError(SimulationError, ValueError):
    """Raised when an environment configuration violates the interface contract."""


class EnvironmentClosedError(SimulationError):
    """Raised when a closed environment is accessed."""


class LoggingError(SimulationError):
    """Raised when the lossless timeline logger cannot persist data."""

