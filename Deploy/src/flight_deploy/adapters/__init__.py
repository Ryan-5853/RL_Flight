from importlib.metadata import entry_points

from .angular_acceleration_cascade import (
    FlightTrainAngularAccelerationCascadeAdapter,
)
from .base import CheckpointAdapter
from .flight_train import FlightTrainMLPAdapter
from .registry import AdapterRegistry


def default_registry() -> AdapterRegistry:
    registry = AdapterRegistry(
        (
            FlightTrainAngularAccelerationCascadeAdapter(),
            FlightTrainMLPAdapter(),
        )
    )
    # Third-party training stacks can contribute adapters without modifying
    # this package. An entry point may expose an adapter instance or class.
    for entry_point in entry_points(group="flight_deploy.adapters"):
        loaded = entry_point.load()
        adapter = loaded() if isinstance(loaded, type) else loaded
        if not isinstance(adapter, CheckpointAdapter):
            raise TypeError(
                f"entry point {entry_point.name!r} is not a CheckpointAdapter"
            )
        registry.register(adapter)
    return registry


__all__ = [
    "AdapterRegistry",
    "CheckpointAdapter",
    "FlightTrainAngularAccelerationCascadeAdapter",
    "FlightTrainMLPAdapter",
    "default_registry",
]
