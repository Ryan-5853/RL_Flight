from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping

from .base import FlightController
from .classical import HybridPIDLQRController, LQRController, PIDController
from .neural import NeuralNetworkController
from .types import ControllerContext


ControllerFactory = Callable[..., FlightController]
_REGISTRY: dict[str, ControllerFactory] = {
    "pid": PIDController,
    "lqr": LQRController,
    "hybrid_pid_lqr": HybridPIDLQRController,
    "neural": NeuralNetworkController,
}


def register_controller(name: str, factory: ControllerFactory) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"controller type is already registered or invalid: {name!r}")
    _REGISTRY[name] = factory


def controller_types() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_controller(
    raw_config: Mapping[str, Any] | None,
    context: ControllerContext,
    *,
    neural_model: Any | None = None,
) -> FlightController:
    config = dict(raw_config or {})
    controller_type = str(config.pop("type", "neural"))
    params = config.pop("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("controller.params must be a mapping")
    merged = {**dict(params), **config}
    factory = _REGISTRY.get(controller_type)
    if factory is None and ":" in controller_type:
        module_name, attribute = controller_type.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
    if factory is None:
        raise ValueError(
            f"unknown controller type {controller_type!r}; available: {controller_types()}"
        )
    if controller_type == "neural" or factory is NeuralNetworkController:
        if neural_model is None:
            raise ValueError("neural controller requires a loaded model")
        return factory(context, merged, model=neural_model)
    return factory(context, merged)
