from __future__ import annotations

from typing import Any, Mapping

from .rewards import AttitudeRewardCalculator
from .commands import VirtualPilotCommandSource


class ComponentRegistry:
    """受白名单约束的配置组件工厂。"""

    _reward_types = {
        "flight_train.rewards.attitude:AttitudeRewardCalculator": AttitudeRewardCalculator,
    }
    _command_source_types = {
        "flight_train.commands:VirtualPilotCommandSource": VirtualPilotCommandSource,
    }

    def build_reward(self, node: Mapping[str, Any]):
        type_name = str(node.get("type", ""))
        factory = self._reward_types.get(type_name)
        if factory is None:
            raise ValueError(f"reward calculator is not registered: {type_name}")
        configured_version = int(node.get("version", -1))
        if configured_version != factory.version:
            raise ValueError(
                f"reward calculator {type_name} requires version {factory.version}, "
                f"got {configured_version}"
            )
        return factory(node.get("params", {}))

    def build_entrypoint(self, node: Mapping[str, Any]):
        module_name, _, attribute = str(node["type"]).partition(":")
        if module_name != "flight_train.runner" or attribute != "run_experiment":
            raise ValueError("only flight_train.runner:run_experiment is registered")
        from .runner import run_experiment
        return run_experiment

    def build_command_source(
        self,
        config,
        batch_size,
        device,
        dtype,
        control_hz,
    ):
        factory = self._command_source_types.get(config.type)
        if factory is None:
            raise ValueError(f"command source is not registered: {config.type}")
        return factory(config, batch_size, device, dtype, control_hz)
