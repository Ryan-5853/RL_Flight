from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch

from ..errors import CheckpointError, UnsupportedCheckpointError
from ..model import ConvertedPolicy, FeedForwardPolicy
from .base import CheckpointAdapter


_LINEAR_KEY = re.compile(
    r"(?:^|\.)(?P<family>network|loc_network)\.(?P<index>\d+)\.weight$"
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{path} must be a mapping")
    return value


def _sequence(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise CheckpointError(f"{path} must be a sequence")
    return tuple(value)


class FlightTrainMLPAdapter(CheckpointAdapter):
    """Convert current Flight Train PPO/SAC MLP actors.

    Model dimensions are inferred from tensors rather than a fixed architecture,
    so arbitrary MLP depths and widths are accepted. The adapter recognizes both
    SAC ``network`` heads (loc + scale) and PPO ``loc_network`` heads.
    """

    name = "flight-train-mlp-v1"
    priority = 100

    def probe(self, checkpoint: Mapping[str, Any]) -> bool:
        actor = checkpoint.get("actor")
        config = checkpoint.get("config")
        if not isinstance(actor, Mapping) or not isinstance(config, Mapping):
            return False
        model = config.get("model")
        return isinstance(model, Mapping) and str(model.get("type", "")).startswith(
            "mlp_"
        )

    def describe(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        config = _mapping(checkpoint.get("config"), "config")
        model = _mapping(config.get("model"), "config.model")
        return {
            "adapter": self.name,
            "checkpoint_schema_version": checkpoint.get(
                "checkpoint_schema_version"
            ),
            "algorithm": checkpoint.get("algorithm_name"),
            "global_control_steps": checkpoint.get("global_control_steps"),
            "model_type": model.get("type"),
            "supported": str(model.get("type", "")).startswith("mlp_"),
        }

    def convert(
        self,
        checkpoint: Mapping[str, Any],
        *,
        source_path: Path,
    ) -> ConvertedPolicy:
        del source_path
        actor = _mapping(checkpoint.get("actor"), "actor")
        config = _mapping(checkpoint.get("config"), "config")
        model_config = _mapping(config.get("model"), "config.model")
        model_type = str(model_config.get("type", ""))
        if not model_type.startswith("mlp_"):
            raise UnsupportedCheckpointError(
                f"{self.name} supports feed-forward MLP actors, got {model_type!r}; "
                "register a stateful architecture adapter for GRU/RNN policies"
            )
        output_dim = self._output_dim(config)
        weights = self._linear_weights(actor)
        if not weights:
            raise CheckpointError("actor contains no recognized MLP linear weights")
        dimensions = [int(weights[0][1].shape[1])]
        previous = dimensions[0]
        for _, weight, bias in weights:
            if weight.ndim != 2 or bias.ndim != 1:
                raise CheckpointError("MLP weights and biases must be rank 2 and 1")
            if weight.shape[1] != previous or bias.shape[0] != weight.shape[0]:
                raise CheckpointError("MLP linear layer shapes are not composable")
            dimensions.append(int(weight.shape[0]))
            previous = int(weight.shape[0])
        stochastic_output = dimensions[-1]
        if stochastic_output == 2 * output_dim:
            dimensions[-1] = output_dim
            strip_distribution_head = True
        elif stochastic_output == output_dim:
            strip_distribution_head = False
        else:
            raise CheckpointError(
                f"actor output has {stochastic_output} values, expected "
                f"{output_dim} deterministic or {2 * output_dim} loc/scale values"
            )
        policy = FeedForwardPolicy(tuple(dimensions))
        target_linears = [
            module for module in policy.modules() if isinstance(module, torch.nn.Linear)
        ]
        with torch.no_grad():
            for index, (target, (_, weight, bias)) in enumerate(
                zip(target_linears, weights, strict=True)
            ):
                if index == len(target_linears) - 1 and strip_distribution_head:
                    weight = weight[:output_dim]
                    bias = bias[:output_dim]
                target.weight.copy_(weight)
                target.bias.copy_(bias)
        policy.eval()
        contract = self._contract(config, input_dim=dimensions[0], output_dim=output_dim)
        architecture = {
            "family": "feedforward_mlp",
            "dimensions": dimensions,
            "hidden_activation": "silu",
            "output_activation": "tanh",
            "source_distribution_head_removed": strip_distribution_head,
        }
        return ConvertedPolicy(
            module=policy,
            input_dim=dimensions[0],
            output_dim=output_dim,
            architecture=architecture,
            contract=contract,
            source_metadata={
                "checkpoint_schema_version": checkpoint.get(
                    "checkpoint_schema_version"
                ),
                "algorithm": checkpoint.get("algorithm_name"),
                "run_id": checkpoint.get("run_id"),
                "global_control_steps": checkpoint.get("global_control_steps"),
            },
        )

    @staticmethod
    def _linear_weights(
        actor: Mapping[str, Any],
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        matches: dict[str, list[tuple[int, str, torch.Tensor]]] = {}
        for key, value in actor.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                continue
            match = _LINEAR_KEY.search(key)
            if match:
                matches.setdefault(match.group("family"), []).append(
                    (int(match.group("index")), key, value)
                )
        family = "loc_network" if matches.get("loc_network") else "network"
        result: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for index, key, weight in sorted(matches.get(family, ())):
            bias_key = key[: -len("weight")] + "bias"
            bias = actor.get(bias_key)
            if not isinstance(bias, torch.Tensor):
                raise CheckpointError(f"actor bias is missing: {bias_key}")
            result.append((index, weight.detach().cpu(), bias.detach().cpu()))
        return result

    @staticmethod
    def _output_dim(config: Mapping[str, Any]) -> int:
        contract = _mapping(config.get("control_contract"), "config.control_contract")
        policy_action = _mapping(
            contract.get("policy_action"), "config.control_contract.policy_action"
        )
        fields = _sequence(
            policy_action.get("fields"),
            "config.control_contract.policy_action.fields",
        )
        if not fields:
            raise CheckpointError("policy action fields must not be empty")
        return len(fields)

    @staticmethod
    def _contract(
        config: Mapping[str, Any],
        *,
        input_dim: int,
        output_dim: int,
    ) -> Mapping[str, Any]:
        contract = dict(
            _mapping(config.get("control_contract"), "config.control_contract")
        )
        task = _mapping(config.get("task", {}), "config.task")
        termination = _mapping(task.get("termination", {}), "config.task.termination")
        command = _mapping(config.get("command_source", {}), "config.command_source")
        params = _mapping(command.get("params", {}), "config.command_source.params")
        sticks = _mapping(params.get("sticks", {}), "command_source.params.sticks")
        yaw = _mapping(sticks.get("yaw", {}), "command_source.params.sticks.yaw")
        contract["input_dim"] = input_dim
        contract["output_dim"] = output_dim
        contract["normalization"] = {
            "angular_velocity_rad_s": termination.get(
                "max_angular_rate_rad_s"
            ),
            "acceleration_m_s2": 9.80665,
            "motor_speed_rad_s": 1800.0,
            "servo_angle_rad": 0.5 * torch.pi,
            "desired_yaw_rate_rad_s": yaw.get("limit_rad_s"),
            "upper_throttle": "2*x-1",
        }
        history = _mapping(
            contract.get("observation_history"),
            "config.control_contract.observation_history",
        )
        mode = str(history.get("mode", ""))
        if mode == "uniform":
            expected_input_dim = 21 * int(history.get("frames", 0))
        elif mode == "multirate_actuator":
            expected_input_dim = (
                21
                + 4 * int(history.get("dense_action_steps", 0))
                + 11 * int(history.get("sparse_physical_frames", 0))
            )
        else:
            raise CheckpointError(
                f"unsupported Flight Train observation history mode: {mode!r}"
            )
        if expected_input_dim != input_dim:
            raise CheckpointError(
                "checkpoint actor input width does not match its control "
                f"contract: actor={input_dim}, contract={expected_input_dim}"
            )
        return contract
