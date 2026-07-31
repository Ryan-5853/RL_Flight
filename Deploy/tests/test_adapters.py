from __future__ import annotations

import unittest

import torch

from flight_deploy.adapters import AdapterRegistry, FlightTrainMLPAdapter
from flight_deploy.errors import UnsupportedCheckpointError


def flight_checkpoint(
    dimensions: tuple[int, ...],
    *,
    algorithm: str = "sac",
    output_dim: int = 4,
) -> dict:
    actor: dict[str, torch.Tensor] = {}
    prefix = (
        "module.0.module.0.module.network"
        if algorithm == "sac"
        else "module.0.module.0.module.loc_network"
    )
    layer_index = 0
    for input_size, output_size in zip(dimensions, dimensions[1:]):
        actor[f"{prefix}.{layer_index}.weight"] = torch.randn(
            output_size, input_size
        )
        actor[f"{prefix}.{layer_index}.bias"] = torch.randn(output_size)
        layer_index += 2
    return {
        "checkpoint_schema_version": 5,
        "algorithm_name": algorithm,
        "run_id": "test-run",
        "global_control_steps": 123,
        "actor": actor,
        "config": {
            "model": {
                "type": "mlp_actor_critic",
                "hidden_sizes": list(dimensions[1:-1]),
            },
            "task": {
                "termination": {"max_angular_rate_rad_s": 8.0},
            },
            "command_source": {
                "params": {
                    "sticks": {
                        "yaw": {"limit_rad_s": 0.35},
                    }
                }
            },
            "control_contract": {
                "version": "test",
                "observation_profile": "test",
                "observation_history": {
                    "mode": "uniform",
                    "frames": 1,
                    "stride_steps": 1,
                },
                "action_transform": {
                    "type": "residual_around_trim",
                    "trim_command": [0.5, 0.0, 0.0, 0.0],
                    "residual_scale": [0.25, 1.0, 1.0, 1.0],
                },
                "policy_action": {
                    "fields": [f"action_{index}" for index in range(output_dim)]
                },
                "external_action": {"fields": ["external"]},
                "simulator_command": {
                    "fields": ["external"]
                    + [f"action_{index}" for index in range(output_dim)]
                },
            },
        },
    }


class FlightTrainAdapterTests(unittest.TestCase):
    def test_sac_distribution_head_is_removed(self) -> None:
        checkpoint = flight_checkpoint((21, 13, 9, 8))
        converted = FlightTrainMLPAdapter().convert(
            checkpoint, source_path=None  # type: ignore[arg-type]
        )
        self.assertEqual(converted.input_dim, 21)
        self.assertEqual(converted.output_dim, 4)
        self.assertEqual(converted.architecture["dimensions"], [21, 13, 9, 4])
        observation = torch.randn(5, 21)
        with torch.inference_mode():
            actual = converted.module(observation)
            actor = checkpoint["actor"]
            expected = torch.nn.functional.silu(
                torch.nn.functional.linear(
                    observation,
                    actor[
                        "module.0.module.0.module.network.0.weight"
                    ],
                    actor["module.0.module.0.module.network.0.bias"],
                )
            )
            expected = torch.nn.functional.silu(
                torch.nn.functional.linear(
                    expected,
                    actor[
                        "module.0.module.0.module.network.2.weight"
                    ],
                    actor["module.0.module.0.module.network.2.bias"],
                )
            )
            expected = torch.tanh(
                torch.nn.functional.linear(
                    expected,
                    actor[
                        "module.0.module.0.module.network.4.weight"
                    ][:4],
                    actor["module.0.module.0.module.network.4.bias"][:4],
                )
            )
        torch.testing.assert_close(actual, expected)

    def test_ppo_deterministic_head_accepts_arbitrary_depth(self) -> None:
        checkpoint = flight_checkpoint(
            (21, 17, 11, 7, 4), algorithm="ppo"
        )
        converted = FlightTrainMLPAdapter().convert(
            checkpoint, source_path=None  # type: ignore[arg-type]
        )
        self.assertEqual(
            converted.architecture["dimensions"], [21, 17, 11, 7, 4]
        )
        self.assertFalse(
            converted.architecture["source_distribution_head_removed"]
        )

    def test_registry_can_be_extended_and_reports_unknown(self) -> None:
        registry = AdapterRegistry((FlightTrainMLPAdapter(),))
        self.assertEqual(registry.names(), ("flight-train-mlp-v1",))
        with self.assertRaises(UnsupportedCheckpointError):
            registry.select({"unknown": True})
