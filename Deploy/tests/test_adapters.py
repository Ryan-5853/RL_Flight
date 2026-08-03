from __future__ import annotations

import unittest

import torch

from flight_deploy.adapters import (
    AdapterRegistry,
    FlightTrainAngularAccelerationCascadeAdapter,
    FlightTrainMLPAdapter,
    default_registry,
)
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

    def test_allocated_angular_acceleration_uses_dedicated_adapter(self) -> None:
        checkpoint = flight_checkpoint((63, 16, 6), output_dim=3)
        checkpoint["config"]["task"] = {
            "outer_loop": {
                "type": "attitude_pid",
                "proportional_gain": [18.0, 18.0, 8.0],
                "integral_gain": [1.5, 1.5, 0.75],
                "derivative_gain": [7.0, 7.0, 4.0],
                "integral_limit_rad_s": [0.15, 0.15, 0.2],
                "max_angular_acceleration_rad_s2": [6.0, 6.0, 3.0],
            },
            "termination": {"max_angular_rate_rad_s": 6.0},
        }
        contract = checkpoint["config"]["control_contract"]
        contract.update(
            {
                "observation_profile": (
                    "angular_acceleration_allocated_inner_loop_21d_v2"
                ),
                "observation_history": {
                    "mode": "uniform",
                    "frames": 3,
                    "stride_steps": 1,
                },
                "action_transform": {
                    "type": "coaxial_differential_cyclic",
                    "lower_motor_upper_ratio": 0.947558738884,
                    "trim_command": [0.53537068747, 0.0, 0.0, 0.0],
                    "residual_scale": [0.12, 0.4, 0.4],
                },
                "policy_action": {
                    "fields": [
                        "lower_motor_differential",
                        "servo_cyclic_a",
                        "servo_cyclic_b",
                    ]
                },
                "external_action": {"fields": ["upper_motor"]},
                "simulator_command": {
                    "fields": [
                        "upper_motor",
                        "lower_motor",
                        "servo_1",
                        "servo_2",
                        "servo_3",
                    ]
                },
            }
        )

        adapter = default_registry().select(checkpoint)
        self.assertIsInstance(
            adapter, FlightTrainAngularAccelerationCascadeAdapter
        )
        converted = adapter.convert(
            checkpoint, source_path=None  # type: ignore[arg-type]
        )
        self.assertEqual(converted.input_dim, 63)
        self.assertEqual(converted.output_dim, 3)
        self.assertEqual(
            converted.contract["version"],
            "angular_acceleration_cascade_v1",
        )
        self.assertEqual(
            converted.contract["controller"]["outer_loop"][
                "proportional_gain"
            ],
            [18.0, 18.0, 8.0],
        )
        self.assertEqual(
            converted.contract["normalization"][
                "desired_angular_acceleration_rad_s2"
            ],
            [6.0, 6.0, 3.0],
        )
