from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from inference_package import (
    InferenceModelAdapter,
    InferencePackageMetadata,
    load_flight_deploy_package,
)


class _FakePackage:
    metadata = InferencePackageMetadata(
        format_version=1,
        package_id="fake-v1",
        observation_dim=21,
        action_dim=4,
        output_mode="residual_4",
        recurrent=False,
    )

    def infer(self, observation, recurrent_state, is_init):
        del recurrent_state, is_init
        return observation[:, :4].clone(), None

    def reset(self):
        return None

    def warmup(self, observation):
        self.infer(
            observation,
            None,
            torch.ones((1, 1), dtype=torch.bool),
        )

    def close(self):
        return None

    def describe(self):
        return {"package_id": self.metadata.package_id}


class InferencePackageTests(unittest.TestCase):
    def test_adapter_enforces_deployment_shape_and_forwards_action(self) -> None:
        adapter = InferenceModelAdapter(_FakePackage())
        observation = torch.arange(21, dtype=torch.float32)[None]

        action, recurrent = adapter.forward_step(observation)

        torch.testing.assert_close(action, observation[:, :4])
        self.assertIsNone(recurrent)

    def test_metadata_rejects_training_shape_mismatch(self) -> None:
        metadata = InferencePackageMetadata(
            format_version=1,
            package_id="bad",
            observation_dim=20,
            action_dim=4,
            output_mode="residual_4",
        )
        with self.assertRaisesRegex(ValueError, "observation_dim=21"):
            metadata.validate()

    def test_default_loader_consumes_flight_deploy_bundle(self) -> None:
        from flight_deploy import BundleBuilder

        prefix = "module.0.module.0.module.network"
        checkpoint = {
            "checkpoint_schema_version": 5,
            "algorithm_name": "sac",
            "run_id": "webui-test",
            "global_control_steps": 1,
            "actor": {
                f"{prefix}.0.weight": torch.randn(8, 63),
                f"{prefix}.0.bias": torch.randn(8),
                f"{prefix}.2.weight": torch.randn(8, 8),
                f"{prefix}.2.bias": torch.randn(8),
            },
            "config": {
                "model": {
                    "type": "mlp_actor_critic",
                    "hidden_sizes": [8],
                },
                "task": {
                    "termination": {"max_angular_rate_rad_s": 8.0}
                },
                "command_source": {
                    "params": {
                        "sticks": {
                            "yaw": {"limit_rad_s": 0.35}
                        }
                    }
                },
                "control_contract": {
                    "version": "self_stabilize_v1",
                    "observation_profile": "attitude_self_stabilize_21d_v3",
                    "observation_history": {
                        "mode": "uniform",
                        "frames": 3,
                        "stride_steps": 1,
                    },
                    "action_transform": {
                        "type": "residual_around_trim",
                        "trim_command": [0.5, 0.0, 0.0, 0.0],
                        "residual_scale": [0.25, 1.0, 1.0, 1.0],
                    },
                    "policy_action": {
                        "fields": ["a0", "a1", "a2", "a3"]
                    },
                    "external_action": {"fields": ["external"]},
                    "simulator_command": {
                        "fields": [
                            "external",
                            "a0",
                            "a1",
                            "a2",
                            "a3",
                        ]
                    },
                    "normalization": {
                        "angular_velocity_rad_s": 8.0,
                        "acceleration_m_s2": 9.80665,
                        "motor_speed_rad_s": 1800.0,
                        "servo_angle_rad": 1.5707963267948966,
                        "desired_yaw_rate_rad_s": 0.35,
                        "upper_throttle": "2*x-1",
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            bundle_path = root / "bundle"
            torch.save(checkpoint, checkpoint_path)
            BundleBuilder().build(
                checkpoint_path,
                bundle_path,
                verify_checksum=False,
            )

            package = load_flight_deploy_package(
                bundle_path, torch.device("cpu"), torch.float32
            )
            action, state = package.infer(
                torch.zeros((1, 21)),
                None,
                torch.ones((1, 1), dtype=torch.bool),
            )

            self.assertEqual(package.metadata.output_mode, "residual_4")
            self.assertEqual(package.metadata.observation_dim, 21)
            self.assertEqual(action.shape, (1, 4))
            self.assertIsNone(state)
            package.infer(
                torch.ones((1, 21)),
                None,
                torch.zeros((1, 1), dtype=torch.bool),
            )
            torch.testing.assert_close(
                package.history.observation(),
                torch.cat(
                    (
                        torch.zeros((1, 42)),
                        torch.ones((1, 21)),
                    ),
                    dim=1,
                ),
            )
            torch.testing.assert_close(
                package.action_to_command(
                    torch.zeros((1, 4)),
                    torch.full((1, 1), 0.7),
                ),
                torch.tensor([[0.7, 0.5, 0.0, 0.0, 0.0]]),
            )

    def test_cascade_bundle_is_directly_usable_by_webui_controller(self) -> None:
        from flight_deploy import BundleBuilder

        prefix = "module.0.module.0.module.network"
        checkpoint = {
            "checkpoint_schema_version": 5,
            "algorithm_name": "sac",
            "run_id": "cascade-webui-test",
            "global_control_steps": 1,
            "actor": {
                f"{prefix}.0.weight": torch.zeros(8, 63),
                f"{prefix}.0.bias": torch.zeros(8),
                f"{prefix}.2.weight": torch.zeros(6, 8),
                f"{prefix}.2.bias": torch.zeros(6),
            },
            "config": {
                "model": {"type": "mlp_actor_critic", "hidden_sizes": [8]},
                "task": {
                    "outer_loop": {
                        "type": "attitude_pid",
                        "proportional_gain": [18.0, 18.0, 8.0],
                        "integral_gain": [1.5, 1.5, 0.75],
                        "derivative_gain": [7.0, 7.0, 4.0],
                        "integral_limit_rad_s": [0.15, 0.15, 0.2],
                        "max_angular_acceleration_rad_s2": [6.0, 6.0, 3.0],
                    },
                    "termination": {"max_angular_rate_rad_s": 6.0},
                },
                "command_source": {
                    "params": {"sticks": {"yaw": {"limit_rad_s": 0.35}}}
                },
                "control_contract": {
                    "version": "self_stabilize_v1",
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
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoint.pt"
            bundle_path = root / "bundle"
            torch.save(checkpoint, checkpoint_path)
            bundle = BundleBuilder().build(
                checkpoint_path, bundle_path, verify_checksum=False
            )
            self.assertEqual(
                bundle.manifest["adapter"],
                "flight-train-angular-acceleration-cascade-v1",
            )
            package = load_flight_deploy_package(
                bundle_path, torch.device("cpu"), torch.float32
            )
            self.assertEqual(
                package.metadata.output_mode,
                "coaxial_differential_cyclic_3",
            )
            zeros3 = torch.zeros((1, 3))
            identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            angle = torch.tensor(0.05)
            target = torch.stack(
                (
                    torch.cos(angle),
                    torch.sin(angle),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                )
            )[None]
            state = SimpleNamespace(
                attitude_q_wb=identity,
                angular_velocity_b=zeros3.clone(),
                linear_acceleration_n=zeros3.clone(),
                motor_speed=torch.zeros((1, 2)),
                servo_angle=zeros3.clone(),
            )
            reference = SimpleNamespace(
                target_attitude_q_wb=target,
                target_angular_velocity_b=zeros3.clone(),
                collective_command=torch.tensor([[0.7]]),
            )
            action, recurrent = package.infer_control(
                state,
                reference,
                zeros3,
                None,
                torch.ones((1, 1), dtype=torch.bool),
            )
            self.assertEqual(action.shape, (1, 3))
            self.assertIsNone(recurrent)
            self.assertGreater(
                float(package.last_desired_angular_acceleration[0, 0]), 0.0
            )
            self.assertIn(
                "controller.angular_acceleration_error_b",
                package.control_diagnostics(),
            )
            command = package.action_to_command(action, torch.tensor([[0.7]]))
            self.assertEqual(command.shape, (1, 5))
            self.assertAlmostEqual(
                float(command[0, 1]), 0.7 * 0.947558738884, places=6
            )
            self.assertAlmostEqual(float(command[0, 2:].sum()), 0.0, places=6)
            package.reset()
            self.assertFalse(bool(package.estimator_initialized[0]))
            self.assertFalse(bool(package.history.initialized[0]))


if __name__ == "__main__":
    unittest.main()
