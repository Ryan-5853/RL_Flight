from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
