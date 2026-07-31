from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from flight_controller import (
    ControllerContext,
    ControllerReference,
    ControllerState,
    create_controller,
)
from simenv.config import load_and_materialize


ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.materialized = load_and_materialize(
            SIM_CONFIG,
            2,
            torch.device("cpu"),
            torch.float64,
        )
        cls.context = ControllerContext(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float64,
            control_dt=1.0 / 500.0,
            parameters=cls.materialized.parameters,
        )

    def state_at_trim(self, controller) -> ControllerState:
        zeros = torch.zeros(2, 3, dtype=torch.float64)
        attitude = torch.zeros(2, 4, dtype=torch.float64)
        attitude[:, 0] = 1.0
        return ControllerState(
            position_n=zeros.clone(),
            velocity_n=zeros.clone(),
            attitude_q_wb=attitude,
            angular_velocity_b=zeros.clone(),
            linear_acceleration_n=zeros.clone(),
            motor_speed=controller.trim.motor_speed.clone(),
            servo_angle=zeros.clone(),
        )

    def reference(self) -> ControllerReference:
        zeros = torch.zeros(2, 3, dtype=torch.float64)
        attitude = torch.zeros(2, 4, dtype=torch.float64)
        attitude[:, 0] = 1.0
        return ControllerReference(
            target_position_n=zeros.clone(),
            target_velocity_n=zeros.clone(),
            target_attitude_q_wb=attitude,
            target_angular_velocity_b=zeros.clone(),
            collective_command=torch.zeros(2, 1, dtype=torch.float64),
        )

    def test_hover_trim_balances_weight_and_reaction_torque(self) -> None:
        controller = create_controller(
            {"type": "pid", "params": {"collective_mode": "hover"}},
            self.context,
        )
        wrench = controller.plant.steady_wrench(controller.trim.command)
        torch.testing.assert_close(
            wrench[:, 0],
            self.materialized.parameters["body.mass"] * 9.80665,
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            wrench[:, 1:],
            torch.zeros_like(wrench[:, 1:]),
            atol=2e-6,
            rtol=0,
        )

    def test_all_classical_controllers_share_bounded_physical_interface(self) -> None:
        for controller_type in ("pid", "lqr", "hybrid_pid_lqr"):
            with self.subTest(controller_type=controller_type):
                controller = create_controller(
                    {
                        "type": controller_type,
                        "params": {"collective_mode": "hover"},
                    },
                    self.context,
                )
                output = controller.step(
                    self.state_at_trim(controller), self.reference()
                )
                self.assertEqual(output.command.shape, (2, 5))
                self.assertTrue(torch.isfinite(output.command).all())
                self.assertTrue((output.command[:, :2] >= 0).all())
                self.assertTrue((output.command[:, :2] <= 1).all())
                self.assertTrue((output.command[:, 2:] >= -1).all())
                self.assertTrue((output.command[:, 2:] <= 1).all())

    def test_lqr_design_is_discrete_stable(self) -> None:
        controller = create_controller(
            {"type": "lqr", "params": {"collective_mode": "hover"}},
            self.context,
        )
        self.assertLess(max(abs(controller._lqr_poles)), 1.0)

    def test_masked_reset_only_changes_selected_hybrid_state(self) -> None:
        controller = create_controller(
            {"type": "hybrid_pid_lqr", "params": {"collective_mode": "hover"}},
            self.context,
        )
        controller.height_integral[:] = torch.tensor([1.0, 2.0])
        controller.attitude_integral[:] = 0.3
        controller.lqr_blend[:] = torch.tensor([0.8, 0.6])
        controller.reset(torch.tensor([True, False]))
        self.assertEqual(controller.height_integral.tolist(), [0.0, 2.0])
        self.assertAlmostEqual(controller.lqr_blend[0].item(), 0.0)
        self.assertAlmostEqual(controller.lqr_blend[1].item(), 0.6)
        self.assertTrue((controller.attitude_integral[0] == 0).all())
        self.assertTrue((controller.attitude_integral[1] == 0.3).all())

    def test_pid_corrects_small_roll_error_with_finite_command(self) -> None:
        controller = create_controller(
            {"type": "pid", "params": {"collective_mode": "hover"}},
            self.context,
        )
        state = self.state_at_trim(controller)
        angle = math.radians(5.0) / 2.0
        state.attitude_q_wb[:, 0] = math.cos(angle)
        state.attitude_q_wb[:, 1] = math.sin(angle)
        output = controller.step(state, self.reference())
        self.assertTrue(torch.isfinite(output.command).all())
        self.assertGreater(
            float(output.diagnostics["controller.desired_moment_b"][:, 0].abs().min()),
            0.0,
        )

    def test_plain_feedforward_network_uses_same_physical_command_interface(
        self,
    ) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(21, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 4),
            torch.nn.Tanh(),
        ).to(dtype=torch.float64)
        controller = create_controller(
            {"type": "neural", "params": {"output_mode": "residual_4"}},
            self.context,
            neural_model=model,
        )
        trim_controller = create_controller(
            {"type": "pid", "params": {"collective_mode": "hover"}},
            self.context,
        )
        output = controller.step(
            self.state_at_trim(trim_controller),
            ControllerReference(
                **{
                    **self.reference().__dict__,
                    "collective_command": torch.full(
                        (2, 1), 0.7, dtype=torch.float64
                    ),
                }
            ),
        )
        self.assertEqual(output.command.shape, (2, 5))
        torch.testing.assert_close(
            output.command[:, 0], torch.full((2,), 0.7, dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
