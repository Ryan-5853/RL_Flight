from __future__ import annotations

from dataclasses import replace
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

    def test_optional_lqr_integral_augmentation_is_stable(self) -> None:
        controller = create_controller(
            {
                "type": "lqr",
                "params": {
                    "collective_mode": "hover",
                    "lqr": {
                        "integral_state_scales": [0.2, 0.2, 0.5],
                    },
                },
            },
            self.context,
        )
        self.assertTrue(controller._lqr_integral_enabled)
        self.assertEqual(tuple(controller._lqr_gain.shape), (5, 13))
        self.assertLess(max(abs(controller._lqr_poles)), 1.0)
        output = controller.step(
            self.state_at_trim(controller), self.reference()
        )
        self.assertEqual(
            tuple(output.diagnostics["controller.lqr_state"].shape),
            (2, 13),
        )

    def test_external_upper_lqr_has_four_outputs_and_pilot_owns_upper(self) -> None:
        controller = create_controller(
            {
                "type": "lqr",
                "params": {
                    "collective_mode": "external_upper",
                    "virtual_pilot": {
                        "throttle": {
                            "minimum": 0.20,
                            "maximum": 0.85,
                            "initial": 0.56,
                            "spool_duration_s": 0.0,
                        },
                        "height_controller": {
                            "target_m": 0.0,
                            "proportional_gain": 0.08,
                            "integral_gain": 0.04,
                            "error_limit_m": 5.0,
                        },
                    },
                    "pid": {"altitude": {}, "attitude": {}},
                    "lqr": {
                        "state_scales": [
                            0.0872665,
                            0.0872665,
                            1.0,
                            1.0,
                            2.0,
                            300.0,
                            300.0,
                            0.15,
                            0.15,
                            0.15,
                        ],
                        "integral_state_scales": [0.05, 0.05, 0.30],
                        "input_scales": [0.16, 0.40, 0.40, 0.40],
                        "input_weight_scale": 0.10,
                    },
                    "allocation": {
                        "input_weights": [1.5, 1.5, 1.0, 1.0, 1.0],
                        "damping": 1.0e-5,
                    },
                },
            },
            self.context,
        )
        self.assertTrue(controller.upper_external)
        self.assertEqual(tuple(controller._lqr_gain.shape), (4, 13))
        self.assertLess(max(abs(controller._lqr_poles)), 1.0)
        active = torch.ones(2, dtype=torch.bool)
        state = self.state_at_trim(controller)
        reference = self.reference()
        output = controller.step(state, reference, active)
        command = output.command
        self.assertEqual(tuple(command.shape), (2, 5))
        self.assertTrue((command[:, 0] >= 0.20).all())
        self.assertTrue((command[:, 0] <= 0.85).all())
        self.assertTrue(torch.isfinite(command).all())
        # A positive height error (vehicle below target) must raise the pilot's
        # upper throttle on the next cycle.
        below = replace(
            self.state_at_trim(controller),
            position_n=torch.tensor(
                [[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]], dtype=torch.float64
            ),
        )
        first = float(controller.pilot.upper_throttle[0, 0])
        controller.step(below, reference, active)
        self.assertGreater(float(controller.pilot.upper_throttle[0, 0]), first)

    def test_lqi_freezes_integral_while_actuator_command_is_saturated(self) -> None:
        controller = create_controller(
            {
                "type": "lqr",
                "params": {
                    "collective_mode": "hover",
                    "lqr": {"integral_state_scales": [0.2, 0.2, 0.5]},
                },
            },
            self.context,
        )
        state = self.state_at_trim(controller)
        half_angle = math.radians(80.0) / 2.0
        state.attitude_q_wb[:, 0] = math.cos(half_angle)
        state.attitude_q_wb[:, 1] = math.sin(half_angle)
        output = controller.step(state, self.reference())
        self.assertTrue(output.diagnostics["controller.lqr_saturated"].all())
        torch.testing.assert_close(
            controller.attitude_integral,
            torch.zeros_like(controller.attitude_integral),
        )

    def test_lqr_supports_one_scheduled_gain_per_vehicle(self) -> None:
        controller = create_controller(
            {"type": "lqr", "params": {"collective_mode": "hover"}},
            self.context,
        )
        state = self.state_at_trim(controller)
        state.angular_velocity_b[:, 0] = 0.2
        scheduled = controller._lqr_gain[None].expand(2, -1, -1).clone()
        scheduled[0].zero_()
        controller.schedule_lqr_gain(scheduled)
        output = controller.step(state, self.reference())
        delta = output.diagnostics["controller.lqr_delta_command"]
        torch.testing.assert_close(delta[0], torch.zeros(5, dtype=torch.float64))
        self.assertGreater(float(delta[1].abs().max()), 0.0)

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

    def test_three_axis_deployment_model_owns_coaxial_allocation(self) -> None:
        class AllocatedModel:
            def forward_control(
                model_self,
                state,
                reference,
                previous_action,
                recurrent_state,
                is_init,
            ):
                del state, reference, recurrent_state, is_init
                self.assertEqual(previous_action.shape, (2, 3))
                return torch.tensor(
                    [[0.25, 0.5, -0.25], [0.25, 0.5, -0.25]],
                    dtype=torch.float64,
                ), None

            def action_to_command(model_self, action, collective):
                del model_self
                lower = 0.95 * collective + 0.1 * action[:, :1]
                servos = torch.cat(
                    (action[:, 1:2], action[:, 2:3], -action[:, 1:].sum(1, keepdim=True)),
                    dim=1,
                )
                return torch.cat((collective, lower, servos), dim=1)

        controller = create_controller(
            {
                "type": "neural",
                "params": {
                    "output_mode": "coaxial_differential_cyclic_3"
                },
            },
            self.context,
            neural_model=AllocatedModel(),
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
            output.command[:, 2:].sum(1), torch.zeros(2, dtype=torch.float64)
        )
        torch.testing.assert_close(
            controller.previous_action,
            torch.tensor(
                [[0.25, 0.5, -0.25], [0.25, 0.5, -0.25]],
                dtype=torch.float64,
            ),
        )


if __name__ == "__main__":
    unittest.main()
