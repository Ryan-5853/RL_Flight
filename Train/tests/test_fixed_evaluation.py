from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from flight_train.config import load_experiment_config
from flight_train.evaluation import (
    FixedScenario,
    ScoreLimits,
    ScriptedEvaluationCommandSource,
    _command_step_response_time,
    _control_quality_metrics,
    _create_evaluation_environment,
    _create_packed_evaluation_environment,
    _run_packed_scenarios,
    _run_scenario,
    _scenario_command,
    _scenario_trajectory,
    _score_trajectory,
    load_fixed_evaluation_suite,
)
from flight_train.commands import VirtualPilotCommandSource
from flight_train.math import euler_to_quaternion
from flight_train.models import build_sac_actor_critic


ROOT = Path(__file__).parents[1]


class FixedEvaluationTests(unittest.TestCase):
    def test_control_quality_reports_tv_and_low_frequency_rms(self):
        control_hz = 100
        steps = 1000
        time_s = torch.arange(steps, dtype=torch.float32) / control_hz
        action = torch.zeros(steps, 1, 4)
        action[:, 0, 1] = 0.001 * torch.arange(steps)
        action[:, 0, 2] = -0.001 * torch.arange(steps)
        roll_pitch_error = torch.zeros(steps, 1, 2)
        roll_pitch_error[:, 0, 0] = 0.1 * torch.sin(
            2.0 * torch.pi * time_s
        )
        metrics = _control_quality_metrics(
            {"action": action},
            torch.ones(steps, 1, dtype=torch.bool),
            roll_pitch_error,
            control_hz,
        )

        self.assertAlmostEqual(
            metrics["motor_total_variation_per_s"].item(), 0.0
        )
        self.assertAlmostEqual(
            metrics["servo_common_total_variation_per_s"].item(), 0.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["servo_cyclic_total_variation_per_s"].item(), 0.2,
            places=4,
        )
        self.assertAlmostEqual(
            metrics["actuator_energy_proxy_per_s"].item(), 0.2,
            places=4,
        )
        expected_cyclic_effort = (
            action[..., 1:4]
            - action[..., 1:4].mean(dim=-1, keepdim=True)
        ).square().sum(dim=-1).mean()
        self.assertAlmostEqual(
            metrics["actuator_effort_proxy_mean"].item(),
            expected_cyclic_effort.item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics[
                "roll_pitch_error_band_0_5_2_hz_rms_deg"
            ].item(),
            math.degrees(0.1 / math.sqrt(2.0)),
            places=3,
        )

    def test_suite_parses_three_required_subjects(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_attitude_v1.yaml"
        )
        self.assertEqual(
            [item.type for item in suite.scenarios],
            ["hover", "constant_translation", "circle"],
        )
        self.assertAlmostEqual(sum(suite.score_weights.values()), 1.0)

    def test_self_stabilize_suite_scores_yaw_rate_not_yaw_heading(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_self_stabilize_v2.yaml"
        )
        self.assertTrue(suite.self_stabilize_tracking)
        self.assertEqual(suite.schema_version, 2)
        self.assertEqual(
            suite.checkpoint_selection.minimum_hover_survival_s, 2.5
        )

        steps, batch = 100, 2
        zeros_3 = torch.zeros(steps, batch, 3)
        target_q = euler_to_quaternion(
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
        )
        yaw_offset = torch.full((steps, batch), torch.pi / 2.0)
        actual_euler = zeros_3.clone()
        actual_euler[..., 2] = yaw_offset
        actual_q = euler_to_quaternion(
            actual_euler[..., 0],
            actual_euler[..., 1],
            actual_euler[..., 2],
        )
        trajectory = {
            "alive": torch.ones(steps, batch, dtype=torch.bool),
            "action": torch.zeros(steps, batch, 4),
            "position_n": zeros_3.clone(),
            "velocity_n": zeros_3.clone(),
            "attitude_q_wb": actual_q,
            "target_position_n": zeros_3.clone(),
            "target_velocity_n": zeros_3.clone(),
            "target_attitude_q_wb": target_q,
            "target_euler_rad": zeros_3.clone(),
            "actual_euler_rad": actual_euler,
            "yaw_rate_error_rad_s": torch.zeros(steps, batch),
        }
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), 10.0),
            suite.scenarios[0],
            suite.limits,
            100,
            suite.score_weights,
            self_stabilize_tracking=True,
        )
        self.assertGreater(result["metrics"]["attitude_rmse_deg"]["mean"], 89.0)
        self.assertAlmostEqual(
            result["metrics"]["roll_pitch_rmse_deg"]["mean"], 0.0
        )
        self.assertAlmostEqual(
            result["metrics"]["yaw_rate_rmse_rad_s"]["mean"], 0.0
        )
        self.assertAlmostEqual(result["total_score"], 100.0)

    def test_upright_survival_suite_records_but_does_not_score_yaw_rate(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_upright_survival_v3.yaml"
        )
        self.assertEqual(suite.schema_version, 3)
        self.assertTrue(suite.self_stabilize_tracking)
        self.assertEqual(suite.yaw_rate_tracking_weight, 0.0)
        self.assertIsNone(
            suite.checkpoint_selection.maximum_hover_yaw_rate_rmse_rad_s
        )

        steps, batch = 100, 2
        zeros_3 = torch.zeros(steps, batch, 3)
        target_q = euler_to_quaternion(
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
        )
        trajectory = {
            "alive": torch.ones(steps, batch, dtype=torch.bool),
            "action": torch.zeros(steps, batch, 4),
            "position_n": zeros_3.clone(),
            "velocity_n": zeros_3.clone(),
            "attitude_q_wb": target_q.clone(),
            "target_position_n": zeros_3.clone(),
            "target_velocity_n": zeros_3.clone(),
            "target_attitude_q_wb": target_q,
            "target_euler_rad": zeros_3.clone(),
            "actual_euler_rad": zeros_3.clone(),
            "yaw_rate_error_rad_s": torch.full((steps, batch), 10.0),
        }
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), 10.0),
            suite.scenarios[0],
            suite.limits,
            100,
            suite.score_weights,
            self_stabilize_tracking=suite.self_stabilize_tracking,
            yaw_rate_tracking_weight=suite.yaw_rate_tracking_weight,
        )
        self.assertAlmostEqual(
            result["metrics"]["yaw_rate_rmse_rad_s"]["mean"], 10.0
        )
        self.assertAlmostEqual(result["total_score"], 100.0)

    def test_command_tracking_suite_scripts_explicit_yaw_rate_steps(self):
        config = load_experiment_config(
            ROOT
            / "configs/experiments/"
            "mlp_sac_attitude_command_tracking_v1.yaml"
        )
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_small_command_tracking_v1.yaml"
        )
        self.assertEqual(suite.schema_version, 4)
        self.assertEqual(len(suite.scenarios), 8)
        scenario = next(
            item for item in suite.scenarios
            if item.name == "yaw_rate_full_step"
        )
        time_s = torch.tensor([0.0, 2.4, 5.4, 8.4])
        roll, pitch, yaw, yaw_rate = _scenario_command(scenario, time_s)
        torch.testing.assert_close(roll, torch.zeros_like(time_s))
        torch.testing.assert_close(pitch, torch.zeros_like(time_s))
        torch.testing.assert_close(
            yaw_rate,
            torch.tensor([0.0, 0.35, -0.35, 0.0]),
        )
        torch.testing.assert_close(
            yaw,
            torch.tensor([0.0, 0.0, 1.05, 0.0]),
        )

        base = VirtualPilotCommandSource(
            config.command_source,
            1,
            torch.device("cpu"),
            torch.float32,
            10,
        )
        scripted = ScriptedEvaluationCommandSource(
            base,
            scenario,
            10,
        )
        scripted.elapsed_steps.fill_(24)
        scripted._apply_target()
        torch.testing.assert_close(
            scripted.desired_yaw_rate,
            torch.tensor([[0.35]]),
        )
        torch.testing.assert_close(
            base.filtered_stick[:, 2],
            torch.ones(1),
        )

    def test_command_step_tracking_score_does_not_require_position_control(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_small_command_tracking_v1.yaml"
        )
        scenario = next(
            item for item in suite.scenarios if item.name == "roll_step"
        )
        steps, batch = 100, 2
        zeros_3 = torch.zeros(steps, batch, 3)
        target_q = euler_to_quaternion(
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
        )
        trajectory = {
            "alive": torch.ones(steps, batch, dtype=torch.bool),
            "action": torch.zeros(steps, batch, 4),
            "position_n": torch.full((steps, batch, 3), 100.0),
            "velocity_n": torch.full((steps, batch, 3), 100.0),
            "attitude_q_wb": target_q.clone(),
            "target_position_n": zeros_3.clone(),
            "target_velocity_n": zeros_3.clone(),
            "target_attitude_q_wb": target_q,
            "target_euler_rad": zeros_3.clone(),
            "actual_euler_rad": zeros_3.clone(),
            "yaw_rate_error_rad_s": torch.zeros(steps, batch),
        }
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), scenario.duration_s),
            scenario,
            suite.limits,
            100,
            suite.score_weights,
            self_stabilize_tracking=True,
            yaw_rate_tracking_weight=suite.yaw_rate_tracking_weight,
        )
        self.assertAlmostEqual(result["scores"]["tracking"]["mean"], 100.0)

    def test_command_step_response_starts_at_each_transition(self):
        suite = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_small_command_tracking_v1.yaml"
        )
        scenario = next(
            item for item in suite.scenarios if item.name == "roll_step"
        )
        control_hz = 10
        steps = round(scenario.duration_s * control_hz)
        error = torch.zeros(steps, 1)
        # 三次跳变后分别用 0.3s、0.5s、0.2s 收敛；最坏响应应为 0.5s。
        for fraction, delay_steps in (
            (0.20, 3),
            (0.45, 5),
            (0.70, 2),
        ):
            start = round(fraction * scenario.duration_s * control_hz)
            error[start : start + delay_steps] = 5.0
        response = _command_step_response_time(
            error,
            torch.ones_like(error, dtype=torch.bool),
            scenario,
            control_hz,
            threshold=2.0,
            window_s=0.5,
        )
        torch.testing.assert_close(response, torch.tensor([0.5]))

    def test_circle_reference_starts_at_origin_with_tangent_velocity(self):
        scenario = load_fixed_evaluation_suite(
            ROOT / "configs/evaluation/fixed_attitude_v1.yaml"
        ).scenarios[2]
        time = torch.zeros(2)
        origin = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        position, velocity = _scenario_trajectory(scenario, time, origin)
        torch.testing.assert_close(position, origin)
        expected_speed = 2.0 * torch.pi * scenario.circle_radius_m / scenario.circle_period_s
        torch.testing.assert_close(velocity[:, 0], torch.full((2,), expected_speed))
        torch.testing.assert_close(velocity[:, 1:], torch.zeros(2, 2))

    def test_perfect_hover_scores_one_hundred(self):
        steps, batch = 20, 2
        zeros_3 = torch.zeros(steps, batch, 3)
        target_q = euler_to_quaternion(
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
            torch.zeros(steps, batch),
        )
        trajectory = {
            "alive": torch.ones(steps, batch, dtype=torch.bool),
            "action": torch.zeros(steps, batch, 4),
            "position_n": zeros_3.clone(),
            "velocity_n": zeros_3.clone(),
            "attitude_q_wb": target_q.clone(),
            "target_position_n": zeros_3.clone(),
            "target_velocity_n": zeros_3.clone(),
            "target_attitude_q_wb": target_q,
            "target_euler_rad": zeros_3.clone(),
            "actual_euler_rad": zeros_3.clone(),
        }
        limits = ScoreLimits(15.0, 5.0, 3.0, 1.0, 0.25, 0.5, 2.0, 2.0, 0.02, 1.0)
        scenario = FixedScenario(
            "hover", "hover", 0.2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            1.0, 8.0, 0.0, 0.0,
        )
        result = _score_trajectory(
            trajectory,
            torch.full((batch,), 0.2),
            scenario,
            limits,
            100,
            {"survival": 0.35, "tracking": 0.35, "action": 0.15, "response": 0.15},
        )
        self.assertAlmostEqual(result["total_score"], 100.0)

    def test_unknown_suite_field_is_rejected(self):
        source = (ROOT / "configs/evaluation/fixed_attitude_v1.yaml").read_text()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "bad.yaml"
            path.write_text(source + "\nunknown_field: true\n")
            with self.assertRaisesRegex(ValueError, "unknown evaluation suite"):
                load_fixed_evaluation_suite(path)

    def test_packed_fast_path_matches_serial_scenarios(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            simulator = json.loads(
                (ROOT / "configs/environment/sim_smoke.json").read_text()
            )
            simulator["logging"]["directory"] = str(root / "simlogs")
            simulator_path = root / "simulator.json"
            simulator_path.write_text(json.dumps(simulator), encoding="utf-8")
            config = replace(
                load_experiment_config(
                    ROOT / "configs/experiments/mlp_sac_smoke.json"
                ),
                simulator_config=simulator_path,
            )
            source_suite = load_fixed_evaluation_suite(
                ROOT / "configs/evaluation/fixed_self_stabilize_v2.yaml"
            )
            durations = (0.004, 0.006, 0.008)
            suite = replace(
                source_suite,
                parallel_count=2,
                scenarios=tuple(
                    replace(scenario, duration_s=duration)
                    for scenario, duration in zip(
                        source_suite.scenarios,
                        durations,
                    )
                ),
            )
            model = build_sac_actor_critic(
                config.control_contract.observation_dim,
                4,
                config.model,
                torch.device("cpu"),
                config.torch_dtype,
            )

            serial = {}
            for scenario in suite.scenarios:
                env = _create_evaluation_environment(
                    config,
                    suite,
                    scenario,
                    torch.device("cpu"),
                )
                try:
                    serial[scenario.name] = _run_scenario(
                        env,
                        model,
                        scenario,
                        suite.limits,
                        suite.score_weights,
                        self_stabilize_tracking=(
                            suite.self_stabilize_tracking
                        ),
                        yaw_rate_tracking_weight=(
                            suite.yaw_rate_tracking_weight
                        ),
                    )
                finally:
                    env.close()

            packed_env = _create_packed_evaluation_environment(
                config,
                suite,
                torch.device("cpu"),
            )
            reset_calls = 0
            step_calls = 0
            original_reset = packed_env.simulator.reset
            original_step = packed_env.step_without_reset

            def counted_reset(*args, **kwargs):
                nonlocal reset_calls
                reset_calls += 1
                return original_reset(*args, **kwargs)

            def counted_step(*args, **kwargs):
                nonlocal step_calls
                step_calls += 1
                return original_step(*args, **kwargs)

            packed_env.simulator.reset = counted_reset
            packed_env.step_without_reset = counted_step
            try:
                packed = _run_packed_scenarios(
                    packed_env,
                    model,
                    suite,
                )
            finally:
                packed_env.close()

            self.assertEqual(reset_calls, 1)
            self.assertEqual(step_calls, 4)
            for scenario in suite.scenarios:
                serial_result, serial_trajectory = serial[scenario.name]
                packed_result, packed_trajectory = packed[scenario.name]
                self.assertAlmostEqual(
                    serial_result["total_score"],
                    packed_result["total_score"],
                    places=5,
                )
                self.assertEqual(
                    serial_result["metrics"]["survival_time_s"],
                    packed_result["metrics"]["survival_time_s"],
                )
                self.assertEqual(
                    set(serial_trajectory),
                    set(packed_trajectory),
                )
                for name in serial_trajectory:
                    torch.testing.assert_close(
                        packed_trajectory[name],
                        serial_trajectory[name],
                    )


if __name__ == "__main__":
    unittest.main()
