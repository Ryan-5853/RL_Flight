from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from flight_identification.config import load_experiment_config
from flight_identification.control_evaluation import (
    _command_slopes_for_thrust_scale,
    _nominal_actuator_model,
)
from flight_identification.experiment import (
    SIM2REAL_IDENTIFICATION_TARGET_NAMES,
    SIM2REAL_TARGET_NAMES,
    _apply_effectiveness_labels,
    _maximum_thrust_to_weight,
    _sample_parameter_labels,
    _sample_initial_state,
    _sim2real_lqr_targets,
)
from flight_identification.repeated_trial_deployment import (
    RepeatedTrialLQRIdentifier,
    RepeatedTrialLQRScheduler,
)
from flight_identification.offline_models import build_offline_identifier
from flight_identification.offline_log_inference import (
    _log_quality_metrics,
    _quality_gate,
    load_canonical_log_bundle,
)
from flight_identification.sim2real_control_evaluation import (
    _cluster_bootstrap_mean_ci,
    _paired_binary_summary,
)
from flight_identification.sim2real_composite import (
    composite_discrete_model,
    fit_composite_coefficients,
    reconstruct_composite_step_response,
    servo_mode_transform,
    true_servo_mode_step_response,
)
from flight_identification.sim2real_composite_evaluation import (
    _parameter_stratified_convergence,
)
from flight_identification.sim2real_bad_point_screening import _rank_bad_points
from flight_identification.sim2real_composite_training import _supervised_loss
from flight_identification.experiment import CommandDrivenServoObserver
from flight_identification.repeated_trial_training import (
    TrialSetIdentifier,
    _random_trial_mask,
    _retain_converged_trials,
)
from flight_identification.training import effective_lqr_labels
from simenv.config import load_and_materialize


ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "SimEnv" / "configs" / "example.yaml"
CONTROLLER_CONFIG = ROOT / "Controller" / "configs" / "lqr_identification_nominal.yaml"
EMPIRICAL_CONFIG = (
    ROOT / "Identification" / "configs" / "lqr_repeated_trials_empirical_core16_v1.yaml"
)
SIM2REAL_CONFIG = (
    ROOT / "Identification" / "configs" / "lqr_sim2real_micro_repeated16_v1.yaml"
)


class RepeatedTrialIdentifierTests(unittest.TestCase):
    def test_offline_quality_gate_separates_validation_from_flight_release(self) -> None:
        features = torch.zeros(4, 3, 8)
        names = (
            "angular_velocity_b.x",
            "angular_velocity_b.y",
            "angular_velocity_b.z",
            "command.motor_upper",
            "command.motor_lower",
            "command.servo_1",
            "command.servo_2",
            "command.servo_3",
        )
        features[:, :, :3] = 0.2
        features[:, 1:, 3:] = 0.1
        valid = torch.ones(4, 3, dtype=torch.bool)
        metrics = _log_quality_metrics(features, valid, names, features)
        calibration = {
            "artifact_type": "offline_log_quality_calibration",
            "minimum_selected_flights": 4,
            "thresholds": {
                "minimum_axis_rate_rms_rad_s": dict.fromkeys("xyz", 0.1),
                "minimum_command_movement_rms_per_sample": 0.05,
                "maximum_feature_z_score_abs_p95": 1.0,
                "maximum_feature_z_score_abs_max": 1.0,
            },
        }
        result = _quality_gate(metrics, 4, calibration)
        self.assertTrue(result["passed_for_independent_validation"])
        self.assertIn("never accepts a gain for flight", result["meaning"])
        failed = _quality_gate(metrics, 3, calibration)
        self.assertFalse(failed["passed_for_independent_validation"])
        self.assertIn("selected_flight_count_below_4", failed["reasons"])

    def test_parameter_stratification_finds_directional_control_effect(self) -> None:
        labels = torch.arange(8, dtype=torch.float32)[:, None]
        reference = {"converged": torch.zeros(8, dtype=torch.bool)}
        candidate = {
            "converged": torch.tensor(
                [False, False, False, False, True, True, True, True]
            )
        }
        result = _parameter_stratified_convergence(
            candidate, reference, labels, ("scale",), 4, 2
        )
        association = result["top_parameter_associations"][0]
        self.assertEqual(association["name"], "scale")
        self.assertGreater(
            association["pearson_correlation_with_group_convergence_delta"], 0.8
        )

    def test_offline_log_bundle_reorders_canonical_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.npz"
            np.savez(
                path,
                features=np.asarray([[[2.0, 1.0], [4.0, 3.0]]]),
                feature_names=np.asarray(["second", "first"]),
                sample_hz=np.asarray(500.0),
            )
            features, valid, rate = load_canonical_log_bundle(
                path, ("first", "second"), 500.0, 2
            )
        self.assertEqual(rate, 500.0)
        self.assertTrue(valid.all())
        self.assertTrue(
            torch.equal(features[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        )

    def test_offline_log_bundle_applies_checkpoint_integer_decimation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs.npz"
            np.savez(
                path,
                features=np.arange(10, dtype=np.float32).reshape(1, 5, 2),
                feature_names=np.asarray(["first", "second"]),
                sample_hz=np.asarray(500.0),
            )
            features, valid, rate = load_canonical_log_bundle(
                path, ("first", "second"), 100.0, 1
            )
        self.assertEqual(rate, 100.0)
        self.assertEqual(tuple(features.shape), (1, 1, 2))
        self.assertTrue(valid.all())

    def test_projected_response_loss_backpropagates_through_stable_basis(self) -> None:
        prediction = torch.randn(2, 45, requires_grad=True)
        target = torch.randn(2, 45)
        loss = _supervised_loss(
            prediction,
            target,
            torch.zeros(45),
            torch.ones(45),
            "step_response",
            3,
            0.35,
            0.5,
            1.0,
            torch.nn.HuberLoss(),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_stratified_multiaxis_initial_states_cover_each_rate_axis(self) -> None:
        config = load_experiment_config(SIM2REAL_CONFIG)
        config = replace(
            config,
            initial_state=replace(
                config.initial_state,
                sampling_design="stratified_multiaxis",
                maximum_angular_rate_rad_s=(1.0, 1.5, 2.0),
            ),
        )
        attitude, rate = _sample_initial_state(
            16, config, torch.Generator().manual_seed(22), torch.float64
        )
        self.assertEqual(tuple(attitude.shape), (16, 4))
        self.assertEqual(tuple(rate.shape), (16, 3))
        grouped = rate.reshape(2, 8, 3)
        self.assertTrue((grouped.amin(dim=1) < 0).all())
        self.assertTrue((grouped.amax(dim=1) > 0).all())
        self.assertTrue((rate.abs().amax(dim=0) <= torch.tensor([1.0, 1.5, 2.0])).all())

    def test_offline_temporal_identifiers_accept_masked_flight_sets(self) -> None:
        histories = torch.randn(3, 4, 40, 15)
        histories[..., -1] = 1.0
        histories[1, 3, :, -1] = 0.0
        trial_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False], [True, False, True, False]]
        )
        for architecture in ("tcn", "bigru"):
            model = build_offline_identifier(
                architecture,
                40,
                15,
                9,
                (32, 16),
                (24,),
                temporal_channels=(16, 24),
                recurrent_hidden_size=12,
                recurrent_layers=1,
                trial_embedding_size=20,
            )
            result = model(histories, trial_mask)
            self.assertEqual(tuple(result.shape), (3, 9))
            self.assertTrue(torch.isfinite(result).all())

    def test_composite_servo_basis_preserves_control_relevant_response(self) -> None:
        effectiveness = torch.zeros(1, 3, 5, dtype=torch.float64)
        effectiveness[0, :, 2:] = torch.tensor(
            [[8.0, -3.0, 2.0], [1.0, 7.0, -4.0], [2.0, 1.0, 5.0]],
            dtype=torch.float64,
        )
        target = torch.zeros(1, 26, dtype=torch.float64)
        target[:, :15] = effectiveness.reshape(1, 15)
        target[:, 20:23] = torch.log10(
            torch.tensor([[0.015, 0.040, 0.080]], dtype=torch.float64)
        )
        transform = servo_mode_transform(
            effectiveness[0].numpy(), torch.full((3,), 0.35).numpy()
        )
        coefficients = fit_composite_coefficients(
            target, transform, torch.full((3,), 0.35).numpy()
        )
        reconstructed = reconstruct_composite_step_response(coefficients)
        truth = true_servo_mode_step_response(
            target, transform, torch.full((3,), 0.35).numpy()
        )
        relative_error = (
            (reconstructed - truth).square().mean().sqrt()
            / truth.square().mean().sqrt()
        )
        self.assertLess(float(relative_error), 1e-4)
        torch.testing.assert_close(
            torch.from_numpy(transform @ transform.T),
            torch.eye(3, dtype=torch.float64),
        )

        a, b = composite_discrete_model(
            effectiveness[0, :, :2].numpy(),
            torch.tensor([1000.0, 1000.0]).numpy(),
            torch.tensor([0.04, 0.05]).numpy(),
            coefficients[0].numpy(),
            transform,
            torch.full((3,), 0.35).numpy(),
        )
        self.assertEqual(a.shape, (19, 19))
        self.assertEqual(b.shape, (19, 5))
        self.assertTrue(torch.isfinite(torch.from_numpy(a)).all())
        self.assertTrue(torch.isfinite(torch.from_numpy(b)).all())

    def test_bad_point_screening_requires_oracle_recovery_and_preserves_labels(self) -> None:
        group_ids = torch.tensor([10, 20, 30])
        labels = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.4, 0.7]])
        nominal = {
            "safe": torch.tensor([True, True, True, True, True, True]),
            "converged": torch.tensor([False, False, True, True, False, False]),
            "final_attitude_error": torch.tensor([0.3, 0.2, 0.01, 0.01, 0.4, 0.3]),
            "final_rate": torch.tensor([0.4, 0.3, 0.01, 0.01, 0.5, 0.4]),
            "saturation_fraction": torch.tensor([0.2, 0.3, 0.0, 0.0, 0.4, 0.5]),
        }
        oracle = {
            "safe": torch.ones(6, dtype=torch.bool),
            "converged": torch.ones(6, dtype=torch.bool),
            "final_attitude_error": torch.full((6,), 0.01),
            "final_rate": torch.full((6,), 0.01),
            "saturation_fraction": torch.full((6,), 0.05),
        }
        rows, counts = _rank_bad_points(
            group_ids=group_ids,
            labels=labels,
            label_names=("first", "second"),
            label_ranges=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            nominal_result=nominal,
            oracle_result=oracle,
            nominal_local_radius=(1.1, 0.9, 1.2),
            oracle_local_radius=(0.8, 0.8, 0.8),
            repeats=2,
            maximum_points=2,
            maximum_nominal_converged_fraction=0.25,
            minimum_oracle_converged_fraction=0.75,
            minimum_oracle_safe_fraction=1.0,
            minimum_convergence_gain=0.5,
        )
        self.assertEqual(counts["eligible_parameter_groups"], 2)
        self.assertEqual([row["group_id"] for row in rows], [10, 30])
        self.assertAlmostEqual(rows[0]["parameters"]["first"], 0.1)
        self.assertAlmostEqual(rows[0]["parameters"]["second"], 0.2)
        self.assertEqual(rows[0]["nominal_lqi"]["converged_fraction"], 0.0)
        self.assertEqual(rows[0]["oracle_lqi"]["converged_fraction"], 1.0)

    def test_paired_control_statistics_cluster_by_parameter_group(self) -> None:
        nominal = torch.tensor([True, True, False, False, True, False])
        candidate = torch.tensor([True, False, True, False, True, True])
        summary = _paired_binary_summary(candidate, nominal, 3, 2, 17)
        self.assertAlmostEqual(summary["fraction_delta"], 1.0 / 6.0)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(len(summary["group_cluster_bootstrap_95_ci"]), 2)

        interval = _cluster_bootstrap_mean_ci(
            torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0]), 3, 2, 18
        )
        self.assertLessEqual(interval[0], 0.0)
        self.assertGreaterEqual(interval[1], 0.0)

    def test_sim2real_sampling_is_mass_scaled_and_linearly_trimmable(self) -> None:
        config = load_experiment_config(SIM2REAL_CONFIG)
        materialized = load_and_materialize(
            config.simulator_config, 1, torch.device("cpu"), torch.float64
        )
        labels, _ = _sample_parameter_labels(
            32,
            materialized.parameters,
            config,
            torch.Generator().manual_seed(14),
            torch.float64,
        )
        actual = _apply_effectiveness_labels(
            {
                name: value.expand(32, *value.shape[1:]).clone()
                for name, value in materialized.parameters.items()
            },
            labels,
            config.parameterization,
        )
        targets = _sim2real_lqr_targets(actual)
        effectiveness = targets[:, :15].reshape(-1, 3, 5)[:, :, 2:]
        bias = targets[:, 15:18]
        trim = -torch.linalg.lstsq(
            effectiveness, bias.unsqueeze(-1)
        ).solution.squeeze(-1)
        radius_xy = torch.sqrt(labels[:, 1:3] / labels[:, :1])
        self.assertGreaterEqual(float(radius_xy.min()), 0.035)
        self.assertLessEqual(float(radius_xy.max()), 0.075)
        self.assertLessEqual(float(trim.abs().max()), 0.125 + 1e-8)

    def test_failed_trials_are_excluded_from_repeated_training(self) -> None:
        split = {
            "features": torch.randn(3, 3, 4, 2),
            "valid_mask": torch.ones(3, 3, 4, dtype=torch.bool),
            "labels": torch.randn(3, 2),
            "failure_code": torch.tensor([[0, 1, 2], [1, 1, 1], [1, 0, 0]]),
        }
        _retain_converged_trials(split)
        self.assertEqual(tuple(split["features"].shape), (2, 3, 4, 2))
        self.assertEqual(
            split["trial_mask"].tolist(), [[True, False, False], [False, True, True]]
        )
        self.assertFalse(split["valid_mask"][~split["trial_mask"]].any())
        selected = _random_trial_mask(split["trial_mask"])
        self.assertTrue((selected.sum(dim=1) >= 1).all())
        self.assertFalse((selected & ~split["trial_mask"]).any())

    def test_empirical_core_sampling_respects_physical_ranges(self) -> None:
        config = load_experiment_config(EMPIRICAL_CONFIG)
        materialized = load_and_materialize(
            config.simulator_config, 1, torch.device("cpu"), torch.float64
        )
        labels, _ = _sample_parameter_labels(
            512,
            materialized.parameters,
            config,
            torch.Generator().manual_seed(13),
            torch.float64,
        )
        nominal = {
            name: value.expand(512, *value.shape[1:]).clone()
            for name, value in materialized.parameters.items()
        }
        actual = _apply_effectiveness_labels(
            nominal, labels, config.parameterization
        )
        thrust_to_weight = _maximum_thrust_to_weight(actual)
        self.assertGreaterEqual(float(thrust_to_weight.min()), 1.2)
        self.assertLessEqual(float(thrust_to_weight.max()), 5.0)
        self.assertGreaterEqual(float(actual["motors.time_constant"].min()), 0.015)
        self.assertLessEqual(float(actual["motors.time_constant"].max()), 0.120)
        self.assertGreaterEqual(float(actual["servos.tau"].min()), 0.010)
        self.assertLessEqual(float(actual["servos.tau"].max()), 0.080)

    def test_empirical_effective_labels_include_hover_working_point(self) -> None:
        labels = torch.zeros(1, 13, dtype=torch.float64)
        labels[0, 0:3] = torch.tensor([0.1, 0.2, 0.3])
        labels[0, 3] = 0.4
        labels[0, 4] = 0.5
        labels[0, 7:10] = torch.tensor([0.6, 0.7, 0.8])
        effective = effective_lqr_labels(labels, "empirical_core")
        self.assertEqual(tuple(effective.shape), (1, 15))
        self.assertAlmostEqual(float(effective[0, 0]), 0.5)
        self.assertAlmostEqual(float(effective[0, 5]), 0.0)
        self.assertAlmostEqual(float(effective[0, 14]), 0.4)

    def test_thrust_scale_changes_hover_motor_command_slope(self) -> None:
        _, nominal_slopes, _ = _nominal_actuator_model(SIM_CONFIG)
        slopes = _command_slopes_for_thrust_scale(
            SIM_CONFIG, torch.tensor([1.0, 4.0]).numpy()
        )
        torch.testing.assert_close(
            torch.from_numpy(slopes[0]), torch.from_numpy(nominal_slopes)
        )
        self.assertFalse(torch.equal(
            torch.from_numpy(slopes[0, :2]), torch.from_numpy(slopes[1, :2])
        ))

    def test_command_servo_observer_models_deadzone_and_backlash(self) -> None:
        table = torch.tensor(
            [[[[-1.0, -0.35], [0.0, 0.0], [1.0, 0.35]]]],
            dtype=torch.float64,
        ).reshape(1, 1, 3, 2)
        observer = CommandDrivenServoObserver(
            table,
            torch.tensor([[0.02]], dtype=torch.float64),
            torch.tensor([[8.0]], dtype=torch.float64),
            torch.tensor([[0.01]], dtype=torch.float64),
            torch.tensor([[0.015]], dtype=torch.float64),
            0.002,
        )
        observer.advance(torch.tensor([[0.01]], dtype=torch.float64))
        torch.testing.assert_close(observer.angle, torch.zeros_like(observer.angle))
        observer.advance(torch.tensor([[0.1]], dtype=torch.float64))
        self.assertGreater(float(observer.angle[0, 0]), 0.0)
        observer.advance(torch.tensor([[-0.1]], dtype=torch.float64))
        self.assertAlmostEqual(float(observer.target_angle[0, 0]), -0.025, places=12)

    def test_trial_aggregation_is_permutation_invariant(self) -> None:
        torch.manual_seed(7)
        model = TrialSetIdentifier(10, 4, 6, (16, 8), (12,))
        histories = torch.randn(3, 5, 10, 4)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        torch.testing.assert_close(
            model(histories), model(histories[:, permutation]), rtol=1e-6, atol=1e-6
        )

    def test_trial_mask_ignores_excluded_histories(self) -> None:
        torch.manual_seed(8)
        model = TrialSetIdentifier(6, 3, 4, (12, 8), (10,))
        histories = torch.randn(2, 4, 6, 3)
        mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
        changed = histories.clone()
        changed[~mask] = 1e6
        torch.testing.assert_close(
            model(histories, mask), model(changed, mask), rtol=1e-6, atol=1e-6
        )

    def test_single_trial_backward_is_finite(self) -> None:
        torch.manual_seed(9)
        model = TrialSetIdentifier(6, 3, 4, (12, 8), (10,))
        histories = torch.randn(3, 4, 6, 3)
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask[:, 0] = True
        model(histories, mask).square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_runtime_consumes_raw_masked_repeated_trials(self) -> None:
        torch.manual_seed(10)
        model = TrialSetIdentifier(3, 4, 2, (8, 4), (6,))
        checkpoint = {
            "artifact_type": "repeated_trial_lqr_effective_identifier",
            "model_state": model.state_dict(),
            "normalization": {
                "feature_mean": torch.zeros(3),
                "feature_std": torch.ones(3),
                "label_mean": torch.zeros(2),
                "label_std": torch.ones(2),
            },
            "history_steps": 3,
            "feature_count": 4,
            "raw_feature_count": 3,
            "trials_per_group": 4,
            "downsample": 2,
            "trial_hidden_sizes": (8, 4),
            "head_hidden_sizes": (6,),
            "label_names": ("first", "second"),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identifier.pt"
            torch.save(checkpoint, path)
            runtime = RepeatedTrialLQRIdentifier(path)
            histories = torch.randn(2, 4, 6, 3)
            valid = torch.ones(2, 4, 6, dtype=torch.bool)
            prediction = runtime.identify(histories, valid)
            permutation = torch.tensor([2, 0, 3, 1])
            permuted = runtime.identify(
                histories[:, permutation], valid[:, permutation]
            )
        self.assertEqual(tuple(prediction.shape), (2, 2))
        torch.testing.assert_close(prediction, permuted, rtol=1e-6, atol=1e-6)

    def test_scheduler_synthesizes_finite_candidate_gain_and_servo_tau(self) -> None:
        torch.manual_seed(11)
        model = TrialSetIdentifier(3, 4, 14, (8, 4), (6,))
        checkpoint = {
            "artifact_type": "repeated_trial_lqr_effective_identifier",
            "model_state": model.state_dict(),
            "normalization": {
                "feature_mean": torch.zeros(3),
                "feature_std": torch.ones(3),
                "label_mean": torch.zeros(14),
                "label_std": torch.full((14,), 0.01),
            },
            "history_steps": 3,
            "feature_count": 4,
            "raw_feature_count": 3,
            "trials_per_group": 4,
            "downsample": 2,
            "trial_hidden_sizes": (8, 4),
            "head_hidden_sizes": (6,),
            "label_names": tuple(f"label_{index}" for index in range(14)),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identifier.pt"
            torch.save(checkpoint, path)
            scheduler = RepeatedTrialLQRScheduler(
                path, SIM_CONFIG, CONTROLLER_CONFIG
            )
            histories = torch.randn(2, 4, 6, 3)
            valid = torch.ones(2, 4, 6, dtype=torch.bool)
            result = scheduler.synthesize(histories, valid)
        self.assertFalse(scheduler.deployment_validated)
        self.assertEqual(tuple(result.effective_log10.shape), (2, 14))
        self.assertEqual(tuple(result.predicted_gain.shape), (2, 5, 10))
        self.assertEqual(tuple(result.servo_time_constant_s.shape), (2, 3))
        self.assertTrue(result.synthesis_valid.all())
        self.assertTrue(torch.isfinite(result.predicted_gain).all())
        self.assertTrue((result.servo_time_constant_s > 0).all())

    def test_sim2real_scheduler_synthesizes_augmented_lqi_gain(self) -> None:
        torch.manual_seed(15)
        config = load_experiment_config(SIM2REAL_CONFIG)
        materialized = load_and_materialize(
            config.simulator_config, 1, torch.device("cpu"), torch.float64
        )
        full_target = _sim2real_lqr_targets(materialized.parameters)[0].to(torch.float32)
        indices = [
            SIM2REAL_TARGET_NAMES.index(name)
            for name in SIM2REAL_IDENTIFICATION_TARGET_NAMES
        ]
        target = full_target[indices]
        model = TrialSetIdentifier(3, 4, len(target), (8, 4), (6,))
        checkpoint = {
            "artifact_type": "repeated_trial_lqr_effective_identifier",
            "model_state": model.state_dict(),
            "normalization": {
                "feature_mean": torch.zeros(3),
                "feature_std": torch.ones(3),
                "label_mean": target,
                "label_std": torch.full_like(target, 1e-8),
            },
            "history_steps": 3,
            "feature_count": 4,
            "raw_feature_count": 3,
            "trials_per_group": 4,
            "downsample": 2,
            "trial_hidden_sizes": (8, 4),
            "head_hidden_sizes": (6,),
            "label_names": SIM2REAL_IDENTIFICATION_TARGET_NAMES,
            "parameterization": "sim2real_micro",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identifier.pt"
            torch.save(checkpoint, path)
            scheduler = RepeatedTrialLQRScheduler(
                path, config.simulator_config, config.controller_config
            )
            histories = torch.randn(2, 4, 6, 3)
            valid = torch.ones(2, 4, 6, dtype=torch.bool)
            result = scheduler.synthesize(histories, valid)
        self.assertEqual(tuple(result.effective_log10.shape), (2, 24))
        self.assertEqual(tuple(result.predicted_gain.shape), (2, 5, 13))
        self.assertEqual(tuple(result.trim_angular_acceleration_bias.shape), (2, 3))
        self.assertTrue(result.synthesis_valid.all())
        self.assertTrue(torch.isfinite(result.predicted_gain).all())

    def test_empirical_scheduler_uses_fifteenth_thrust_scale_label(self) -> None:
        torch.manual_seed(12)
        model = TrialSetIdentifier(3, 4, 15, (8, 4), (6,))
        checkpoint = {
            "artifact_type": "repeated_trial_lqr_effective_identifier",
            "parameterization": "empirical_core",
            "model_state": model.state_dict(),
            "normalization": {
                "feature_mean": torch.zeros(3),
                "feature_std": torch.ones(3),
                "label_mean": torch.zeros(15),
                "label_std": torch.full((15,), 0.01),
            },
            "history_steps": 3,
            "feature_count": 4,
            "raw_feature_count": 3,
            "trials_per_group": 4,
            "downsample": 2,
            "trial_hidden_sizes": (8, 4),
            "head_hidden_sizes": (6,),
            "label_names": tuple(f"label_{index}" for index in range(15)),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identifier.pt"
            torch.save(checkpoint, path)
            scheduler = RepeatedTrialLQRScheduler(
                path, SIM_CONFIG, CONTROLLER_CONFIG
            )
            result = scheduler.synthesize(
                torch.randn(2, 4, 6, 3),
                torch.ones(2, 4, 6, dtype=torch.bool),
            )
        self.assertEqual(tuple(result.effective_log10.shape), (2, 15))
        self.assertEqual(tuple(result.thrust_to_weight_scale.shape), (2,))
        self.assertTrue((result.thrust_to_weight_scale > 0).all())
        self.assertTrue(result.synthesis_valid.all())


if __name__ == "__main__":
    unittest.main()
