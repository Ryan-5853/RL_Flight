from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .control_evaluation import (
    _command_slopes_for_thrust_scale,
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
)
from .gain_training import _lqr_weights
from .repeated_trial_training import TrialSetIdentifier


@dataclass(frozen=True)
class RepeatedTrialLQRSynthesis:
    effective_log10: torch.Tensor
    synthesis_valid: torch.Tensor
    nominal_gain: torch.Tensor
    predicted_gain: torch.Tensor
    servo_time_constant_s: torch.Tensor
    thrust_to_weight_scale: torch.Tensor
    trim_angular_acceleration_bias: torch.Tensor


class RepeatedTrialLQRIdentifier:
    """One-shot effective-model identification from repeated flight trials."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint = torch.load(
            Path(checkpoint_path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if (
            self.checkpoint.get("artifact_type")
            != "repeated_trial_lqr_effective_identifier"
        ):
            raise ValueError("expected a repeated-trial LQR identifier checkpoint")
        self.label_names: Sequence[str] = tuple(self.checkpoint["label_names"])
        self.downsample = int(self.checkpoint["downsample"])
        self.history_steps = int(self.checkpoint["history_steps"])
        self.raw_feature_count = int(self.checkpoint["raw_feature_count"])
        self.model = TrialSetIdentifier(
            self.history_steps,
            int(self.checkpoint["feature_count"]),
            len(self.label_names),
            tuple(self.checkpoint["trial_hidden_sizes"]),
            tuple(self.checkpoint["head_hidden_sizes"]),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()
        self.normalization = {
            name: value.to(self.device)
            for name, value in self.checkpoint["normalization"].items()
        }

    @torch.no_grad()
    def identify(
        self,
        histories: torch.Tensor,
        valid_mask: torch.Tensor,
        trial_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return effective log10 parameters for `[batch, trials, time, feature]`."""

        if histories.ndim != 4:
            raise ValueError(
                "histories must have shape [batch, trials, time, feature_count]"
            )
        batch_size, trial_count, raw_steps, feature_count = histories.shape
        expected_raw_steps = self.history_steps * self.downsample
        if raw_steps != expected_raw_steps or feature_count != self.raw_feature_count:
            raise ValueError(
                f"expected raw history shape [batch, trials, {expected_raw_steps}, "
                f"{self.raw_feature_count}]"
            )
        if valid_mask.shape != (batch_size, trial_count, raw_steps):
            raise ValueError("valid_mask must match history batch, trial, and time axes")
        value = histories[:, :, :: self.downsample].to(
            self.device, dtype=torch.float32
        )
        valid = valid_mask[:, :, :: self.downsample].to(
            self.device, dtype=torch.bool
        )
        normalized = (
            value - self.normalization["feature_mean"]
        ) / self.normalization["feature_std"]
        normalized = torch.where(
            valid[..., None], normalized, torch.zeros_like(normalized)
        )
        model_input = torch.cat(
            (normalized, valid[..., None].to(normalized.dtype)), dim=3
        )
        if trial_mask is None:
            selected_trials = valid.any(dim=2)
        else:
            if trial_mask.shape != (batch_size, trial_count):
                raise ValueError("trial_mask must have shape [batch, trials]")
            selected_trials = trial_mask.to(self.device, dtype=torch.bool)
            if bool((selected_trials & ~valid.any(dim=2)).any().item()):
                raise ValueError("trial_mask cannot select a trial with no valid samples")
        prediction = self.model(model_input, selected_trials)
        effective = (
            prediction * self.normalization["label_std"]
            + self.normalization["label_mean"]
        )
        return effective.to(histories.device, dtype=histories.dtype)


class RepeatedTrialLQRScheduler(RepeatedTrialLQRIdentifier):
    """Identify once and synthesize candidate gains without an acceptance gate."""

    deployment_validated = False

    def __init__(
        self,
        checkpoint_path: str | Path,
        simulator_config: str | Path = "SimEnv/configs/example.yaml",
        controller_config: str | Path = "Controller/configs/lqr_identification_nominal.yaml",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(checkpoint_path, device)
        self.parameterization = str(
            self.checkpoint.get("parameterization", "legacy_collective")
        )
        expected = {
            "legacy_collective": 14,
            "empirical_core": 15,
            "sim2real_micro": 24,
        }.get(self.parameterization)
        if expected is None:
            raise ValueError(f"unsupported parameterization {self.parameterization!r}")
        if len(self.label_names) != expected:
            raise ValueError(
                f"{self.parameterization} LQR synthesis requires {expected} labels"
            )
        self.simulator_config = Path(simulator_config).expanduser().resolve()
        (
            self.nominal_effectiveness,
            self.command_slopes,
            self.nominal_tau,
        ) = _nominal_actuator_model(self.simulator_config)
        self.q, self.r = _lqr_weights(
            Path(controller_config).expanduser().resolve()
        )
        nominal_a, nominal_b = _discrete_model(
            self.nominal_effectiveness,
            self.command_slopes,
            self.nominal_tau,
            1.0 / 500.0,
            integral=self.q.shape[0] == 13,
        )
        self.nominal_gain_numpy = _lqr_gain(
            nominal_a, nominal_b, self.q, self.r
        )

    def synthesize(
        self,
        histories: torch.Tensor,
        valid_mask: torch.Tensor,
        trial_mask: torch.Tensor | None = None,
    ) -> RepeatedTrialLQRSynthesis:
        effective = self.identify(histories, valid_mask, trial_mask)
        labels = effective.detach().to(torch.float64).cpu().numpy()
        if self.parameterization == "sim2real_micro":
            if "label_min" in self.checkpoint and "label_max" in self.checkpoint:
                labels = np.clip(
                    labels,
                    self.checkpoint["label_min"].numpy(),
                    self.checkpoint["label_max"].numpy(),
                )
            return self._synthesize_sim2real(effective, labels, histories)
        labels[:, :9] = np.clip(labels[:, :9], -6.0, 6.0)
        labels[:, 9:] = np.clip(labels[:, 9:], -2.0, 2.0)
        thrust_scale = (
            np.power(10.0, labels[:, 14])
            if self.parameterization == "empirical_core"
            else np.ones(len(labels), dtype=np.float64)
        )
        command_slopes = _command_slopes_for_thrust_scale(
            self.simulator_config, thrust_scale
        )
        gains = []
        valid = []
        for sample_index, label in enumerate(labels):
            try:
                effectiveness, tau = _scaled_actuator_model(
                    label[:14], self.nominal_effectiveness, self.nominal_tau
                )
                a, b = _discrete_model(
                    effectiveness,
                    command_slopes[sample_index],
                    tau,
                    1.0 / 500.0,
                    integral=self.q.shape[0] == 13,
                )
                gain = _lqr_gain(a, b, self.q, self.r)
                current_valid = bool(np.isfinite(gain).all())
            except (np.linalg.LinAlgError, ValueError):
                gain = self.nominal_gain_numpy
                current_valid = False
            gains.append(gain if current_valid else self.nominal_gain_numpy)
            valid.append(current_valid)
        output_device = histories.device
        output_dtype = histories.dtype
        return RepeatedTrialLQRSynthesis(
            effective_log10=effective,
            synthesis_valid=torch.as_tensor(valid, device=output_device),
            nominal_gain=torch.as_tensor(
                self.nominal_gain_numpy,
                device=output_device,
                dtype=output_dtype,
            ),
            predicted_gain=torch.as_tensor(
                np.stack(gains), device=output_device, dtype=output_dtype
            ),
            servo_time_constant_s=torch.as_tensor(
                self.nominal_tau[None, 2:]
                * np.power(10.0, labels[:, 11:14]),
                device=output_device,
                dtype=output_dtype,
            ),
            thrust_to_weight_scale=torch.as_tensor(
                thrust_scale,
                device=output_device,
                dtype=output_dtype,
            ),
            trim_angular_acceleration_bias=torch.zeros(
                len(labels), 3, device=output_device, dtype=output_dtype
            ),
        )

    def _synthesize_sim2real(
        self,
        effective: torch.Tensor,
        labels: np.ndarray,
        histories: torch.Tensor,
    ) -> RepeatedTrialLQRSynthesis:
        from simenv.config import load_and_materialize
        from .experiment import _maximum_thrust_to_weight

        nominal = load_and_materialize(
            self.simulator_config, 1, torch.device("cpu"), torch.float64
        ).parameters
        nominal_twr = float(_maximum_thrust_to_weight(nominal)[0])
        gains = []
        valid = []
        servo_tau = []
        for label in labels:
            try:
                predicted = dict(zip(self.label_names, label))
                effectiveness = self.nominal_effectiveness.copy()
                for row, axis in enumerate(("roll", "pitch", "yaw")):
                    for column, actuator in enumerate(
                        ("motor_upper", "motor_lower", "servo_1", "servo_2", "servo_3")
                    ):
                        name = f"angular_acceleration_effectiveness.{axis}.{actuator}"
                        if name in predicted:
                            effectiveness[row, column] = predicted[name]
                tau = np.power(
                    10.0,
                    np.asarray(
                        [
                            predicted["log10.motor_tau_upper_s"],
                            predicted["log10.motor_tau_lower_s"],
                            predicted["log10.servo_tau_1_s"],
                            predicted["log10.servo_tau_2_s"],
                            predicted["log10.servo_tau_3_s"],
                        ]
                    ),
                )
                slopes = np.concatenate(
                    (
                        np.power(
                            10.0,
                            np.asarray(
                                [
                                    predicted["log10.command_slope.motor_upper"],
                                    predicted["log10.command_slope.motor_lower"],
                                ]
                            ),
                        ),
                        self.command_slopes[2:],
                    )
                )
                if (
                    not np.isfinite(effectiveness).all()
                    or not np.isfinite(tau).all()
                    or not np.isfinite(slopes).all()
                    or np.any(tau < 0.005)
                    or np.any(tau > 0.25)
                    or np.any(slopes <= 0.0)
                ):
                    raise ValueError("predicted actuator model is outside safety bounds")
                a, b = _discrete_model(
                    effectiveness,
                    slopes,
                    tau,
                    1.0 / 500.0,
                    integral=True,
                )
                gain = _lqr_gain(a, b, self.q, self.r)
                current_valid = bool(np.isfinite(gain).all())
            except (np.linalg.LinAlgError, ValueError):
                gain = self.nominal_gain_numpy
                current_valid = False
                tau = self.nominal_tau
            gains.append(gain if current_valid else self.nominal_gain_numpy)
            valid.append(current_valid)
            servo_tau.append(tau[2:])
        output_device = histories.device
        output_dtype = histories.dtype
        name_to_index = {name: index for index, name in enumerate(self.label_names)}
        thrust_to_weight = np.power(
            10.0, labels[:, name_to_index["log10.maximum_thrust_to_weight"]]
        )
        trim_bias = effective.new_zeros(len(labels), 3)
        trim_bias[:, 0] = effective[
            :, name_to_index["trim_angular_acceleration_bias.roll"]
        ]
        trim_bias[:, 1] = effective[
            :, name_to_index["trim_angular_acceleration_bias.pitch"]
        ]
        return RepeatedTrialLQRSynthesis(
            effective_log10=effective,
            synthesis_valid=torch.as_tensor(valid, device=output_device),
            nominal_gain=torch.as_tensor(
                self.nominal_gain_numpy, device=output_device, dtype=output_dtype
            ),
            predicted_gain=torch.as_tensor(
                np.stack(gains), device=output_device, dtype=output_dtype
            ),
            servo_time_constant_s=torch.as_tensor(
                np.stack(servo_tau), device=output_device, dtype=output_dtype
            ),
            thrust_to_weight_scale=torch.as_tensor(
                thrust_to_weight / nominal_twr,
                device=output_device,
                dtype=output_dtype,
            ),
            trim_angular_acceleration_bias=trim_bias,
        )
