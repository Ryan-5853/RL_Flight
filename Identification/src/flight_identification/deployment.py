from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .control_evaluation import (
    _discrete_model,
    _lqr_gain,
    _nominal_actuator_model,
    _scaled_actuator_model,
)
from .ensemble import calibrated_std
from .ensemble_control_evaluation import (
    EFFECTIVE_LOWER,
    EFFECTIVE_UPPER,
    _classifier_probabilities,
    _uncertainty_plants,
)
from .training import create_identifier_model


@dataclass(frozen=True)
class ScheduledLQRResult:
    effective_mean_log10: torch.Tensor
    effective_std_log10: torch.Tensor
    stability_probability: torch.Tensor
    synthesis_valid: torch.Tensor
    gate_candidate: torch.Tensor
    accepted: torch.Tensor
    nominal_gain: torch.Tensor
    predicted_gain: torch.Tensor
    target_gain: torch.Tensor


class AdaptiveLQRScheduler:
    """Inference and guarded LQR synthesis for a frozen deployment artifact."""

    def __init__(
        self,
        artifact_path: str | Path,
        simulator_config: str | Path = "SimEnv/configs/example.yaml",
        controller_config: str | Path = "Controller/configs/lqr_identification_nominal.yaml",
        device: str | torch.device = "cpu",
        allow_unvalidated_gate: bool = False,
    ) -> None:
        from flight_controller import load_controller_config

        self.device = torch.device(device)
        self.artifact = torch.load(
            Path(artifact_path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if self.artifact.get("artifact_type") != "flight_identification_deployment":
            raise ValueError("expected a flight identification deployment artifact")
        self.gate: Mapping[str, Any] = self.artifact["gain_gate"]
        self.gate_enabled = bool(self.gate.get("enabled", False))
        if allow_unvalidated_gate:
            self.gate_enabled = True
        self.models = []
        self.normalizations = []
        for checkpoint in self.artifact["members"]:
            model = create_identifier_model(
                str(checkpoint.get("architecture", "mlp")),
                int(checkpoint["history_steps"]),
                int(checkpoint["feature_count"]),
                len(checkpoint["label_names"]),
                tuple(checkpoint["hidden_sizes"]),
                float(checkpoint.get("dropout", 0.0)),
                int(checkpoint.get("tcn_channels", 128)),
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.models.append(model)
            self.normalizations.append(
                {
                    name: value.to(self.device)
                    for name, value in checkpoint["normalization"].items()
                }
            )
        loaded_controller = load_controller_config(
            Path(controller_config).expanduser().resolve()
        )
        lqr = loaded_controller["params"]["lqr"]
        state_scales = np.asarray(lqr["state_scales"], dtype=np.float64)
        input_scales = np.asarray(lqr["input_scales"], dtype=np.float64)
        self.q = np.diag(1.0 / state_scales**2)
        self.r = float(lqr["input_weight_scale"]) * np.diag(
            1.0 / input_scales**2
        )
        (
            self.nominal_effectiveness,
            self.command_slopes,
            self.nominal_tau,
        ) = _nominal_actuator_model(Path(simulator_config).expanduser().resolve())
        nominal_a, nominal_b = _discrete_model(
            self.nominal_effectiveness,
            self.command_slopes,
            self.nominal_tau,
            1.0 / 500.0,
        )
        self.nominal_gain_numpy = _lqr_gain(
            nominal_a, nominal_b, self.q, self.r
        )

    @torch.no_grad()
    def _identify(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or history.shape[2] != int(self.artifact["feature_count"]):
            raise ValueError("history must have shape [batch, time, feature_count]")
        downsample = int(self.artifact["downsample"])
        expected_raw_steps = int(self.artifact["history_steps"]) * downsample
        if history.shape[1] != expected_raw_steps:
            raise ValueError(
                f"history must contain exactly {expected_raw_steps} raw steps"
            )
        value = history[:, ::downsample].to(self.device, dtype=torch.float32)
        members = []
        for model, normalization in zip(
            self.models, self.normalizations, strict=True
        ):
            normalized = (
                value - normalization["feature_mean"]
            ) / normalization["feature_std"]
            prediction = model(normalized)
            members.append(
                prediction * normalization["label_std"]
                + normalization["label_mean"]
            )
        member_predictions = torch.stack(members).cpu()
        mean = member_predictions.mean(dim=0)
        standard_deviation = calibrated_std(
            member_predictions, self.artifact["calibration"]
        )
        return mean, standard_deviation

    def _synthesize(
        self, mean: np.ndarray, standard_deviation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        clipped = np.clip(mean, EFFECTIVE_LOWER, EFFECTIVE_UPPER)
        try:
            effectiveness, tau = _scaled_actuator_model(
                clipped, self.nominal_effectiveness, self.nominal_tau
            )
            predicted_a, predicted_b = _discrete_model(
                effectiveness, self.command_slopes, tau, 1.0 / 500.0
            )
            predicted_gain = _lqr_gain(predicted_a, predicted_b, self.q, self.r)
        except (np.linalg.LinAlgError, ValueError):
            feature_count = 3 * len(clipped) + len(
                self.gate["uncertainty_multipliers"]
            ) + 1
            return (
                self.nominal_gain_numpy.copy(),
                np.full(feature_count, np.inf, dtype=np.float32),
                np.full(
                    len(self.gate["uncertainty_multipliers"]),
                    np.inf,
                    dtype=np.float64,
                ),
                False,
            )
        blend = float(self.gate["gain_blend"])
        scheduled_gain = self.nominal_gain_numpy + blend * (
            predicted_gain - self.nominal_gain_numpy
        )
        robust_radii = []
        for multiplier in self.gate["uncertainty_multipliers"]:
            closed_loops = []
            for uncertain_label in _uncertainty_plants(
                clipped, standard_deviation, float(multiplier)
            ):
                uncertain_effectiveness, uncertain_tau = _scaled_actuator_model(
                    uncertain_label, self.nominal_effectiveness, self.nominal_tau
                )
                uncertain_a, uncertain_b = _discrete_model(
                    uncertain_effectiveness,
                    self.command_slopes,
                    uncertain_tau,
                    1.0 / 500.0,
                )
                closed_loops.append(uncertain_a - uncertain_b @ scheduled_gain)
            robust_radii.append(
                float(np.max(np.abs(np.linalg.eigvals(np.stack(closed_loops)))))
            )
        relative_step = np.linalg.norm(
            predicted_gain - self.nominal_gain_numpy
        ) / np.linalg.norm(self.nominal_gain_numpy)
        features = np.concatenate(
            (
                clipped,
                standard_deviation,
                np.abs(clipped),
                np.asarray(robust_radii),
                np.asarray([relative_step]),
            )
        ).astype(np.float32)
        valid = bool(
            np.isfinite(predicted_gain).all()
            and np.isfinite(features).all()
            and np.isfinite(robust_radii).all()
        )
        return predicted_gain, features, np.asarray(robust_radii), valid

    def schedule(self, history: torch.Tensor) -> ScheduledLQRResult:
        mean, standard_deviation = self._identify(history)
        predicted_gains = []
        risk_features = []
        robust_radii = []
        synthesis_valid = []
        for current_mean, current_std in zip(mean.numpy(), standard_deviation.numpy()):
            try:
                predicted_gain, features, current_radii, current_valid = self._synthesize(
                    current_mean, current_std
                )
            except (np.linalg.LinAlgError, ValueError):
                predicted_gain = self.nominal_gain_numpy.copy()
                current_radii = np.full(
                    len(self.gate["uncertainty_multipliers"]), np.inf
                )
                features = np.full(int(self.gate.get("input_count", 1)), np.inf)
                current_valid = False
            predicted_gains.append(predicted_gain)
            risk_features.append(features)
            robust_radii.append(current_radii)
            synthesis_valid.append(current_valid)
        predicted_gain_numpy = np.stack(predicted_gains)
        if self.gate.get("gate_type") == "physical_robust_radius":
            radii = np.stack(robust_radii)[:, 0]
            candidate = radii <= float(self.gate["robust_radius_threshold"])
            probabilities = np.clip(1.0 - radii, 0.0, 1.0)
        else:
            probabilities = _classifier_probabilities(
                self.gate,
                np.nan_to_num(
                    np.stack(risk_features), nan=0.0, posinf=1e6, neginf=-1e6
                ),
            )
            candidate = probabilities >= float(self.gate["probability_threshold"])
        valid = np.asarray(synthesis_valid, dtype=bool)
        probabilities = np.where(valid, probabilities, 0.0)
        candidate &= valid
        accepted = candidate & self.gate_enabled
        blend = float(self.gate["gain_blend"])
        scheduled = self.nominal_gain_numpy[None] + blend * (
            predicted_gain_numpy - self.nominal_gain_numpy[None]
        )
        target = np.where(
            accepted[:, None, None], scheduled, self.nominal_gain_numpy[None]
        )
        output_device = history.device
        output_dtype = history.dtype
        nominal = torch.as_tensor(
            self.nominal_gain_numpy, device=output_device, dtype=output_dtype
        )
        return ScheduledLQRResult(
            effective_mean_log10=mean.to(output_device, output_dtype),
            effective_std_log10=standard_deviation.to(output_device, output_dtype),
            stability_probability=torch.as_tensor(
                probabilities, device=output_device, dtype=output_dtype
            ),
            synthesis_valid=torch.as_tensor(valid, device=output_device),
            gate_candidate=torch.as_tensor(candidate, device=output_device),
            accepted=torch.as_tensor(accepted, device=output_device),
            nominal_gain=nominal,
            predicted_gain=torch.as_tensor(
                predicted_gain_numpy, device=output_device, dtype=output_dtype
            ),
            target_gain=torch.as_tensor(
                target, device=output_device, dtype=output_dtype
            ),
        )
