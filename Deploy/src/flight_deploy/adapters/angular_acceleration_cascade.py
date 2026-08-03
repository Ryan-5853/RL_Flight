from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..errors import CheckpointError
from ..model import ConvertedPolicy
from .flight_train import FlightTrainMLPAdapter, _mapping, _sequence


class FlightTrainAngularAccelerationCascadeAdapter(FlightTrainMLPAdapter):
    """Export the allocated angular-acceleration policy with its controller ABI."""

    name = "flight-train-angular-acceleration-cascade-v1"
    priority = 200

    _PROFILE = "angular_acceleration_allocated_inner_loop_21d_v2"
    _TRANSFORM = "coaxial_differential_cyclic"

    def probe(self, checkpoint: Mapping[str, Any]) -> bool:
        if not super().probe(checkpoint):
            return False
        config = checkpoint.get("config")
        if not isinstance(config, Mapping):
            return False
        contract = config.get("control_contract")
        return (
            isinstance(contract, Mapping)
            and contract.get("observation_profile") == self._PROFILE
            and isinstance(contract.get("action_transform"), Mapping)
            and contract["action_transform"].get("type") == self._TRANSFORM
        )

    def describe(self, checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            **super().describe(checkpoint),
            "adapter": self.name,
            "controller_type": "angular_acceleration_cascade",
        }

    def convert(
        self,
        checkpoint: Mapping[str, Any],
        *,
        source_path: Path,
    ) -> ConvertedPolicy:
        converted = super().convert(checkpoint, source_path=source_path)
        if converted.output_dim != 3:
            raise CheckpointError(
                "allocated angular-acceleration policy must have three outputs"
            )
        config = _mapping(checkpoint.get("config"), "config")
        task = _mapping(config.get("task"), "config.task")
        training_outer = _mapping(task.get("outer_loop"), "config.task.outer_loop")
        termination = _mapping(
            task.get("termination"), "config.task.termination"
        )
        contract = dict(converted.contract)
        history = _mapping(
            contract.get("observation_history"),
            "config.control_contract.observation_history",
        )
        if history.get("mode") != "uniform":
            raise CheckpointError(
                "angular-acceleration cascade export requires uniform history"
            )
        fields = _sequence(
            _mapping(
                contract.get("policy_action"),
                "config.control_contract.policy_action",
            ).get("fields"),
            "config.control_contract.policy_action.fields",
        )
        expected_fields = (
            "lower_motor_differential",
            "servo_cyclic_a",
            "servo_cyclic_b",
        )
        if fields != expected_fields:
            raise CheckpointError(
                "allocated angular-acceleration policy action fields are incompatible"
            )
        command_limit = self._vector3(
            training_outer,
            "max_angular_acceleration_rad_s2",
        )
        contract["version"] = "angular_acceleration_cascade_v1"
        contract["base_observation"] = {
            "dimension": 21,
            "fields": [
                {"name": "desired_angular_acceleration_b", "width": 3},
                {"name": "actual_angular_acceleration_b", "width": 3},
                {"name": "angular_velocity_b", "width": 3},
                {"name": "linear_acceleration_n", "width": 3},
                {"name": "motor_speed", "width": 2},
                {"name": "servo_angle", "width": 3},
                {"name": "upper_throttle_command", "width": 1},
                {"name": "previous_policy_action", "width": 3},
            ],
        }
        contract["normalization"] = {
            "desired_angular_acceleration_rad_s2": list(command_limit),
            "actual_angular_acceleration_rad_s2": list(command_limit),
            "angular_velocity_rad_s": float(
                termination.get("max_angular_rate_rad_s", 6.0)
            ),
            "acceleration_m_s2": 9.80665,
            "motor_speed_rad_s": 1800.0,
            "servo_angle_rad": 1.5707963267948966,
            "upper_throttle": "2*x-1",
            "previous_policy_action": "identity",
        }
        contract["controller"] = {
            "type": "attitude_pid_angular_acceleration_cascade",
            "version": 1,
            "control_hz": 500,
            "attitude_error": "shortest_body_rotation_vector",
            "outer_loop": {
                # A/B tests in the 500 Hz WebUI loop showed that lower-bandwidth
                # candidates increased attitude RMSE without reducing saturation.
                "proportional_gain": [18.0, 18.0, 8.0],
                "integral_gain": [1.5, 1.5, 0.75],
                "derivative_gain": [7.0, 7.0, 4.0],
                "integral_limit_rad_s": [0.15, 0.15, 0.2],
                "max_angular_acceleration_rad_s2": list(command_limit),
            },
            "angular_acceleration_estimator": {
                "type": "backward_difference",
                "source": "angular_velocity_b",
                "reset_value": [0.0, 0.0, 0.0],
            },
            "identified_inner_loop": {
                "method": "fixed-suite first-order ARX",
                "delay_ms": [4.0, 4.0, 2.0],
                "time_constant_ms": [234.35, 250.0, 304.0],
                "effective_bandwidth_rad_s": [4.27, 4.0, 3.29],
            },
            "outer_loop_validation": {
                "suite": "webui_virtual_pilot_10s_seed_52040",
                "selected_attitude_rmse_deg": 1.587,
                "selected_max_attitude_error_deg": 2.716,
                "selected_action_saturation_fraction": 0.0,
                "candidate_order_best_to_worst": [
                    "training_bandwidth",
                    "identified_bandwidth_compromise",
                    "medium_bandwidth",
                    "low_bandwidth",
                ],
            },
            "training_outer_loop": dict(training_outer),
        }
        simulator_compatibility = self._simulator_compatibility(source_path)
        if simulator_compatibility is not None:
            contract["simulator_compatibility"] = simulator_compatibility
        converted.contract = contract
        converted.source_metadata = {
            **dict(converted.source_metadata),
            "controller_type": "angular_acceleration_cascade",
        }
        return converted

    @staticmethod
    def _vector3(node: Mapping[str, Any], name: str) -> tuple[float, float, float]:
        values = _sequence(node.get(name), f"config.task.outer_loop.{name}")
        if len(values) != 3:
            raise CheckpointError(f"config.task.outer_loop.{name} must have length 3")
        result = tuple(float(value) for value in values)
        if any(value <= 0 for value in result):
            raise CheckpointError(f"config.task.outer_loop.{name} must be positive")
        return result  # type: ignore[return-value]

    @classmethod
    def _simulator_compatibility(
        cls, source_path: Path | None
    ) -> Mapping[str, Any] | None:
        if source_path is None:
            return None
        run_manifest_path = source_path.parent.parent / "manifest.json"
        if not run_manifest_path.is_file():
            return None
        try:
            run_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            simulator_path = Path(run_manifest["simulator_config"])
            simulator_config = yaml.safe_load(
                simulator_path.read_text(encoding="utf-8")
            )
        except (KeyError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise CheckpointError(
                f"failed to read source simulator contract: {exc}"
            ) from exc
        if not isinstance(simulator_config, Mapping):
            raise CheckpointError("source simulator configuration is not a mapping")
        return {
            "fingerprint_version": 2,
            "sha256": cls._simulator_fingerprint(simulator_config),
            "source_sha256": run_manifest.get("simulator_config_sha256"),
            "comparison": "effective_single_instance_dynamics",
            "excluded_top_level_fields": [
                "schema_version",
                "seed",
                "parallel",
                "initial_state",
                "logging",
            ],
            "configuration": dict(simulator_config),
        }

    @classmethod
    def _simulator_fingerprint(cls, config: Mapping[str, Any]) -> str:
        relevant = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "schema_version",
                "seed",
                "parallel",
                "initial_state",
                "logging",
            }
        }
        canonical = json.dumps(
            cls._canonical_value(relevant),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def _canonical_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            zero_delay = cls._parameter_value(value.get("delay")) == 0.0
            result = {}
            for key, item in value.items():
                key = str(key)
                if key == "name":
                    continue
                if (
                    key == "randomization"
                    and isinstance(item, Mapping)
                    and item.get("distribution", "none") == "none"
                ):
                    continue
                if key == "interpolation" and zero_delay:
                    continue
                result[key] = cls._canonical_value(item)
            return result
        if isinstance(value, (list, tuple)):
            return [cls._canonical_value(item) for item in value]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        raise CheckpointError(
            f"unsupported simulator contract value {type(value).__name__}"
        )

    @staticmethod
    def _parameter_value(value: Any) -> float | None:
        if isinstance(value, Mapping):
            value = value.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
