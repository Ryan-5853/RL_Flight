from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class InitialStateConfig:
    minimum_tilt_rad: float
    maximum_tilt_rad: float
    maximum_angular_rate_rad_s: tuple[float, float, float]


@dataclass(frozen=True)
class ConvergenceConfig:
    maximum_roll_pitch_error_rad: float
    maximum_angular_rate_rad_s: float
    hold_s: float
    safety_tilt_rad: float
    safety_angular_rate_rad_s: float


@dataclass(frozen=True)
class WindowConfig:
    length_s: float
    stride_s: float
    split_fractions: tuple[float, float, float]
    maximum_start_s: float | None = None


@dataclass(frozen=True)
class IdentificationExperimentConfig:
    source_path: Path
    simulator_config: Path
    controller_config: Path
    output_directory: Path
    seed: int
    device: str
    dtype: str
    measurement_mode: str
    parameter_groups: int
    initial_conditions_per_group: int
    parallel_count: int
    episode_duration_s: float
    log10_effectiveness_range: tuple[float, float]
    initial_state: InitialStateConfig
    convergence: ConvergenceConfig
    window: WindowConfig

    @property
    def control_hz(self) -> int:
        return 500

    @property
    def episode_steps(self) -> int:
        return round(self.episode_duration_s * self.control_hz)

    @property
    def window_steps(self) -> int:
        return round(self.window.length_s * self.control_hz)

    @property
    def stride_steps(self) -> int:
        return round(self.window.stride_s * self.control_hz)


def load_experiment_config(path: str | Path) -> IdentificationExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("identification experiment root must be a mapping")
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("identification experiment schema_version must equal 1")

    def mapping(name: str) -> Mapping[str, Any]:
        value = raw.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        return value

    def positive(node: Mapping[str, Any], name: str) -> float:
        value = float(node.get(name, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def relative_path(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty path")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        return candidate.resolve()

    run = mapping("run")
    sampling = mapping("sampling")
    initial = mapping("initial_state")
    convergence = mapping("convergence")
    window = mapping("window")
    scale_range = tuple(float(v) for v in sampling.get("log10_effectiveness_range", ()))
    if len(scale_range) != 2 or not all(math.isfinite(v) for v in scale_range):
        raise ValueError("sampling.log10_effectiveness_range must have two finite values")
    if scale_range[0] >= scale_range[1]:
        raise ValueError("sampling.log10_effectiveness_range must be ordered")
    rate = tuple(float(v) for v in initial.get("maximum_angular_rate_rad_s", ()))
    if len(rate) != 3 or any(not math.isfinite(v) or v < 0 for v in rate):
        raise ValueError("initial_state.maximum_angular_rate_rad_s must contain 3 nonnegative values")
    fractions = tuple(float(v) for v in window.get("split_fractions", ()))
    if len(fractions) != 3 or any(v <= 0 for v in fractions):
        raise ValueError("window.split_fractions must contain 3 positive values")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-8):
        raise ValueError("window.split_fractions must sum to 1")

    groups = int(sampling.get("parameter_groups", 0))
    conditions = int(sampling.get("initial_conditions_per_group", 0))
    parallel = int(run.get("parallel_count", 0))
    if groups <= 0 or conditions <= 0 or parallel <= 0:
        raise ValueError("parameter_groups, initial_conditions_per_group and parallel_count must be positive")
    if parallel < conditions:
        raise ValueError("run.parallel_count must fit at least one complete parameter group")
    parallel -= parallel % conditions

    minimum_tilt = float(initial.get("minimum_tilt_rad", 0.0))
    maximum_tilt = float(initial.get("maximum_tilt_rad", 0.0))
    if not 0.0 <= minimum_tilt <= maximum_tilt < math.pi / 2:
        raise ValueError("initial tilt range must satisfy 0 <= min <= max < pi/2")
    episode_duration = positive(sampling, "episode_duration_s")
    window_length = positive(window, "length_s")
    window_stride = positive(window, "stride_s")
    maximum_start_raw = window.get("maximum_start_s")
    maximum_start_s = (
        None if maximum_start_raw is None else float(maximum_start_raw)
    )
    if maximum_start_s is not None and (
        not math.isfinite(maximum_start_s) or maximum_start_s < 0.0
    ):
        raise ValueError("window.maximum_start_s must be finite and nonnegative")
    if window_length > episode_duration:
        raise ValueError("window length cannot exceed episode duration")
    for value, name in (
        (episode_duration, "episode_duration_s"),
        (window_length, "window.length_s"),
        (window_stride, "window.stride_s"),
        (positive(convergence, "hold_s"), "convergence.hold_s"),
    ):
        steps = value * 500.0
        if not math.isclose(steps, round(steps), abs_tol=1e-8):
            raise ValueError(f"{name} must be an integer number of 500 Hz steps")

    dtype = str(run.get("dtype", "float32"))
    if dtype not in {"float32", "float64"}:
        raise ValueError("run.dtype must be float32 or float64")
    observation = raw.get("observation", {})
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    measurement_mode = str(observation.get("measurement_mode", "ideal"))
    if measurement_mode not in {"ideal", "simulated_sensors"}:
        raise ValueError(
            "observation.measurement_mode must be ideal or simulated_sensors"
        )
    return IdentificationExperimentConfig(
        source_path=source,
        simulator_config=relative_path(raw.get("simulator_config"), "simulator_config"),
        controller_config=relative_path(raw.get("controller_config"), "controller_config"),
        output_directory=relative_path(run.get("output_directory"), "run.output_directory"),
        seed=int(run.get("seed", 0)),
        device=str(run.get("device", "cpu")),
        dtype=dtype,
        measurement_mode=measurement_mode,
        parameter_groups=groups,
        initial_conditions_per_group=conditions,
        parallel_count=parallel,
        episode_duration_s=episode_duration,
        log10_effectiveness_range=(scale_range[0], scale_range[1]),
        initial_state=InitialStateConfig(
            minimum_tilt_rad=minimum_tilt,
            maximum_tilt_rad=maximum_tilt,
            maximum_angular_rate_rad_s=rate,  # type: ignore[arg-type]
        ),
        convergence=ConvergenceConfig(
            maximum_roll_pitch_error_rad=positive(convergence, "maximum_roll_pitch_error_rad"),
            maximum_angular_rate_rad_s=positive(convergence, "maximum_angular_rate_rad_s"),
            hold_s=positive(convergence, "hold_s"),
            safety_tilt_rad=positive(convergence, "safety_tilt_rad"),
            safety_angular_rate_rad_s=positive(convergence, "safety_angular_rate_rad_s"),
        ),
        window=WindowConfig(
            length_s=window_length,
            stride_s=window_stride,
            split_fractions=fractions,  # type: ignore[arg-type]
            maximum_start_s=maximum_start_s,
        ),
    )
