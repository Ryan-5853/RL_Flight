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
    sampling_design: str = "random"


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
class EmpiricalCoreRanges:
    inertia_scale: tuple[float, float]
    thrust_to_weight: tuple[float, float]
    motor_reaction_scale: tuple[float, float]
    motor_time_constant_s: tuple[float, float]
    grid_effectiveness_scale: tuple[float, float]
    servo_time_constant_s: tuple[float, float]


@dataclass(frozen=True)
class Sim2RealMicroRanges:
    mass_kg: tuple[float, float]
    inertia_xy_radius_of_gyration_m: tuple[float, float]
    inertia_z_radius_of_gyration_m: tuple[float, float]
    thrust_to_weight: tuple[float, float]
    motor_reaction_scale: tuple[float, float]
    motor_reaction_ratio: tuple[float, float]
    motor_time_constant_s: tuple[float, float]
    direct_center_xy_radius_m: tuple[float, float]
    direct_center_z_m: tuple[float, float]
    direct_thrust_fraction: tuple[float, float]
    grid_radius_m: tuple[float, float]
    grid_azimuth_error_rad: tuple[float, float]
    grid_center_z_m: tuple[float, float]
    grid_effectiveness_scale: tuple[float, float]
    servo_time_constant_s: tuple[float, float]
    servo_max_speed_rad_s: tuple[float, float]
    coupling_attenuation: tuple[float, float]


@dataclass(frozen=True)
class Sim2RealFeasibility:
    maximum_linear_trim_servo_angle_rad: float
    maximum_linear_trim_residual_rad_s2: float


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
    parameterization: str = "legacy_collective"
    empirical_ranges: EmpiricalCoreRanges | None = None
    sim2real_ranges: Sim2RealMicroRanges | None = None
    sim2real_feasibility: Sim2RealFeasibility | None = None

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
    scale_range = tuple(
        float(v)
        for v in sampling.get("log10_effectiveness_range", (-2.0, 2.0))
    )
    if len(scale_range) != 2 or not all(math.isfinite(v) for v in scale_range):
        raise ValueError("sampling.log10_effectiveness_range must have two finite values")
    if scale_range[0] >= scale_range[1]:
        raise ValueError("sampling.log10_effectiveness_range must be ordered")

    parameterization = str(
        sampling.get("parameterization", "legacy_collective")
    )
    if parameterization not in {
        "legacy_collective",
        "empirical_core",
        "sim2real_micro",
    }:
        raise ValueError(
            "sampling.parameterization must be legacy_collective, empirical_core, "
            "or sim2real_micro"
        )

    def ordered_range(
        node: Mapping[str, Any],
        name: str,
        *,
        maximum: float | None = None,
        signed: bool = False,
    ) -> tuple[float, float]:
        values = tuple(float(v) for v in node.get(name, ()))
        positivity = all(math.isfinite(v) for v in values) if signed else all(
            math.isfinite(v) and v > 0.0 for v in values
        )
        if (
            len(values) != 2
            or not positivity
            or values[0] >= values[1]
            or (maximum is not None and values[1] > maximum)
        ):
            suffix = f" and <= {maximum}" if maximum is not None else ""
            sign = "signed " if signed else "positive "
            raise ValueError(
                f"sampling.empirical_ranges.{name} must contain two ordered "
                f"{sign}finite values{suffix}"
            )
        return values[0], values[1]

    empirical_ranges = None
    if parameterization == "empirical_core":
        ranges = sampling.get("empirical_ranges")
        if not isinstance(ranges, Mapping):
            raise ValueError(
                "sampling.empirical_ranges must be a mapping for empirical_core"
            )
        empirical_ranges = EmpiricalCoreRanges(
            inertia_scale=ordered_range(ranges, "inertia_scale"),
            thrust_to_weight=ordered_range(ranges, "thrust_to_weight"),
            motor_reaction_scale=ordered_range(
                ranges, "motor_reaction_scale"
            ),
            motor_time_constant_s=ordered_range(
                ranges, "motor_time_constant_s"
            ),
            grid_effectiveness_scale=ordered_range(
                ranges, "grid_effectiveness_scale"
            ),
            servo_time_constant_s=ordered_range(
                ranges, "servo_time_constant_s"
            ),
        )
    sim2real_ranges = None
    sim2real_feasibility = None
    if parameterization == "sim2real_micro":
        ranges = sampling.get("sim2real_ranges")
        if not isinstance(ranges, Mapping):
            raise ValueError(
                "sampling.sim2real_ranges must be a mapping for sim2real_micro"
            )

        def sim_range(name: str) -> tuple[float, float]:
            return ordered_range(ranges, name)

        sim2real_ranges = Sim2RealMicroRanges(
            mass_kg=sim_range("mass_kg"),
            inertia_xy_radius_of_gyration_m=sim_range(
                "inertia_xy_radius_of_gyration_m"
            ),
            inertia_z_radius_of_gyration_m=sim_range(
                "inertia_z_radius_of_gyration_m"
            ),
            thrust_to_weight=sim_range("thrust_to_weight"),
            motor_reaction_scale=sim_range("motor_reaction_scale"),
            motor_reaction_ratio=sim_range("motor_reaction_ratio"),
            motor_time_constant_s=sim_range("motor_time_constant_s"),
            direct_center_xy_radius_m=sim_range(
                "direct_center_xy_radius_m"
            ),
            direct_center_z_m=ordered_range(
                ranges, "direct_center_z_m", signed=True
            ),
            direct_thrust_fraction=sim_range("direct_thrust_fraction"),
            grid_radius_m=sim_range("grid_radius_m"),
            grid_azimuth_error_rad=tuple(
                float(value)
                for value in ranges.get("grid_azimuth_error_rad", ())
            ),
            grid_center_z_m=sim_range("grid_center_z_m"),
            grid_effectiveness_scale=sim_range(
                "grid_effectiveness_scale"
            ),
            servo_time_constant_s=sim_range("servo_time_constant_s"),
            servo_max_speed_rad_s=sim_range("servo_max_speed_rad_s"),
            coupling_attenuation=tuple(
                float(value)
                for value in ranges.get("coupling_attenuation", ())
            ),
        )
        for name in ("grid_azimuth_error_rad", "coupling_attenuation"):
            values = getattr(sim2real_ranges, name)
            if (
                len(values) != 2
                or not all(math.isfinite(value) for value in values)
                or values[0] > values[1]
            ):
                raise ValueError(
                    f"sampling.sim2real_ranges.{name} must contain two ordered "
                    "finite values"
                )
        if sim2real_ranges.direct_thrust_fraction[1] >= 1.0:
            raise ValueError("direct_thrust_fraction maximum must be below 1")
        if not 0.0 <= sim2real_ranges.coupling_attenuation[0]:
            raise ValueError("coupling_attenuation minimum must be nonnegative")
        if sim2real_ranges.coupling_attenuation[1] > 1.0:
            raise ValueError("coupling_attenuation maximum must not exceed 1")
        feasibility = sampling.get("feasibility")
        if not isinstance(feasibility, Mapping):
            raise ValueError(
                "sampling.feasibility must be a mapping for sim2real_micro"
            )
        sim2real_feasibility = Sim2RealFeasibility(
            maximum_linear_trim_servo_angle_rad=positive(
                feasibility, "maximum_linear_trim_servo_angle_rad"
            ),
            maximum_linear_trim_residual_rad_s2=positive(
                feasibility, "maximum_linear_trim_residual_rad_s2"
            ),
        )
    rate = tuple(float(v) for v in initial.get("maximum_angular_rate_rad_s", ()))
    if len(rate) != 3 or any(not math.isfinite(v) or v < 0 for v in rate):
        raise ValueError("initial_state.maximum_angular_rate_rad_s must contain 3 nonnegative values")
    initial_sampling_design = str(initial.get("sampling_design", "random"))
    if initial_sampling_design not in {"random", "stratified_multiaxis"}:
        raise ValueError(
            "initial_state.sampling_design must be random or stratified_multiaxis"
        )
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
            sampling_design=initial_sampling_design,
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
        parameterization=parameterization,
        empirical_ranges=empirical_ranges,
        sim2real_ranges=sim2real_ranges,
        sim2real_feasibility=sim2real_feasibility,
    )
