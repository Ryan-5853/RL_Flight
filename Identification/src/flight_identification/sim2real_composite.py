from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.linalg import expm


DEFAULT_BASIS_TIME_CONSTANTS_S = (0.015, 0.040, 0.080)
DEFAULT_RESPONSE_TIMES_S = tuple(np.arange(0.002, 0.202, 0.002))
DEFAULT_RESPONSE_SNAPSHOT_TIMES_S = (0.010, 0.020, 0.040, 0.080, 0.160)


def servo_mode_transform(
    nominal_effectiveness: np.ndarray,
    servo_command_slopes: np.ndarray,
) -> np.ndarray:
    """Return orthonormal servo command modes ordered by nominal authority."""

    command_effectiveness = nominal_effectiveness[:, 2:] * np.asarray(
        servo_command_slopes, dtype=np.float64
    )[None, :]
    _, _, transform = np.linalg.svd(command_effectiveness, full_matrices=True)
    for row in range(len(transform)):
        pivot = int(np.argmax(np.abs(transform[row])))
        if transform[row, pivot] < 0.0:
            transform[row] *= -1.0
    return transform


def composite_target_names(
    basis_time_constants_s: Sequence[float] = DEFAULT_BASIS_TIME_CONSTANTS_S,
    adaptive_mode_indices: Sequence[int] = (1, 2),
) -> tuple[str, ...]:
    return tuple(
        f"servo_composite.tau_{round(tau * 1000):03d}ms.{axis}.mode_{mode + 1}"
        for tau in basis_time_constants_s
        for axis in ("roll", "pitch", "yaw")
        for mode in adaptive_mode_indices
    )


def composite_response_target_names(
    response_times_s: Sequence[float] = DEFAULT_RESPONSE_SNAPSHOT_TIMES_S,
    adaptive_mode_indices: Sequence[int] = (0, 1, 2),
) -> tuple[str, ...]:
    return tuple(
        f"servo_step.t_{round(time * 1000):03d}ms.{axis}.mode_{mode + 1}"
        for time in response_times_s
        for axis in ("roll", "pitch", "yaw")
        for mode in adaptive_mode_indices
    )


def fit_coefficients_from_step_response(
    response: torch.Tensor,
    response_times_s: Sequence[float],
    servo_command_slope: float,
    basis_time_constants_s: Sequence[float] = DEFAULT_BASIS_TIME_CONSTANTS_S,
    ridge: float = 1e-6,
) -> torch.Tensor:
    value = response.to(torch.float64)
    times = value.new_tensor(response_times_s)
    basis_tau = value.new_tensor(basis_time_constants_s)
    basis_step = servo_command_slope * (
        1.0 - torch.exp(-times[:, None] / basis_tau[None, :])
    )
    weights = torch.exp(-times / 0.12).sqrt()
    weighted_basis = basis_step * weights[:, None]
    projection = torch.linalg.solve(
        weighted_basis.T @ weighted_basis
        + ridge
        * torch.eye(
            len(basis_tau), dtype=value.dtype, device=value.device
        ),
        weighted_basis.T * weights[None, :],
    )
    return torch.einsum("bt,ntam->nbam", projection, value)


def fit_composite_coefficients(
    targets: torch.Tensor,
    mode_transform: np.ndarray,
    servo_command_slopes: np.ndarray,
    basis_time_constants_s: Sequence[float] = DEFAULT_BASIS_TIME_CONSTANTS_S,
    response_times_s: Sequence[float] = DEFAULT_RESPONSE_TIMES_S,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Project true command-to-acceleration steps onto fixed stable filters."""

    value = targets.to(torch.float64)
    effectiveness = value[:, :15].reshape(-1, 3, 5)[:, :, 2:]
    servo_tau = torch.pow(10.0, value[:, 20:23])
    slopes = value.new_tensor(np.asarray(servo_command_slopes, dtype=np.float64))
    transform_inverse = value.new_tensor(
        np.asarray(mode_transform, dtype=np.float64).T
    )
    times = value.new_tensor(response_times_s)
    true_step = torch.einsum(
        "nai,nti,im->ntam",
        effectiveness,
        slopes[None, None, :]
        * (1.0 - torch.exp(-times[None, :, None] / servo_tau[:, None, :])),
        transform_inverse,
    )

    return fit_coefficients_from_step_response(
        true_step,
        response_times_s,
        float(np.mean(servo_command_slopes)),
        basis_time_constants_s,
        ridge,
    )


def reconstruct_composite_step_response(
    coefficients: torch.Tensor,
    basis_time_constants_s: Sequence[float] = DEFAULT_BASIS_TIME_CONSTANTS_S,
    response_times_s: Sequence[float] = DEFAULT_RESPONSE_TIMES_S,
    servo_command_slope: float = 0.35,
) -> torch.Tensor:
    times = coefficients.new_tensor(response_times_s)
    tau = coefficients.new_tensor(basis_time_constants_s)
    basis_step = servo_command_slope * (
        1.0 - torch.exp(-times[:, None] / tau[None, :])
    )
    return torch.einsum("tb,nbam->ntam", basis_step, coefficients)


def true_servo_mode_step_response(
    targets: torch.Tensor,
    mode_transform: np.ndarray,
    servo_command_slopes: np.ndarray,
    response_times_s: Sequence[float] = DEFAULT_RESPONSE_TIMES_S,
) -> torch.Tensor:
    value = targets.to(torch.float64)
    effectiveness = value[:, :15].reshape(-1, 3, 5)[:, :, 2:]
    servo_tau = torch.pow(10.0, value[:, 20:23])
    slopes = value.new_tensor(np.asarray(servo_command_slopes, dtype=np.float64))
    transform_inverse = value.new_tensor(
        np.asarray(mode_transform, dtype=np.float64).T
    )
    times = value.new_tensor(response_times_s)
    return torch.einsum(
        "nai,nti,im->ntam",
        effectiveness,
        slopes[None, None, :]
        * (1.0 - torch.exp(-times[None, :, None] / servo_tau[:, None, :])),
        transform_inverse,
    )


def merge_adaptive_composite_coefficients(
    adaptive: np.ndarray,
    nominal: np.ndarray,
    adaptive_mode_indices: Sequence[int] = (1, 2),
) -> np.ndarray:
    merged = np.broadcast_to(nominal, (len(adaptive), *nominal.shape)).copy()
    merged[:, :, :, list(adaptive_mode_indices)] = adaptive.reshape(
        len(adaptive), nominal.shape[0], 3, len(adaptive_mode_indices)
    )
    return merged


def composite_discrete_model(
    motor_effectiveness: np.ndarray,
    motor_command_slopes: np.ndarray,
    motor_time_constants_s: np.ndarray,
    composite_coefficients: np.ndarray,
    mode_transform: np.ndarray,
    servo_command_slopes: np.ndarray,
    basis_time_constants_s: Sequence[float] = DEFAULT_BASIS_TIME_CONSTANTS_S,
    dt: float = 1.0 / 500.0,
    integral: bool = True,
    upper_external: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    basis_count = len(basis_time_constants_s)
    latent_count = 3 * basis_count
    base_state_count = 7 + latent_count
    state_count = base_state_count + (3 if integral else 0)
    a = np.zeros((state_count, state_count), dtype=np.float64)
    input_count = 4 if upper_external else 5
    b = np.zeros((state_count, input_count), dtype=np.float64)
    a[0, 2] = 1.0
    a[1, 3] = 1.0
    a[2:5, 5:7] = motor_effectiveness
    a[5:7, 5:7] = np.diag(-1.0 / motor_time_constants_s)
    if upper_external:
        # The upper motor is externally commanded (pilot); only the lower motor
        # remains a control input.
        b[6, 0] = motor_command_slopes[1] / motor_time_constants_s[1]
    else:
        b[5:7, :2] = np.diag(
            motor_command_slopes / motor_time_constants_s
        )
    servo_input = np.asarray(mode_transform) @ np.diag(servo_command_slopes)
    for basis_index, tau in enumerate(basis_time_constants_s):
        start = 7 + 3 * basis_index
        stop = start + 3
        a[2:5, start:stop] = composite_coefficients[basis_index]
        a[start:stop, start:stop] = -np.eye(3) / tau
        servo_start = 1 if upper_external else 2
        b[start:stop, servo_start:] = servo_input / tau
    if integral:
        a[base_state_count, 0] = 1.0
        a[base_state_count + 1, 1] = 1.0
        a[base_state_count + 2, 4] = 1.0
    augmented = np.block(
        [
            [a, b],
            [
                np.zeros(
                    (input_count, state_count + input_count),
                    dtype=np.float64,
                )
            ],
        ]
    )
    discrete = expm(augmented * dt)
    return discrete[:state_count, :state_count], discrete[:state_count, state_count:]


def composite_lqr_weights(
    original_state_scales: np.ndarray,
    integral_state_scales: np.ndarray,
    input_scales: np.ndarray,
    input_weight_scale: float,
    basis_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    latent_scale = float(np.mean(original_state_scales[7:10])) * np.sqrt(basis_count)
    state_scales = np.concatenate(
        (
            original_state_scales[:7],
            np.full(3 * basis_count, latent_scale),
            integral_state_scales,
        )
    )
    return (
        np.diag(1.0 / state_scales**2),
        input_weight_scale * np.diag(1.0 / input_scales**2),
    )
