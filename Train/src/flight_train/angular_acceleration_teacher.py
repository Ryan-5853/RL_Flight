from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .config import ControlContractConfig, TaskConfig


def normalized_action_effectiveness(
    parameters: Mapping[str, torch.Tensor],
    control_contract: ControlContractConfig,
) -> torch.Tensor:
    """Linearized ``d angular_acceleration / d normalized_action`` at trim."""

    if control_contract.action_transform_type != "coaxial_differential_cyclic":
        raise ValueError(
            "incremental angular-acceleration teacher requires the coaxial "
            "differential/cyclic action transform"
        )
    from flight_controller.plant import LocalPlantModel

    first = {name: value[:1].to(torch.float64) for name, value in parameters.items()}
    trim = torch.tensor(
        [
            control_contract.policy_action_trim[0],
            control_contract.lower_motor_upper_ratio
            * control_contract.policy_action_trim[0],
            *control_contract.policy_action_trim[1:4],
        ],
        device=next(iter(first.values())).device,
        dtype=torch.float64,
    )[None]
    command_effectiveness = LocalPlantModel(first).control_effectiveness(trim)[
        0, 1:4
    ]
    motor_scale, cyclic_a_scale, cyclic_b_scale = (
        control_contract.policy_action_residual_scale
    )
    action_to_command = command_effectiveness.new_tensor(
        [
            [0.0, 0.0, 0.0],
            [motor_scale, 0.0, 0.0],
            [0.0, cyclic_a_scale, 0.0],
            [0.0, -0.5 * cyclic_a_scale, 0.5 * 3.0**0.5 * cyclic_b_scale],
            [0.0, -0.5 * cyclic_a_scale, -0.5 * 3.0**0.5 * cyclic_b_scale],
        ]
    )
    inertia = first["body.inertia_diagonal_b"][0]
    return command_effectiveness @ action_to_command / inertia[:, None]


@dataclass
class IncrementalAngularAccelerationTeacher:
    """Model-based INDI teacher matching the neural policy ``forward_step`` API."""

    inverse_effectiveness: torch.Tensor
    command_limit: torch.Tensor
    correction_gain: torch.Tensor
    base_observation_dim: int

    @classmethod
    def create(
        cls,
        parameters: Mapping[str, torch.Tensor],
        control_contract: ControlContractConfig,
        task: TaskConfig,
        correction_gain: tuple[float, float, float],
    ) -> "IncrementalAngularAccelerationTeacher":
        effectiveness = normalized_action_effectiveness(
            parameters, control_contract
        )
        if torch.linalg.cond(effectiveness) > 1.0e6:
            raise ValueError("normalized action effectiveness is ill-conditioned")
        device = effectiveness.device
        return cls(
            inverse_effectiveness=torch.linalg.inv(effectiveness).to(
                dtype=next(iter(parameters.values())).dtype
            ),
            command_limit=torch.tensor(
                task.outer_loop_pid.max_angular_acceleration_rad_s2,
                device=device,
                dtype=next(iter(parameters.values())).dtype,
            ),
            correction_gain=torch.tensor(
                correction_gain,
                device=device,
                dtype=next(iter(parameters.values())).dtype,
            ),
            base_observation_dim=control_contract.base_observation_dim,
        )

    @torch.no_grad()
    def forward_step(
        self,
        observation: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        is_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        del recurrent_state, is_init
        current = observation[:, -self.base_observation_dim :]
        desired = current[:, 0:3] * self.command_limit
        actual = current[:, 3:6] * self.command_limit
        previous_action = current[:, 18:21]
        correction = (desired - actual) @ self.inverse_effectiveness.T
        action = previous_action + self.correction_gain * correction
        return action.clamp(-1.0, 1.0), None
