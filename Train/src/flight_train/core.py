from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDictBase


@dataclass(frozen=True)
class EnvSpec:
    """训练层使用的环境静态契约。

    设备和数据类型在环境创建时固定，采样器据此直接在目标设备上创建循环状态，
    避免 rollout 热路径中出现隐式的 CPU/GPU 拷贝。
    """

    parallel_count: int
    observation_dim: int
    action_dim: int
    device: torch.device
    dtype: torch.dtype
    physics_hz: int
    control_hz: int
    observation_fields: tuple[str, ...]


class BatchedControlEnv(Protocol):
    """批量控制环境的最小接口，隔离训练算法与具体仿真实现。

    ``reset`` 和 ``step`` 返回的 TensorDict 必须与 ``spec.device`` 同设备；
    ``step`` 接收形状为 ``[B, action_dim]`` 的标准化动作。
    """

    @property
    def spec(self) -> EnvSpec: ...

    def reset(
        self,
        mask: torch.Tensor | None = None,
        *,
        static_parameters: TensorDictBase | None = None,
    ) -> TensorDictBase: ...

    def step(self, standard_action: torch.Tensor) -> TensorDictBase: ...

    def close(self) -> None: ...


def assert_tensor_on(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
    name: str,
) -> None:
    """在接口边界校验张量类型、设备、数据类型和精确形状。

    这里故意不自动调用 ``to`` 或 reshape：静默修正会掩盖上游设备搬运或
    batch 语义错误，并可能让 rollout 在不知情的情况下离开 GPU。
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def tensordict_to_device(
    value: TensorDictBase, device: torch.device
) -> TensorDictBase:
    """逐叶迁移 checkpoint TensorDict，并重建可信的设备元数据。

    ``torch.load(map_location="cpu")`` 会迁移叶张量，但某些 TensorDict 版本仍会
    保留保存时的 CUDA ``device`` 元数据。此时直接 ``value.to(cuda)`` 会错误地
    判断无需迁移。``apply`` 会无条件访问每个张量叶节点，并用目标设备重建容器。
    """

    moved = value.apply(lambda tensor: tensor.to(device), device=device)
    if moved is None:  # TensorDict.apply(inplace=False) 按契约应返回新对象。
        raise RuntimeError("failed to rebuild checkpoint TensorDict on target device")
    for key, tensor in moved.items(include_nested=True, leaves_only=True):
        if isinstance(tensor, torch.Tensor) and tensor.device != device:
            raise ValueError(
                f"checkpoint TensorDict leaf remains on {tensor.device}: {key}"
            )
    return moved
