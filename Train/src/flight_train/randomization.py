from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class StaticParameterSpec:
    name: str
    low: Any
    high: Any
    distribution: str
    mode: str
    unit: str
    seed_stream: str


class StaticRandomizer:
    """训练层静态参数采样器；每个实例和 episode 使用独立确定性随机流。"""

    def __init__(self, specs: tuple[StaticParameterSpec, ...], seed: int, device: torch.device, dtype: torch.dtype) -> None:
        self.specs = specs
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._baselines: dict[str, torch.Tensor] | None = None
        self._stream_seeds = {
            spec.seed_stream: int.from_bytes(
                hashlib.blake2b(
                    f"{self.seed}:{spec.seed_stream}".encode(), digest_size=8
                ).digest(),
                "little",
            )
            & ((1 << 53) - 1)
            for spec in specs
        }

    def bind_baselines(self, parameters: Mapping[str, torch.Tensor]) -> None:
        """从 SimEnv 已解析参数绑定唯一标称值；训练配置不再复制 baseline。"""

        baselines: dict[str, torch.Tensor] = {}
        for spec in self.specs:
            if spec.name not in parameters:
                raise ValueError(f"unknown static randomization parameter: {spec.name}")
            value = parameters[spec.name]
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError(
                    f"SimEnv baseline device/dtype mismatch for {spec.name}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"SimEnv baseline is non-finite: {spec.name}")
            baselines[spec.name] = value.detach().clone()
        self._baselines = baselines

    def sample(self, mask: torch.Tensor, episode_id: torch.Tensor) -> TensorDict:
        if mask.shape != episode_id.shape or mask.dtype != torch.bool:
            raise ValueError("mask and episode_id must be [B] bool/int tensors")
        if mask.device != self.device or episode_id.device != self.device:
            raise ValueError("mask and episode_id must be on the randomizer device")
        if self._baselines is None:
            raise RuntimeError("static randomizer baselines have not been bound from SimEnv")
        batch = mask.shape[0]
        values: dict[str, torch.Tensor] = {}
        for spec in self.specs:
            baseline = self._baselines[spec.name]
            if baseline.shape[0] != batch:
                raise ValueError(f"SimEnv baseline batch changed for {spec.name}")
            if spec.distribution == "uniform":
                low = torch.as_tensor(
                    spec.low, device=self.device, dtype=self.dtype
                ).expand_as(baseline)
                high = torch.as_tensor(
                    spec.high, device=self.device, dtype=self.dtype
                ).expand_as(baseline)
                # 正弦哈希完全由稳定身份决定，不维护会被稀疏 reset 顺序扰动的全局 RNG。
                instance = torch.arange(batch, device=self.device, dtype=self.dtype)
                identity = episode_id.to(self.dtype) * 104729.0 + instance * 13007.0
                base = identity + float(self._stream_seeds[spec.seed_stream] % 1_000_003)
                extra_dims = baseline.ndim - 1
                if extra_dims:
                    base = base.reshape(batch, *([1] * extra_dims))
                    element = torch.arange(
                        baseline[0].numel(), device=self.device, dtype=self.dtype
                    ).reshape(*baseline.shape[1:])
                    base = base + element * 433.0
                uniform = torch.frac(torch.sin(base) * 43758.5453123).abs()
                draw = low + uniform * (high - low)
                sampled = baseline * (1.0 + draw) if spec.mode == "relative" else draw
            else:
                raise ValueError(f"unsupported static distribution: {spec.distribution}")
            if not bool(torch.isfinite(sampled).all()):
                raise ValueError(f"static randomization produced non-finite values: {spec.name}")
            values[spec.name] = sampled
        return TensorDict(values, batch_size=[batch], device=self.device)

    def state_dict(self) -> Mapping[str, Any]:
        return {"seed": self.seed, "stream_seeds": dict(self._stream_seeds)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed or dict(state["stream_seeds"]) != self._stream_seeds:
            raise ValueError("static randomizer state does not match its configuration")
