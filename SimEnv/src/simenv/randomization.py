from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch

from .errors import ConfigurationError


class ParameterRandomizer:
    """Materialize Randomizable nodes as deterministic batched tensors."""

    def __init__(
        self,
        base_seed: int,
        parallel_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._base_seed = base_seed
        self._parallel_count = parallel_count
        self._device = device
        self._dtype = dtype

    def sample(self, node: Any, path: str) -> torch.Tensor:
        if not isinstance(node, Mapping) or "value" not in node:
            raise ConfigurationError(f"{path} must be a mapping containing 'value'")

        try:
            nominal = torch.as_tensor(node["value"], dtype=self._dtype, device=self._device)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{path}.value must contain numeric data") from exc

        nominal = nominal.unsqueeze(0).expand(self._parallel_count, *nominal.shape)
        spec = node.get("randomization")
        if spec is None or spec.get("distribution", "none") == "none":
            return nominal.clone()
        if not isinstance(spec, Mapping):
            raise ConfigurationError(f"{path}.randomization must be a mapping")

        distribution = spec.get("distribution")
        mode = spec.get("mode", "absolute")
        if mode not in {"absolute", "relative"}:
            raise ConfigurationError(f"{path}.randomization.mode must be absolute or relative")

        generator = torch.Generator(device=self._device)
        generator.manual_seed(self._seed_for(path))
        shape = nominal.shape

        if distribution == "normal":
            mean = self._broadcast(spec.get("mean", 0.0), nominal, f"{path}.randomization.mean")
            stddev = self._broadcast(
                spec.get("stddev"), nominal, f"{path}.randomization.stddev"
            )
            if torch.any(stddev < 0):
                raise ConfigurationError(f"{path}.randomization.stddev must be non-negative")
            offset = torch.randn(shape, generator=generator, device=self._device, dtype=self._dtype)
            offset = offset * stddev + mean
        elif distribution == "uniform":
            low = self._broadcast(spec.get("min"), nominal, f"{path}.randomization.min")
            high = self._broadcast(spec.get("max"), nominal, f"{path}.randomization.max")
            if torch.any(high < low):
                raise ConfigurationError(f"{path}.randomization.max must be >= min")
            offset = torch.rand(shape, generator=generator, device=self._device, dtype=self._dtype)
            offset = low + offset * (high - low)
        else:
            raise ConfigurationError(
                f"{path}.randomization.distribution must be none, normal, or uniform"
            )

        if "clip" in spec:
            clip = spec["clip"]
            if not isinstance(clip, (list, tuple)) or len(clip) != 2:
                raise ConfigurationError(f"{path}.randomization.clip must be [min, max]")
            clip_min = self._broadcast(clip[0], nominal, f"{path}.randomization.clip[0]")
            clip_max = self._broadcast(clip[1], nominal, f"{path}.randomization.clip[1]")
            if torch.any(clip_max < clip_min):
                raise ConfigurationError(f"{path}.randomization.clip maximum must be >= minimum")
            offset = torch.maximum(torch.minimum(offset, clip_max), clip_min)

        result = nominal * (1 + offset) if mode == "relative" else nominal + offset
        if not torch.isfinite(result).all():
            raise ConfigurationError(f"{path} randomization produced non-finite values")
        return result

    def _broadcast(self, value: Any, target: torch.Tensor, path: str) -> torch.Tensor:
        if value is None:
            raise ConfigurationError(f"{path} is required")
        try:
            tensor = torch.as_tensor(value, dtype=self._dtype, device=self._device)
            return torch.broadcast_to(tensor, target.shape)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ConfigurationError(f"{path} is not broadcastable to {tuple(target.shape)}") from exc

    def _seed_for(self, path: str) -> int:
        payload = f"{self._base_seed}:{path}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "little") & ((1 << 63) - 1)

