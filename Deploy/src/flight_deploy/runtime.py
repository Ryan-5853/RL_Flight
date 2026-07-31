from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .bundle import DeploymentBundle


@dataclass(frozen=True)
class BenchmarkResult:
    steps: int
    batch_size: int
    mean_us: float
    p50_us: float
    p99_us: float
    maximum_us: float
    steps_per_second: float

    def as_dict(self) -> Mapping[str, Any]:
        return self.__dict__


class PolicyRuntime:
    """Generic low-latency runtime for a deployment bundle.

    It consumes already preprocessed observations. Domain-specific sensor,
    history, and action transformations are described by the bundle contract
    and can be composed around this component without changing the model ABI.
    """

    def __init__(
        self,
        bundle: DeploymentBundle,
        module: torch.nn.Module,
        *,
        device: torch.device,
        backend: str,
    ) -> None:
        self.bundle = bundle
        self.module = module
        self.device = device
        self.backend = backend
        interface = bundle.manifest["interface"]
        self.input_dim = int(interface["input"]["shape"][-1])
        self.output_dim = int(interface["output"]["shape"][-1])
        self.stateful = bool(interface["stateful"])

    @classmethod
    def load(
        cls,
        bundle_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        backend: str = "torchscript",
        verify_integrity: bool = True,
    ) -> "PolicyRuntime":
        bundle = DeploymentBundle.load(
            bundle_path, verify_integrity=verify_integrity
        )
        target = torch.device(device)
        if backend == "torchscript":
            module = torch.jit.load(
                str(bundle.directory / "model.ts"), map_location=target
            ).eval()
        elif backend == "torch_export":
            module = (
                torch.export.load(bundle.directory / "model.pt2")
                .module()
                .to(target)
            )
        else:
            raise ValueError(
                "backend must be 'torchscript' or 'torch_export'"
            )
        return cls(bundle, module, device=target, backend=backend)

    def infer(self, observation: torch.Tensor) -> torch.Tensor:
        if self.stateful:
            raise RuntimeError("use a stateful backend for this policy")
        if observation.ndim != 2 or observation.shape[-1] != self.input_dim:
            raise ValueError(
                f"observation must have shape [B,{self.input_dim}]"
            )
        if observation.device != self.device:
            raise ValueError(
                f"observation is on {observation.device}, runtime is on {self.device}"
            )
        if observation.dtype != torch.float32:
            raise ValueError("runtime observation dtype must be float32")
        with torch.inference_mode():
            output = self.module(observation)
        if output.shape != (observation.shape[0], self.output_dim):
            raise RuntimeError("backend returned an invalid output shape")
        return output

    def benchmark(
        self,
        *,
        steps: int = 10_000,
        warmup_steps: int = 100,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        if min(steps, warmup_steps, batch_size) <= 0:
            raise ValueError("benchmark counts must be positive")
        observation = torch.zeros(
            batch_size, self.input_dim, device=self.device, dtype=torch.float32
        )
        for _ in range(warmup_steps):
            self.infer(observation)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        samples = torch.empty(steps, dtype=torch.float64)
        started = time.perf_counter()
        for index in range(steps):
            before = time.perf_counter_ns()
            self.infer(observation)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            samples[index] = (time.perf_counter_ns() - before) / 1_000.0
        elapsed = time.perf_counter() - started
        return BenchmarkResult(
            steps=steps,
            batch_size=batch_size,
            mean_us=float(samples.mean().item()),
            p50_us=float(torch.quantile(samples, 0.50).item()),
            p99_us=float(torch.quantile(samples, 0.99).item()),
            maximum_us=float(samples.max().item()),
            steps_per_second=steps / elapsed,
        )
