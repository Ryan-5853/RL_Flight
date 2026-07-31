from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

import torch

from .environment import SimulationEnvironment
from .types import AdvanceResult


Policy = Callable[[torch.Tensor], torch.Tensor]
StepCallback = Callable[[int, "RealtimeSimulationEnvironment"], None]


@dataclass(frozen=True)
class RealtimeLoopStats:
    """Wall-clock statistics from one real-time inference run."""

    steps: int
    elapsed_s: float
    achieved_hz: float
    mean_compute_us: float
    p50_compute_us: float
    p99_compute_us: float
    max_compute_us: float
    deadline_misses: int
    max_lateness_us: float


class RealtimeSimulationEnvironment:
    """Optimized, explicitly unlogged single-environment execution path.

    The numerical model and checkpoint format are delegated to
    :class:`SimulationEnvironment`.  This facade fixes ``B=1``, compiles the two
    expensive tensor kernels, exposes a reusable packed observation buffer, and
    provides an optional 500 Hz policy loop with deadline telemetry.
    """

    def __init__(
        self,
        environment: SimulationEnvironment,
        *,
        compile_kernels: bool,
        compile_mode: str,
    ) -> None:
        if environment.parallel_count != 1:
            raise ValueError("real-time simulation requires parallel_count=1")
        self._environment = environment
        self._compile_kernels = compile_kernels
        self._compile_mode = compile_mode
        self._compiled = False
        self._observation_buffers: dict[
            tuple[str, tuple[str, ...]], torch.Tensor
        ] = {}
        if compile_kernels:
            self._install_compiled_kernels()

    @classmethod
    def create(
        cls,
        config_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        compile_kernels: bool = True,
        compile_mode: str = "reduce-overhead",
        dynamic_randomization: Mapping[str, Mapping[str, Any]] | None = None,
        dynamic_seed: int | None = None,
    ) -> "RealtimeSimulationEnvironment":
        """Create a ``B=1`` environment without a disk logger.

        ``device="cpu"`` is recommended when the policy also runs on the CPU.
        Use one CUDA device for both simulator and policy if GPU inference is
        required; crossing PCIe in every 2 ms tick defeats this path's purpose.
        """

        environment = SimulationEnvironment.create(
            config_path,
            parallel_count=1,
            device=device,
            dtype=dtype,
            dynamic_randomization=dynamic_randomization,
            dynamic_seed=dynamic_seed,
            logging_enabled=False,
        )
        return cls(
            environment,
            compile_kernels=compile_kernels,
            compile_mode=compile_mode,
        )

    @property
    def device(self) -> torch.device:
        return self._environment.device

    @property
    def dtype(self) -> torch.dtype:
        return self._environment.dtype

    @property
    def compiled(self) -> bool:
        return self._compiled

    @property
    def environment(self) -> SimulationEnvironment:
        """Underlying environment for reset/checkpoint and detailed inspection."""

        return self._environment

    @property
    def observation_layout(self) -> Mapping[str, tuple[str, ...]]:
        """Available field order for packed truth and sensor observations."""

        return MappingProxyType(
            {
                "truth": tuple(self._environment._truth),
                "sensor": tuple(self._environment._sensors),
            }
        )

    def warmup(
        self,
        *,
        steps: int = 3,
        control: torch.Tensor | None = None,
        policy: Policy | None = None,
        observation_source: Literal["truth", "sensor"] = "sensor",
        observation_fields: Sequence[str] | None = None,
    ) -> float:
        """Compile simulator/policy and restore the exact pre-warmup state."""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("warmup steps must be a positive integer")
        if policy is not None and control is not None:
            raise ValueError("warmup accepts either policy or control, not both")
        if control is None:
            control = torch.zeros((1, 5), device=self.device, dtype=self.dtype)
        checkpoint = self._environment.state_dict()
        started_ns = time.perf_counter_ns()
        try:
            # ``inference_mode`` would turn newly assigned state tensors into
            # inference tensors, which then reject reset/checkpoint in-place
            # updates outside that context.  ``no_grad`` keeps the state fully
            # compatible with the regular environment lifecycle.
            with torch.no_grad():
                for _ in range(steps):
                    step_control = control
                    if policy is not None:
                        observation = self.observation_vector(
                            observation_source, observation_fields
                        )
                        step_control = self._validate_policy_control(
                            policy(observation)
                        )
                    self._environment.advance(step_control)
                self._synchronize()
        finally:
            self._environment.load_state_dict(checkpoint)
        return (time.perf_counter_ns() - started_ns) / 1e9

    def observation_vector(
        self,
        source: Literal["truth", "sensor"] = "sensor",
        fields: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Pack selected fields into a reusable ``[1,N]`` device tensor.

        The returned tensor is overwritten by the next call with the same
        ``source`` and field order.  This intentional view-like lifetime avoids
        per-tick observation allocation.
        """

        available = (
            self._environment._truth
            if source == "truth"
            else self._environment._sensors
            if source == "sensor"
            else None
        )
        if available is None:
            raise ValueError("source must be 'truth' or 'sensor'")
        selected = tuple(available) if fields is None else tuple(fields)
        unknown = [name for name in selected if name not in available]
        if unknown:
            raise KeyError(f"unknown {source} observation fields: {unknown}")
        if not selected:
            raise ValueError("at least one observation field is required")

        key = (source, selected)
        width = sum(available[name].numel() for name in selected)
        buffer = self._observation_buffers.get(key)
        if buffer is None:
            buffer = torch.empty((1, width), device=self.device, dtype=self.dtype)
            self._observation_buffers[key] = buffer
        cursor = 0
        for name in selected:
            value = available[name].reshape(1, -1)
            next_cursor = cursor + value.shape[1]
            buffer[:, cursor:next_cursor].copy_(value)
            cursor = next_cursor
        return buffer

    def state_views(
        self,
        source: Literal["truth", "sensor"],
        fields: Sequence[str],
    ) -> tuple[torch.Tensor, ...]:
        """Return current internal tensors without cloning.

        This is a deliberately narrow real-time API.  The tuple itself may be
        retained, but compiled state commit can replace its tensors on the next
        step, so consumers must request fresh views at every control boundary
        and must never mutate them.
        """

        available = (
            self._environment._truth
            if source == "truth"
            else self._environment._sensors
            if source == "sensor"
            else None
        )
        if available is None:
            raise ValueError("source must be 'truth' or 'sensor'")
        selected = tuple(fields)
        unknown = [name for name in selected if name not in available]
        if unknown:
            raise KeyError(f"unknown {source} state fields: {unknown}")
        return tuple(available[name] for name in selected)

    def advance(self, control: torch.Tensor) -> AdvanceResult:
        """Advance exactly one 2 ms simulation tick."""

        return self._environment.advance(control)

    def run_policy(
        self,
        policy: Policy,
        *,
        steps: int,
        observation_source: Literal["truth", "sensor"] = "sensor",
        observation_fields: Sequence[str] | None = None,
        realtime: bool = True,
        spin_us: float = 100.0,
        synchronize_cuda: bool = True,
        disable_gc: bool = True,
        callback: StepCallback | None = None,
    ) -> RealtimeLoopStats:
        """Run policy inference followed by simulation at the 500 Hz timebase.

        ``policy`` receives a reusable ``[1,N]`` tensor and must return a
        floating tensor shaped ``[1,5]`` (or ``[5]``) on the simulator device.
        Keeping both policy and simulator on one device is required for the
        latency numbers to be meaningful.
        """

        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(spin_us) or spin_us < 0:
            raise ValueError("spin_us must be finite and non-negative")

        period_ns = 2_000_000
        spin_ns = int(spin_us * 1_000)
        compute_ns = [0] * steps
        deadline_misses = 0
        max_lateness_ns = 0
        started_ns = time.perf_counter_ns()
        restore_gc = disable_gc and gc.isenabled()
        if restore_gc:
            gc.disable()
        try:
            with torch.no_grad():
                for index in range(steps):
                    tick_started_ns = time.perf_counter_ns()
                    observation = self.observation_vector(
                        observation_source, observation_fields
                    )
                    control = policy(observation)
                    control = self._validate_policy_control(control)
                    result = self._environment.advance(control)
                    if self.device.type == "cuda" and synchronize_cuda:
                        self._synchronize()
                    tick_completed_ns = time.perf_counter_ns()
                    compute_ns[index] = tick_completed_ns - tick_started_ns

                    if (
                        self.device.type != "cuda" or synchronize_cuda
                    ) and not bool(result.valid[0].item()):
                        raise RuntimeError(
                            "simulation became invalid with error code "
                            f"{int(result.error_code[0].item())}"
                        )
                    if callback is not None:
                        callback(index, self)

                    if realtime:
                        deadline_ns = started_ns + (index + 1) * period_ns
                        now_ns = time.perf_counter_ns()
                        if now_ns > deadline_ns:
                            deadline_misses += 1
                            max_lateness_ns = max(
                                max_lateness_ns, now_ns - deadline_ns
                            )
                        else:
                            self._wait_until(deadline_ns, spin_ns)
        finally:
            if restore_gc:
                gc.enable()

        self._synchronize()
        elapsed_s = (time.perf_counter_ns() - started_ns) / 1e9
        ordered = sorted(compute_ns)
        mean_compute_ns = sum(compute_ns) / len(compute_ns)
        return RealtimeLoopStats(
            steps=steps,
            elapsed_s=elapsed_s,
            achieved_hz=steps / elapsed_s,
            mean_compute_us=mean_compute_ns / 1_000.0,
            p50_compute_us=self._percentile(ordered, 0.50) / 1_000.0,
            p99_compute_us=self._percentile(ordered, 0.99) / 1_000.0,
            max_compute_us=max(compute_ns) / 1_000.0,
            deadline_misses=deadline_misses,
            max_lateness_us=max_lateness_ns / 1_000.0,
        )

    def close(self) -> None:
        self._environment.close()

    def __enter__(self) -> "RealtimeSimulationEnvironment":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _install_compiled_kernels(self) -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("real-time execution requires torch.compile")
        environment = self._environment
        self._environment._dynamics.step = torch.compile(
            self._environment._dynamics.step,
            fullgraph=True,
            mode=self._compile_mode,
        )
        self._environment._sensor_kernel.step = torch.compile(
            self._environment._sensor_kernel.step,
            fullgraph=True,
            mode=self._compile_mode,
        )
        truth_names = tuple(environment._truth)

        def finite_kernel(
            values: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            finite = torch.ones(1, dtype=torch.bool, device=values[0].device)
            for value in values:
                finite = finite & torch.isfinite(value).reshape(1, -1).all(dim=1)
            return finite

        def commit_kernel(
            current: tuple[torch.Tensor, ...],
            candidate: tuple[torch.Tensor, ...],
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            committed: list[torch.Tensor] = []
            for old, new in zip(current, candidate):
                expanded = mask.reshape(1, *([1] * (old.ndim - 1)))
                committed.append(torch.where(expanded, new, old))
            return tuple(committed)

        compiled_finite = torch.compile(
            finite_kernel,
            fullgraph=True,
            mode=self._compile_mode,
        )
        compiled_commit = torch.compile(
            commit_kernel,
            fullgraph=True,
            mode=self._compile_mode,
        )

        def state_is_finite(
            state: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
            if tuple(state) != truth_names:
                raise RuntimeError("dynamics kernel changed the truth-state schema")
            return compiled_finite(tuple(state[name] for name in truth_names))

        def commit_state(
            candidate: Mapping[str, torch.Tensor],
            mask: torch.Tensor,
        ) -> None:
            current_values = tuple(
                environment._truth[name] for name in truth_names
            )
            candidate_values = tuple(candidate[name] for name in truth_names)
            committed = compiled_commit(current_values, candidate_values, mask)
            environment._truth = dict(zip(truth_names, committed))

        environment._state_is_finite = state_is_finite
        environment._commit_state = commit_state
        self._compiled = True

    def _validate_policy_control(self, control: torch.Tensor) -> torch.Tensor:
        if not isinstance(control, torch.Tensor):
            raise TypeError("policy must return a torch.Tensor")
        if control.shape == (5,):
            control = control.reshape(1, 5)
        if control.shape != (1, 5):
            raise ValueError("policy control must have shape (1,5) or (5,)")
        if control.device != self.device or control.dtype != self.dtype:
            raise ValueError(
                f"policy control must use {self.device}/{self.dtype}, "
                f"got {control.device}/{control.dtype}"
            )
        return control

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _wait_until(deadline_ns: int, spin_ns: int) -> None:
        remaining_ns = deadline_ns - time.perf_counter_ns()
        if remaining_ns > spin_ns:
            time.sleep((remaining_ns - spin_ns) / 1e9)
        while time.perf_counter_ns() < deadline_ns:
            pass

    @staticmethod
    def _percentile(ordered: Sequence[int], fraction: float) -> int:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]
