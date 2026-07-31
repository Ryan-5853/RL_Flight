from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .config import ExperimentConfig
from .evaluation import FixedEvaluationSuite, run_fixed_evaluation


@dataclass(frozen=True)
class EvaluationJob:
    """One immutable policy/checkpoint pair submitted to the evaluation worker."""

    job_id: str
    control_steps: int
    directory: Path
    source_checkpoint: Path
    pinned_checkpoint: Path
    policy_checkpoint: Path
    result_path: Path
    log_path: Path
    source_checkpoint_sha256: str


@dataclass(frozen=True)
class EvaluationOutcome:
    """A completed worker result awaiting main-process recording and acknowledgement."""

    job: EvaluationJob
    status: str
    evaluation_directory: Path | None
    report: Mapping[str, Any] | None
    error: str | None
    traceback: str | None
    exit_code: int | None


@dataclass
class _RunningEvaluation:
    job: EvaluationJob
    process: Any | None
    outcome: EvaluationOutcome | None = None
    delivered: bool = False


WorkerTarget = Callable[
    [
        ExperimentConfig,
        FixedEvaluationSuite,
        Path,
        Path,
        Path,
        Path,
        str,
        Path,
    ],
    None,
]


class AsyncEvaluationManager:
    """Run bounded spawned workers and coalesce excess requests to one latest job."""

    def __init__(
        self,
        config: ExperimentConfig,
        suite: FixedEvaluationSuite,
        run_directory: Path,
        *,
        worker_target: WorkerTarget | None = None,
        multiprocessing_context: Any | None = None,
    ) -> None:
        self.config = config
        self.suite = suite
        self.run_directory = Path(run_directory).resolve()
        self.output_root = self.run_directory / "evaluations"
        self.jobs_root = self.run_directory / "evaluation_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._worker_target = worker_target or _evaluation_worker
        self.max_in_flight = config.evaluation.execution.max_in_flight
        self._context = (
            multiprocessing_context
            if multiprocessing_context is not None
            else multiprocessing.get_context("spawn")
        )
        self._running: dict[str, _RunningEvaluation] = {}
        self._pending: EvaluationJob | None = None
        self._ready_job_ids: deque[str] = deque()

    @property
    def has_work(self) -> bool:
        return bool(self._running) or self._pending is not None

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def running_steps(self) -> tuple[int, ...]:
        return tuple(
            sorted(slot.job.control_steps for slot in self._running.values())
        )

    @property
    def pending_step(self) -> int | None:
        return None if self._pending is None else self._pending.control_steps

    def submit(
        self,
        checkpoint: Path,
        control_steps: int,
        actor_state: Mapping[str, Any],
    ) -> str:
        """Pin a full checkpoint and submit or replace the latest pending policy."""

        job = self._prepare_job(Path(checkpoint), control_steps, actor_state)
        if len(self._running) < self.max_in_flight:
            try:
                self._start(job)
            except BaseException:
                self._cleanup_job(job, preserve_diagnostics=True)
                raise
            return "started"
        previous_pending = self._pending
        self._pending = job
        if previous_pending is not None:
            self._cleanup_job(previous_pending, preserve_diagnostics=False)
            return "replaced_pending"
        return "queued"

    def poll(self) -> EvaluationOutcome | None:
        """Return one finished outcome without releasing its pinned checkpoint."""

        self._collect_finished_processes()
        while self._ready_job_ids:
            job_id = self._ready_job_ids.popleft()
            slot = self._running.get(job_id)
            if slot is None or slot.outcome is None or slot.delivered:
                continue
            slot.delivered = True
            return slot.outcome
        return None

    def wait(self) -> EvaluationOutcome:
        """Wait for the current worker while retaining the pin until acknowledge()."""

        if not self._running:
            raise RuntimeError("there is no running evaluation to wait for")
        while True:
            outcome = self.poll()
            if outcome is not None:
                return outcome
            active = next(
                (
                    slot.process
                    for slot in self._running.values()
                    if slot.process is not None
                ),
                None,
            )
            if active is None:
                raise RuntimeError("async evaluation workers disappeared")
            active.join(timeout=0.25)

    def acknowledge(
        self,
        outcome: EvaluationOutcome,
        *,
        launch_pending: bool = True,
    ) -> None:
        """Release one processed job and immediately launch the newest pending one."""

        slot = self._running.get(outcome.job.job_id)
        if slot is None:
            raise ValueError("evaluation outcome does not match the running job")
        if not slot.delivered or slot.outcome is not outcome:
            raise RuntimeError("evaluation outcome must be polled before acknowledgement")
        finished = slot.job
        pending = self._pending
        del self._running[outcome.job.job_id]
        self._cleanup_job(
            finished,
            preserve_diagnostics=outcome.status != "completed",
        )
        if (
            pending is not None
            and launch_pending
            and len(self._running) < self.max_in_flight
        ):
            self._pending = None
            try:
                self._start(pending)
            except BaseException:
                self._cleanup_job(pending, preserve_diagnostics=True)
                raise
        elif pending is not None and not launch_pending:
            self._pending = None
            self._cleanup_job(pending, preserve_diagnostics=False)

    def shutdown(self, *, cancel: bool) -> None:
        """Stop outstanding work and release every transient checkpoint pin."""

        if cancel:
            for slot in self._running.values():
                process = slot.process
                if process is not None and process.is_alive():
                    process.terminate()
        for slot in self._running.values():
            process = slot.process
            if process is None:
                continue
            if cancel:
                process.join(timeout=10.0)
                if process.is_alive():
                    process.kill()
                    process.join()
            elif process.is_alive():
                process.join()
            else:
                process.join()
            process.close()
            slot.process = None
        for slot in self._running.values():
            self._cleanup_job(slot.job, preserve_diagnostics=False)
        if self._pending is not None:
            self._cleanup_job(self._pending, preserve_diagnostics=False)
        self._running.clear()
        self._pending = None
        self._ready_job_ids.clear()

    def _prepare_job(
        self,
        checkpoint: Path,
        control_steps: int,
        actor_state: Mapping[str, Any],
    ) -> EvaluationJob:
        source = checkpoint.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"evaluation checkpoint does not exist: {source}")
        source_checksum = source.with_suffix(source.suffix + ".sha256")
        digest = _read_sha256_sidecar(source_checksum)
        job_id = f"step_{control_steps}_{uuid.uuid4().hex[:8]}"
        directory = self.jobs_root / job_id
        directory.mkdir(parents=False, exist_ok=False)
        pinned = directory / f"step_{control_steps}.pt"
        pinned_checksum = pinned.with_suffix(pinned.suffix + ".sha256")
        policy = directory / f"policy_step_{control_steps}.pt"
        result = directory / "result.json"
        log = directory / "worker.log"
        job = EvaluationJob(
            job_id=job_id,
            control_steps=control_steps,
            directory=directory,
            source_checkpoint=source,
            pinned_checkpoint=pinned,
            policy_checkpoint=policy,
            result_path=result,
            log_path=log,
            source_checkpoint_sha256=digest,
        )
        try:
            # A hard link pins the full resumable state without duplicating its data.
            os.link(source, pinned)
            os.link(source_checksum, pinned_checksum)
            _write_policy_checkpoint(policy, control_steps, actor_state)
            _atomic_write_json(
                directory / "job.json",
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "global_control_steps": control_steps,
                    "source_checkpoint": str(source),
                    "source_checkpoint_sha256": digest,
                    "pinned_checkpoint": pinned.name,
                    "policy_checkpoint": policy.name,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        except BaseException:
            self._cleanup_job(job, preserve_diagnostics=False)
            raise
        return job

    def _start(self, job: EvaluationJob) -> None:
        if len(self._running) >= self.max_in_flight:
            raise RuntimeError("all asynchronous evaluation worker slots are occupied")
        process = self._context.Process(
            target=self._worker_target,
            args=(
                self.config,
                self.suite,
                job.policy_checkpoint,
                self.output_root,
                job.result_path,
                job.source_checkpoint,
                job.source_checkpoint_sha256,
                job.log_path,
            ),
            name=f"flight-evaluation-{job.control_steps}",
            daemon=False,
        )
        try:
            process.start()
        except BaseException:
            process.close()
            raise
        self._running[job.job_id] = _RunningEvaluation(
            job=job,
            process=process,
        )

    def _collect_finished_processes(self) -> None:
        for job_id, slot in tuple(self._running.items()):
            process = slot.process
            if process is None or slot.outcome is not None or process.is_alive():
                continue
            process.join()
            exit_code = process.exitcode
            process.close()
            slot.process = None
            slot.outcome = self._read_outcome(slot.job, exit_code)
            self._ready_job_ids.append(job_id)

    def _read_outcome(
        self, job: EvaluationJob, exit_code: int | None
    ) -> EvaluationOutcome:
        if not job.result_path.is_file():
            return EvaluationOutcome(
                job=job,
                status="failed",
                evaluation_directory=None,
                report=None,
                error=(
                    "evaluation worker exited without an atomic result "
                    f"(exit_code={exit_code})"
                ),
                traceback=None,
                exit_code=exit_code,
            )
        try:
            result = json.loads(job.result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EvaluationOutcome(
                job=job,
                status="failed",
                evaluation_directory=None,
                report=None,
                error=f"evaluation result is unreadable: {exc}",
                traceback=None,
                exit_code=exit_code,
            )
        status = str(result.get("status", "failed"))
        if status != "completed":
            return EvaluationOutcome(
                job=job,
                status="failed",
                evaluation_directory=None,
                report=None,
                error=str(result.get("error", "evaluation worker failed")),
                traceback=(
                    None
                    if result.get("traceback") is None
                    else str(result["traceback"])
                ),
                exit_code=exit_code,
            )
        evaluation_directory = Path(
            str(result.get("evaluation_directory", ""))
        ).resolve()
        report_path = evaluation_directory / "report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EvaluationOutcome(
                job=job,
                status="failed",
                evaluation_directory=evaluation_directory,
                report=None,
                error=f"evaluation report is unreadable: {exc}",
                traceback=None,
                exit_code=exit_code,
            )
        return EvaluationOutcome(
            job=job,
            status="completed",
            evaluation_directory=evaluation_directory,
            report=report,
            error=None,
            traceback=None,
            exit_code=exit_code,
        )

    @staticmethod
    def _cleanup_job(
        job: EvaluationJob, *, preserve_diagnostics: bool
    ) -> None:
        transient = (
            job.pinned_checkpoint,
            job.pinned_checkpoint.with_suffix(
                job.pinned_checkpoint.suffix + ".sha256"
            ),
            job.policy_checkpoint,
            job.policy_checkpoint.with_suffix(
                job.policy_checkpoint.suffix + ".sha256"
            ),
            job.policy_checkpoint.with_suffix(".tmp"),
            job.policy_checkpoint.with_suffix(
                job.policy_checkpoint.suffix + ".sha256.tmp"
            ),
            job.result_path.with_suffix(".tmp"),
            (job.directory / "job.json").with_suffix(".tmp"),
        )
        for path in transient:
            path.unlink(missing_ok=True)
        if preserve_diagnostics:
            return
        for name in ("job.json", "result.json", "worker.log"):
            (job.directory / name).unlink(missing_ok=True)
        try:
            job.directory.rmdir()
        except FileNotFoundError:
            pass


def _evaluation_worker(
    config: ExperimentConfig,
    suite: FixedEvaluationSuite,
    policy_checkpoint: Path,
    output_root: Path,
    result_path: Path,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    log_path: Path,
) -> None:
    """Spawn target: evaluate a small policy snapshot and atomically publish result."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        # Redirect Python and native stdout/stderr so the worker cannot corrupt the
        # training process's live terminal progress display.
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            torch.use_deterministic_algorithms(
                config.seeds.deterministic_algorithms
            )
            torch.backends.cudnn.benchmark = config.seeds.cudnn_benchmark
            result = run_fixed_evaluation(
                config,
                policy_checkpoint,
                suite,
                output_root=output_root,
                report_checkpoint_path=source_checkpoint,
                report_checkpoint_sha256=source_checkpoint_sha256,
            )
            payload = {
                "schema_version": 1,
                "status": "completed",
                "global_control_steps": int(
                    result["report"]["checkpoint_global_control_steps"]
                ),
                "evaluation_directory": str(result["evaluation_directory"]),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        except BaseException as exc:
            payload = {
                "schema_version": 1,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        _atomic_write_json(result_path, payload)


def _write_policy_checkpoint(
    destination: Path,
    control_steps: int,
    actor_state: Mapping[str, Any],
) -> None:
    cpu_actor: dict[str, torch.Tensor] = {}
    for name, value in actor_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"actor state must contain tensors, got {type(value).__name__} "
                f"for {name!r}"
            )
        cpu_actor[str(name)] = value.detach().to(device="cpu", copy=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(
        {
            "checkpoint_schema_version": "evaluation-policy-v1",
            "global_control_steps": int(control_steps),
            "actor": cpu_actor,
        },
        temporary,
    )
    _fsync_file(temporary)
    temporary.replace(destination)
    digest = _sha256(destination)
    checksum = destination.with_suffix(destination.suffix + ".sha256")
    checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
    checksum_tmp.write_text(digest + "\n", encoding="ascii")
    _fsync_file(checksum_tmp)
    checksum_tmp.replace(checksum)
    _fsync_directory(destination.parent)


def _read_sha256_sidecar(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"checkpoint checksum is missing: {path}")
    digest = path.read_text(encoding="ascii").strip()
    if len(digest) != 64:
        raise ValueError(f"checkpoint checksum is invalid: {path}")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"checkpoint checksum is invalid: {path}") from exc
    return digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _fsync_file(temporary)
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
