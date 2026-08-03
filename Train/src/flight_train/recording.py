from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from tensordict import TensorDictBase

from .config import ExperimentConfig


class InsufficientDiskSpaceError(RuntimeError):
    """Checkpoint write was rejected to preserve a configured disk reserve."""

    def __init__(
        self,
        path: Path,
        *,
        available_bytes: int,
        required_bytes: int,
        reserve_bytes: int,
        estimated_checkpoint_bytes: int,
        operation: str,
    ) -> None:
        self.path = path
        self.available_bytes = available_bytes
        self.required_bytes = required_bytes
        self.reserve_bytes = reserve_bytes
        self.estimated_checkpoint_bytes = estimated_checkpoint_bytes
        self.operation = operation
        super().__init__(
            f"insufficient disk space for {operation}: path={path}, "
            f"available={available_bytes} bytes, required={required_bytes} bytes "
            f"(checkpoint estimate={estimated_checkpoint_bytes} bytes, "
            f"reserve={reserve_bytes} bytes)"
        )


class RunRecorder:
    """训练张量与 CPU/文件系统之间唯一的低频记录边界。

    每次运行创建不可复用的新目录，归档解析后的配置、软件版本、仿真配置哈希、
    指标和 checkpoint。热路径只把设备上的标量张量传入 ``metrics``。
    """

    def __init__(self, config: ExperimentConfig) -> None:
        # 规范化 JSON 后计算指纹：相同配置内容不受键顺序和空白差异影响。
        resolved = json.dumps(config.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fingerprint = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = str(uuid.uuid4())
        self.directory = config.run.output_root / config.name / f"{stamp}_{fingerprint}_{self.run_id[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._checkpoint_directory = self.directory / "checkpoints"
        self._checkpoint_directory.mkdir()
        self._checkpoint_index_path = self._checkpoint_directory / "index.json"
        self._checkpoint_keep_last = config.checkpoint.keep_last
        self._minimum_free_space_bytes = (
            config.checkpoint.minimum_free_space_bytes
        )
        self._checkpoint_size_estimate = (
            config.checkpoint.resume_from.stat().st_size
            if config.checkpoint.resume_from is not None
            else 0
        )
        self._checkpoint_entries: list[dict[str, Any]] = []
        self._best_evaluation_ranks: dict[str, tuple[float, ...]] = {}
        (self.directory / "config.json").write_text(
            json.dumps(config.raw, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._manifest_path = self.directory / "manifest.json"
        manifest = {
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchrl": _torchrl_version(),
            "device": config.run.device,
            "dtype": config.run.dtype,
            "simulator_config": str(config.simulator_config),
            "simulator_config_sha256": _sha256(config.simulator_config),
            "entrypoint": config.entrypoint.type,
            "entrypoint_version": config.entrypoint.version,
            "reward_calculator": config.reward.calculator.type,
            "reward_calculator_version": config.reward.calculator.version,
            "model_architecture": config.model.architecture,
            "algorithm": config.algorithm_name,
            "observation_profile": config.control_contract.observation_profile,
            "observation_history_mode": (
                config.control_contract.observation_history_mode
            ),
            "observation_history_frames": (
                config.control_contract.observation_history_frames
            ),
            "observation_history_stride_steps": (
                config.control_contract.observation_history_stride_steps
            ),
            "observation_history_dense_action_steps": (
                config.control_contract
                .observation_history_dense_action_steps
            ),
            "observation_history_sparse_physical_frames": (
                config.control_contract
                .observation_history_sparse_physical_frames
            ),
            "observation_history_sparse_physical_stride_steps": (
                config.control_contract
                .observation_history_sparse_physical_stride_steps
            ),
            "control_contract": config.control_contract.version,
            "action_transform": (
                config.control_contract.action_transform_type
            ),
            "lower_motor_upper_ratio": (
                config.control_contract.lower_motor_upper_ratio
            ),
            "policy_action_fields": list(config.control_contract.policy_action_fields),
            "external_action_fields": list(config.control_contract.external_action_fields),
            "simulator_command_fields": list(config.control_contract.simulator_command_fields),
            "policy_action_trim": list(
                config.control_contract.policy_action_trim
            ),
            "policy_action_residual_scale": list(
                config.control_contract.policy_action_residual_scale
            ),
            "command_source": config.command_source.type,
            "command_source_version": config.command_source.version,
            "command_source_seed": config.command_source.seed,
            "static_randomization_seed": config.static_randomization.seed,
            "dynamic_randomization_seed": config.dynamic_randomization.seed,
            "exact_resume_supported": True,
            "checkpoint_interval_control_steps": config.checkpoint.interval_control_steps,
            "checkpoint_keep_last": config.checkpoint.keep_last,
            "checkpoint_minimum_free_space_bytes": (
                config.checkpoint.minimum_free_space_bytes
            ),
            "evaluation_enabled": config.evaluation.enabled,
            "evaluation_interval_control_steps": (
                config.evaluation.interval_control_steps
            ),
            "evaluation_execution_mode": config.evaluation.execution.mode,
            "evaluation_max_in_flight": (
                config.evaluation.execution.max_in_flight
            ),
            "evaluation_pending_policy": (
                config.evaluation.execution.pending_policy
            ),
            "evaluation_wait_for_final": (
                config.evaluation.execution.wait_for_final
            ),
            "evaluation_failure_policy": (
                config.evaluation.execution.failure_policy
            ),
        }
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._metrics = (self.directory / "metrics.jsonl").open("a", encoding="utf-8")

    def metrics(
        self, control_steps: int, values: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        """记录一次更新的标量指标；这里允许发生每轮一次的设备同步。"""

        # 同步严格位于采样和反向传播之外，不进入逐控制步热路径。
        record: dict[str, Any] = {
            "global_control_steps": control_steps,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in values.items():
            if value.numel() == 1:
                record[key] = float(value.detach().item())
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.ensure_checkpoint_capacity(
            self._checkpoint_size_estimate + len(line.encode("utf-8"))
        )
        self._metrics.write(line)
        self._metrics.flush()
        return record

    def evaluation(
        self, control_steps: int, result: Mapping[str, Any]
    ) -> None:
        """追加一次训练中固定评测摘要，不与 PPO 指标行混在一起。"""

        report = result["report"]
        record = {
            "global_control_steps": control_steps,
            "utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_directory": str(result["evaluation_directory"]),
            "total_score": float(report["total_score"]),
            "status": str(report["status"]),
            "scenarios": {
                name: {
                    "total_score": float(values["total_score"]),
                    "survival_time_s_mean": float(
                        values["metrics"]["survival_time_s"]["mean"]
                    ),
                    "attitude_rmse_deg_mean": float(
                        values["metrics"]["attitude_rmse_deg"]["mean"]
                    ),
                    "roll_pitch_rmse_deg_mean": float(
                        values["metrics"]["roll_pitch_rmse_deg"]["mean"]
                    ),
                    "yaw_rate_rmse_rad_s_mean": float(
                        values["metrics"]["yaw_rate_rmse_rad_s"]["mean"]
                    ),
                    "motor_total_variation_per_s_mean": float(
                        values["metrics"]["motor_total_variation_per_s"]["mean"]
                    ),
                    "servo_common_total_variation_per_s_mean": float(
                        values["metrics"][
                            "servo_common_total_variation_per_s"
                        ]["mean"]
                    ),
                    "servo_cyclic_total_variation_per_s_mean": float(
                        values["metrics"][
                            "servo_cyclic_total_variation_per_s"
                        ]["mean"]
                    ),
                    "roll_pitch_error_band_0_5_2_hz_rms_deg_mean": float(
                        values["metrics"][
                            "roll_pitch_error_band_0_5_2_hz_rms_deg"
                        ]["mean"]
                    ),
                }
                for name, values in report["scenarios"].items()
            },
        }
        path = self.directory / "evaluations.jsonl"
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.ensure_checkpoint_capacity(
            self._checkpoint_size_estimate + len(line.encode("utf-8"))
        )
        with path.open("a", encoding="utf-8") as output:
            output.write(line)

    def evaluation_failure(
        self,
        control_steps: int,
        *,
        error: str,
        detail: str | None = None,
    ) -> None:
        """Record an asynchronous evaluation failure without stopping training."""

        record: dict[str, Any] = {
            "global_control_steps": control_steps,
            "utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": error,
        }
        if detail:
            record["detail"] = detail
        path = self.directory / "evaluations.jsonl"
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.ensure_checkpoint_capacity(
            self._checkpoint_size_estimate + len(line.encode("utf-8"))
        )
        with path.open("a", encoding="utf-8") as output:
            output.write(line)

    def promote_best_evaluation_checkpoint(
        self,
        checkpoint: Path,
        *,
        control_steps: int,
        hover_survival_s: float,
        hover_roll_pitch_rmse_deg: float,
        hover_yaw_rate_rmse_rad_s: float,
        total_score: float,
        quality_passed: bool = True,
    ) -> bool:
        """质量门槛通过后，按综合评测总分保留 best。"""

        if not quality_passed:
            return False
        # 总分已综合生存、跟踪、动作与响应；悬停生存仅在总分相同时打破平局，
        # 避免极小的生存时长差异覆盖明显更好的稳定质量。
        rank = (total_score, hover_survival_s)
        return self._promote_evaluation_checkpoint(
            "fixed",
            rank,
            checkpoint,
            control_steps=control_steps,
            hover_survival_s=hover_survival_s,
            hover_roll_pitch_rmse_deg=hover_roll_pitch_rmse_deg,
            hover_yaw_rate_rmse_rad_s=hover_yaw_rate_rmse_rad_s,
            total_score=total_score,
            quality_passed=True,
        )

    def promote_evaluation_checkpoints(
        self,
        checkpoint: Path,
        *,
        control_steps: int,
        hover_survival_s: float,
        hover_roll_pitch_rmse_deg: float,
        hover_yaw_rate_rmse_rad_s: float,
        total_score: float,
        minimum_hover_survival_s: float,
        quality_passed: bool,
    ) -> tuple[str, ...]:
        """独立保留综合、直立、偏航诊断和完全合格的评测 checkpoint。

        多个名字通过硬链接指向 checkpoint 数据，不会为同一次评测重复占用
        大型模型文件的磁盘块。周期 checkpoint 被轮转删除后，best 链接仍然
        独立有效。
        """

        survived = hover_survival_s >= minimum_hover_survival_s
        ranks: dict[str, tuple[float, ...]] = {
            # 综合分是最终评测目标，生存时间仅用于打破平局。
            "total": (total_score, hover_survival_s),
            # 先要求达到最低生存门槛；达标后优先最小化横滚/俯仰误差。
            # 尚无达标候选时，优先保留生存时间最长的恢复点。
            "upright": (
                (
                    1.0,
                    -hover_roll_pitch_rmse_deg,
                    hover_survival_s,
                    total_score,
                )
                if survived
                else (
                    0.0,
                    hover_survival_s,
                    -hover_roll_pitch_rmse_deg,
                    total_score,
                )
            ),
            # yaw best 是诊断模型，不冒充安全可用模型；生存和直立用于平局。
            "yaw": (
                -hover_yaw_rate_rmse_rad_s,
                hover_survival_s,
                -hover_roll_pitch_rmse_deg,
                total_score,
            ),
        }
        updated: list[str] = []
        for selection, rank in ranks.items():
            if self._promote_evaluation_checkpoint(
                selection,
                rank,
                checkpoint,
                control_steps=control_steps,
                hover_survival_s=hover_survival_s,
                hover_roll_pitch_rmse_deg=hover_roll_pitch_rmse_deg,
                hover_yaw_rate_rmse_rad_s=hover_yaw_rate_rmse_rad_s,
                total_score=total_score,
                quality_passed=quality_passed,
            ):
                updated.append(selection)
        if quality_passed and self.promote_best_evaluation_checkpoint(
            checkpoint,
            control_steps=control_steps,
            hover_survival_s=hover_survival_s,
            hover_roll_pitch_rmse_deg=hover_roll_pitch_rmse_deg,
            hover_yaw_rate_rmse_rad_s=hover_yaw_rate_rmse_rad_s,
            total_score=total_score,
        ):
            updated.append("fixed")
        return tuple(updated)

    def _promote_evaluation_checkpoint(
        self,
        selection: str,
        rank: tuple[float, ...],
        checkpoint: Path,
        *,
        control_steps: int,
        hover_survival_s: float,
        hover_roll_pitch_rmse_deg: float,
        hover_yaw_rate_rmse_rad_s: float,
        total_score: float,
        quality_passed: bool,
    ) -> bool:
        current_rank = self._best_evaluation_ranks.get(selection)
        if current_rank is not None and rank <= current_rank:
            return False
        stem = (
            "best_fixed_evaluation"
            if selection == "fixed"
            else f"best_{selection}_evaluation"
        )
        destination = self._checkpoint_directory / f"{stem}.pt"
        temporary = destination.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        try:
            os.link(checkpoint, temporary)
        except OSError:
            # 非同一文件系统或不支持硬链接时仍保持正确性。
            shutil.copyfile(checkpoint, temporary)
            _fsync_file(temporary)
        temporary.replace(destination)
        digest = _checkpoint_sha256(checkpoint)
        checksum = destination.with_suffix(destination.suffix + ".sha256")
        checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
        checksum_tmp.write_text(digest + "\n", encoding="ascii")
        _fsync_file(checksum_tmp)
        checksum_tmp.replace(checksum)
        metadata = {
            "selection": selection,
            "selection_rank": list(rank),
            "global_control_steps": control_steps,
            "hover_survival_s": hover_survival_s,
            "hover_roll_pitch_rmse_deg": hover_roll_pitch_rmse_deg,
            "hover_yaw_rate_rmse_rad_s": hover_yaw_rate_rmse_rad_s,
            "quality_gate_passed": quality_passed,
            "total_score": total_score,
            "source_checkpoint": checkpoint.name,
            "file": destination.name,
            "sha256": digest,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = self._checkpoint_directory / f"{stem}.json"
        metadata_tmp = metadata_path.with_suffix(".tmp")
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _fsync_file(metadata_tmp)
        metadata_tmp.replace(metadata_path)
        _fsync_directory(self._checkpoint_directory)
        self._best_evaluation_ranks[selection] = rank
        return True

    def checkpoint(
        self, control_steps: int, state: Mapping[str, Any], *, kind: str
    ) -> Path:
        """先写临时文件再原子替换，避免中断留下半个 checkpoint。"""

        if kind not in {"periodic", "final", "interrupt"}:
            raise ValueError(f"unsupported checkpoint kind: {kind}")
        destination = self._checkpoint_directory / f"step_{control_steps}.pt"
        temporary = destination.with_suffix(".tmp")
        state_estimate = _estimated_torch_save_bytes(state)
        self._checkpoint_size_estimate = max(
            self._checkpoint_size_estimate,
            state_estimate,
        )
        self.ensure_checkpoint_capacity(self._checkpoint_size_estimate)
        temporary.unlink(missing_ok=True)
        try:
            torch.save(dict(state), temporary)
            _fsync_file(temporary)
        except BaseException as exc:
            available = shutil.disk_usage(self._checkpoint_directory).free
            partial_bytes = temporary.stat().st_size if temporary.exists() else 0
            if (
                _is_no_space_error(exc)
                or available < self._minimum_free_space_bytes
            ):
                estimate = max(
                    self._checkpoint_size_estimate,
                    partial_bytes,
                )
                error = self._space_error(
                    estimate,
                    operation=f"write checkpoint at step {control_steps}",
                    available_bytes=available,
                )
                temporary.unlink(missing_ok=True)
                raise error from exc
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(destination)
        _fsync_directory(self._checkpoint_directory)
        self._checkpoint_size_estimate = max(
            self._checkpoint_size_estimate,
            destination.stat().st_size,
        )
        digest = _sha256(destination)
        checksum = destination.with_suffix(destination.suffix + ".sha256")
        checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
        checksum_tmp.write_text(digest + "\n", encoding="ascii")
        _fsync_file(checksum_tmp)
        checksum_tmp.replace(checksum)
        _fsync_directory(self._checkpoint_directory)
        entry = {
            "global_control_steps": control_steps,
            "kind": kind,
            "file": destination.name,
            "sha256": digest,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._checkpoint_entries = [
            item
            for item in self._checkpoint_entries
            if item["global_control_steps"] != control_steps
        ]
        self._checkpoint_entries.append(entry)
        self._checkpoint_entries.sort(key=lambda item: item["global_control_steps"])
        removed_entries: list[dict[str, Any]] = []
        while len(self._checkpoint_entries) > self._checkpoint_keep_last:
            removed_entries.append(self._checkpoint_entries.pop(0))
        # 先提交不再引用旧文件的新索引。若随后清理时进程终止，最坏只留下
        # 未被索引引用的孤立旧文件，不会让 index.json 指向已删除 checkpoint。
        self._write_checkpoint_index()
        for removed in removed_entries:
            removed_path = self._checkpoint_directory / removed["file"]
            removed_path.unlink(missing_ok=True)
            removed_path.with_suffix(removed_path.suffix + ".sha256").unlink(
                missing_ok=True
            )
        if removed_entries:
            _fsync_directory(self._checkpoint_directory)
        return destination

    def ensure_checkpoint_capacity(
        self, estimated_checkpoint_bytes: int | None = None
    ) -> None:
        """Reject work before the next atomic checkpoint would consume the reserve."""

        estimate = max(
            0,
            (
                self._checkpoint_size_estimate
                if estimated_checkpoint_bytes is None
                else estimated_checkpoint_bytes
            ),
        )
        available = shutil.disk_usage(self._checkpoint_directory).free
        required = estimate + self._minimum_free_space_bytes
        if available < required:
            raise self._space_error(
                estimate,
                operation="reserve capacity for the next checkpoint",
                available_bytes=available,
            )

    @property
    def latest_checkpoint_step(self) -> int | None:
        if not self._checkpoint_entries:
            return None
        return int(self._checkpoint_entries[-1]["global_control_steps"])

    def _space_error(
        self,
        estimate: int,
        *,
        operation: str,
        available_bytes: int,
    ) -> InsufficientDiskSpaceError:
        return InsufficientDiskSpaceError(
            self._checkpoint_directory,
            available_bytes=available_bytes,
            required_bytes=estimate + self._minimum_free_space_bytes,
            reserve_bytes=self._minimum_free_space_bytes,
            estimated_checkpoint_bytes=estimate,
            operation=operation,
        )

    def _write_checkpoint_index(self) -> None:
        payload = {
            "schema_version": 1,
            "keep_last": self._checkpoint_keep_last,
            "checkpoints": self._checkpoint_entries,
        }
        temporary = self._checkpoint_index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _fsync_file(temporary)
        temporary.replace(self._checkpoint_index_path)
        _fsync_directory(self._checkpoint_directory)

    def relabel_checkpoint(self, control_steps: int, *, kind: str) -> None:
        """同一安全边界已保存时只更新类型，避免重复序列化大型状态。"""

        if kind not in {"periodic", "final", "interrupt"}:
            raise ValueError(f"unsupported checkpoint kind: {kind}")
        for entry in self._checkpoint_entries:
            if entry["global_control_steps"] == control_steps:
                entry["kind"] = kind
                self._write_checkpoint_index()
                return
        raise ValueError(f"checkpoint step is not indexed: {control_steps}")

    def bind_environment(
        self,
        *,
        batch_id: str,
        instance_ids: tuple[str, ...],
        simulator_log_directory: Path,
    ) -> None:
        """把训练 run 与 SimEnv 物理日志、实例身份建立可追溯关联。"""

        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "simulator_batch_id": batch_id,
                "simulator_instance_ids": list(instance_ids),
                "simulator_log_directory": str(simulator_log_directory),
            }
        )
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self._manifest_path)

    def bind_resume(
        self,
        *,
        checkpoint: Path,
        parent_run_id: str | None,
        mode: str = "exact",
        source_global_control_steps: int | None = None,
    ) -> None:
        """记录精确续训或策略热启动来源，新 run 保持独立身份。"""

        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "resume_mode": mode,
                "resume_checkpoint": str(checkpoint),
                "parent_run_id": parent_run_id,
                "resume_source_global_control_steps": source_global_control_steps,
            }
        )
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self._manifest_path)

    def close(
        self,
        status: str,
        control_steps: int,
        *,
        reason: str | None = None,
        detail: str | None = None,
        last_checkpoint_step: int | None = None,
    ) -> None:
        """关闭指标流并写入本次运行的最终状态。"""

        if not self._metrics.closed:
            self._metrics.close()
        payload: dict[str, Any] = {
            "status": status,
            "global_control_steps": control_steps,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
        }
        if reason is not None:
            payload["reason"] = reason
        if detail is not None:
            payload["detail"] = detail
        checkpoint_step = (
            self.latest_checkpoint_step
            if last_checkpoint_step is None
            else last_checkpoint_step
        )
        if checkpoint_step is not None:
            payload["last_checkpoint_step"] = checkpoint_step
        destination = self.directory / "status.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _fsync_file(temporary)
        temporary.replace(destination)
        _fsync_directory(self.directory)


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大型配置附件一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimated_torch_save_bytes(state: Mapping[str, Any]) -> int:
    """Estimate serialized tensor storage plus conservative container overhead."""

    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int, int]] = set()

    def visit(value: Any) -> int:
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storages:
                return 0
            seen_storages.add(key)
            return storage.nbytes()
        if isinstance(value, TensorDictBase):
            return sum(visit(value.get(key)) for key in value.keys())
        if isinstance(value, Mapping):
            return sum(visit(child) for child in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(child) for child in value)
        return 0

    tensor_bytes = visit(state)
    # Zip records, pickle metadata, alignment, and non-tensor objects are small
    # relative to replay storage. Keep both a percentage and a fixed allowance.
    return math.ceil(tensor_bytes * 1.05) + 64 * 1024 * 1024


def _is_no_space_error(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, OSError) and current.errno in {
            errno.ENOSPC,
            errno.EDQUOT,
        }:
            return True
        message = str(current).lower()
        if (
            "no space left on device" in message
            or "disk quota exceeded" in message
            or "iostream error" in message
            or "unexpected pos" in message
        ):
            return True
        current = current.__cause__
    return False


def _checkpoint_sha256(path: Path) -> str:
    """复用 checkpoint 写入时生成的摘要，缺失时才重新扫描大型文件。"""

    checksum = path.with_suffix(path.suffix + ".sha256")
    if checksum.is_file():
        digest = checksum.read_text(encoding="ascii").strip()
        if len(digest) == 64:
            return digest
    return _sha256(path)


def _fsync_file(path: Path) -> None:
    """将已写临时文件提交到存储设备，再执行原子 rename。"""

    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """提交目录项变更；不支持目录 fsync 的平台保留原子 rename 语义。"""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """校验 sidecar SHA-256 后加载 checkpoint；损坏或缺摘要时拒绝。"""

    source = Path(path)
    checksum = source.with_suffix(source.suffix + ".sha256")
    if not checksum.is_file():
        raise ValueError(f"checkpoint checksum is missing: {checksum}")
    expected = checksum.read_text(encoding="ascii").strip()
    actual = _sha256(source)
    if not expected or actual != expected:
        raise ValueError(f"checkpoint checksum mismatch: {source}")
    value = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return value


def tensor_state_sha256(value: Any) -> str:
    """对嵌套张量状态计算稳定摘要，不依赖 torch.save 容器元数据。"""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, TensorDictBase):
            digest.update(b"tensordict\0")
            update(item.to_dict())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda entry: str(entry)):
                digest.update(str(key).encode("utf-8") + b"\0")
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                update(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr(item).encode("utf-8") + b"\0")
        else:
            raise TypeError(f"unsupported state digest value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _torchrl_version() -> str:
    import torchrl

    return torchrl.__version__
