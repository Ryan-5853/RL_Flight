from __future__ import annotations

import hashlib
import json
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
        self._checkpoint_entries: list[dict[str, Any]] = []
        self._best_evaluation_survival_s = float("-inf")
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
            "control_contract": config.control_contract.version,
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
        self._metrics.write(json.dumps(record, ensure_ascii=False) + "\n")
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
                }
                for name, values in report["scenarios"].items()
            },
        }
        path = self.directory / "evaluations.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def promote_best_evaluation_checkpoint(
        self,
        checkpoint: Path,
        *,
        control_steps: int,
        hover_survival_s: float,
        total_score: float,
    ) -> bool:
        """按固定悬停生存时间保留独立 best checkpoint。"""

        if hover_survival_s <= self._best_evaluation_survival_s:
            return False
        destination = self._checkpoint_directory / "best_fixed_evaluation.pt"
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(checkpoint, temporary)
        _fsync_file(temporary)
        temporary.replace(destination)
        digest = _sha256(destination)
        checksum = destination.with_suffix(destination.suffix + ".sha256")
        checksum.write_text(digest + "\n", encoding="ascii")
        _fsync_file(checksum)
        metadata = {
            "global_control_steps": control_steps,
            "hover_survival_s": hover_survival_s,
            "total_score": total_score,
            "source_checkpoint": checkpoint.name,
            "file": destination.name,
            "sha256": digest,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = self._checkpoint_directory / "best_fixed_evaluation.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _fsync_file(metadata_path)
        _fsync_directory(self._checkpoint_directory)
        self._best_evaluation_survival_s = hover_survival_s
        return True

    def checkpoint(
        self, control_steps: int, state: Mapping[str, Any], *, kind: str
    ) -> Path:
        """先写临时文件再原子替换，避免中断留下半个 checkpoint。"""

        if kind not in {"periodic", "final", "interrupt"}:
            raise ValueError(f"unsupported checkpoint kind: {kind}")
        destination = self._checkpoint_directory / f"step_{control_steps}.pt"
        temporary = destination.with_suffix(".tmp")
        torch.save(dict(state), temporary)
        _fsync_file(temporary)
        temporary.replace(destination)
        _fsync_directory(self._checkpoint_directory)
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

    def bind_resume(self, *, checkpoint: Path, parent_run_id: str | None) -> None:
        """记录精确续训来源，新 run 保持独立身份和日志目录。"""

        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "resume_mode": "exact",
                "resume_checkpoint": str(checkpoint),
                "parent_run_id": parent_run_id,
            }
        )
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self._manifest_path)

    def close(self, status: str, control_steps: int) -> None:
        """关闭指标流并写入本次运行的最终状态。"""

        if not self._metrics.closed:
            self._metrics.close()
        (self.directory / "status.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "global_control_steps": control_steps,
                    "ended_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免把大型配置附件一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
