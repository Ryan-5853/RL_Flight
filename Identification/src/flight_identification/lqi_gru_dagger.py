from __future__ import annotations

import argparse
from dataclasses import replace
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from .config import load_experiment_config
from .experiment import (
    _apply_effectiveness_labels,
    _expand_mapping,
    _sample_initial_state,
    _set_initial_observation_state,
    _yaw_free_attitude,
)
from .lqi_gru_closed_loop import evaluate_closed_loop
from .lqi_gru_distillation import StudentStepState, load_student, step_student, train
from .lqi_gru_experiment import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    _action_names,
    _oracle_gains,
    _raw_config,
    _node,
)


def _base_manifest(dataset: Path) -> Mapping[str, Any]:
    manifest_path = dataset / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def prepare_dagger_dataset(
    base_dataset: Path,
    output_dataset: Path,
    observation_names: Sequence[str],
    overwrite: bool = False,
) -> Mapping[str, Any]:
    """Create a working DAgger dataset from the behavior-cloning shards.

    The parameter-group payload and split semantics are copied unchanged so
    train/validation/test isolation is identical to the base dataset. Shards are
    copied with observations restricted to the requested student schema so DAgger
    shards appended later can share one observation contract.
    """

    manifest = _base_manifest(base_dataset)
    if output_dataset.exists() and any(output_dataset.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output dataset is not empty (use --overwrite to rebuild): {output_dataset}"
            )
        shutil.rmtree(output_dataset)
    output_dataset.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        base_dataset / "parameter_groups_audit_only.pt",
        output_dataset / "parameter_groups_audit_only.pt",
    )
    requested = tuple(observation_names)
    full_to_index = {name: index for index, name in enumerate(OBSERVATION_NAMES)}
    missing = [name for name in requested if name not in full_to_index]
    if missing:
        raise ValueError(f"unknown observation names requested: {missing}")
    indices = torch.tensor([full_to_index[name] for name in requested], dtype=torch.int64)
    for split in ("train", "validation", "test"):
        source = base_dataset / split
        if not source.exists():
            continue
        target = output_dataset / split
        target.mkdir(parents=True, exist_ok=True)
        for shard_path in sorted(source.glob("shard_*.pt")):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            if tuple(shard["observation_names"]) != OBSERVATION_NAMES:
                raise ValueError("base shards must use the canonical 25-D schema")
            shard["observations"] = shard["observations"].index_select(2, indices)
            shard["observation_names"] = requested
            torch.save(shard, target / shard_path.name)
    updated = dict(manifest)
    updated["observation_names"] = list(requested)
    updated["dagger_schema_version"] = 1
    updated["base_dataset"] = str(base_dataset)
    (output_dataset / "manifest.json").write_text(
        json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8"
    )
    return updated


def _next_shard_index(dataset: Path, split: str) -> int:
    existing = [int(path.stem.split("_")[1]) for path in (dataset / split).glob("shard_*.pt")]
    return max(existing) + 1 if existing else 0


def _shard_iterations(dataset: Path, split: str) -> set[int]:
    values: set[int] = set()
    for path in (dataset / split).glob("shard_*.pt"):
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if "dagger_iteration" in shard:
            values.add(int(shard["dagger_iteration"]))
    return values


def extend_parameter_payload(
    dataset: Path,
    extra_labels: torch.Tensor,
    extra_split: int = 0,
) -> Mapping[str, Any]:
    """Append new parameter groups (default train split) to a dataset payload."""

    payload_path = dataset / "parameter_groups_audit_only.pt"
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    existing = payload["labels_audit_only"].shape[0]
    if extra_labels.ndim != 2 or extra_labels.shape[1] != payload["labels_audit_only"].shape[1]:
        raise ValueError("extra labels must match the payload label width")
    new_ids = torch.arange(existing, existing + extra_labels.shape[0], dtype=torch.int64)
    payload["labels_audit_only"] = torch.cat(
        (payload["labels_audit_only"], extra_labels.to(payload["labels_audit_only"].dtype)), 0
    )
    payload["split_assignment"] = torch.cat(
        (
            payload["split_assignment"],
            torch.full(
                (extra_labels.shape[0],),
                extra_split,
                dtype=payload["split_assignment"].dtype,
            ),
        ),
        0,
    )
    torch.save(payload, payload_path)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extended_parameter_groups"] = {
        "new_group_ids": new_ids.tolist(),
        "split": ["train", "validation", "test"][extra_split],
        "extra_count": int(extra_labels.shape[0]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"new_group_ids": new_ids.tolist()}


def _save_dagger_shard(
    output: Path,
    shard_index: int,
    dagger_iteration: int,
    split_assignment: torch.Tensor,
    group_ids: torch.Tensor,
    observations: torch.Tensor,
    teacher_actions: torch.Tensor,
    valid_mask: torch.Tensor,
    labels: torch.Tensor,
    lqi_gains: torch.Tensor,
    label_names: tuple[str, ...],
    observation_names: Sequence[str],
    action_names: Sequence[str] = ACTION_NAMES,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_index, split_name in enumerate(("train", "validation")):
        selected = split_assignment[group_ids] == split_index
        counts[split_name] = int(selected.sum())
        if not bool(selected.any()):
            continue
        directory = output / split_name
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "observations": observations[selected].contiguous(),
                "teacher_actions": teacher_actions[selected].contiguous(),
                "valid_mask": valid_mask[selected].contiguous(),
                "parameter_labels_audit_only": labels[selected].contiguous(),
                "oracle_lqi_gain_audit_only": lqi_gains[selected].contiguous(),
                "group_id": group_ids[selected].contiguous(),
                "observation_names": tuple(observation_names),
                "action_names": tuple(action_names),
                "label_names": label_names,
                "leakage_contract": "audit tensors are never student inputs",
                "dagger_iteration": dagger_iteration,
            },
            directory / f"shard_{shard_index:06d}.pt",
        )
    return counts


@torch.no_grad()
def generate_dagger_shards(
    config_path: Path,
    base_dataset: Path,
    checkpoint_path: Path,
    output_dataset: Path,
    train_groups: int,
    validation_groups: int,
    trials: int,
    device: torch.device,
    seed: int,
    shard_index: int | None = None,
    dagger_iteration: int | None = None,
    pilot_height_mode: str = "training_truth",
    selected_group_ids: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    """One DAgger pass: student closed-loop rollouts labeled by the oracle LQI.

    The student controls the environment with its own action feedback, exactly
    like deployment. Every visited state is labeled with the per-airframe
    truth-parameter LQI command evaluated at that same state. Partial (unsafe)
    prefixes are retained because they are the highest-value DAgger data.
    """

    from flight_controller import (
        ControllerContext,
        ControllerReference,
        ControllerState,
        compose_external_upper_command,
        create_controller,
        load_controller_config,
    )
    from flight_train.config import _virtual_pilot
    from simenv import SimulationEnvironment
    from simenv.config import load_and_materialize

    if pilot_height_mode not in {"training_truth", "fixed_target"}:
        raise ValueError("pilot_height_mode must be training_truth or fixed_target")
    config = load_experiment_config(config_path)
    raw = _raw_config(config_path)
    if config.parameterization != "sim2real_micro":
        raise ValueError("DAgger v1 requires sim2real_micro parameterization")
    manifest = _base_manifest(base_dataset)
    payload = torch.load(
        base_dataset / "parameter_groups_audit_only.pt",
        map_location="cpu",
        weights_only=False,
    )
    split_assignment = payload["split_assignment"]
    labels_all = payload["labels_audit_only"]
    label_names = tuple(payload["label_names"])
    train_ids = torch.nonzero(split_assignment == 0).flatten()
    validation_ids = torch.nonzero(split_assignment == 1).flatten()
    if shard_index is None:
        shard_index = _next_shard_index(output_dataset, "train")
    if dagger_iteration is None:
        dagger_iteration = shard_index
    if selected_group_ids is not None:
        selected_groups = torch.as_tensor(
            list(selected_group_ids), dtype=torch.int64
        )
        if len(selected_groups) == 0:
            raise ValueError("selected_group_ids must not be empty")
        if int(selected_groups.max()) >= labels_all.shape[0]:
            raise ValueError("selected_group_ids exceeds payload group count")
        selected_train = selected_groups
    else:
        if train_groups:
            start = (shard_index * train_groups) % max(len(train_ids), 1)
            selected_train = torch.cat(
                (
                    train_ids[start:],
                    train_ids[: max(start + train_groups - len(train_ids), 0)],
                )
            )[:train_groups]
        else:
            selected_train = torch.empty(0, dtype=torch.int64)
        selected_validation = validation_ids[:validation_groups]
        selected_groups = torch.cat((selected_train, selected_validation))
    if len(selected_groups) == 0:
        raise ValueError("no groups selected for DAgger rollout")
    group_count = len(selected_groups)
    batch_size = group_count * trials
    group_ids = selected_groups.repeat_interleave(trials)
    episode_labels = labels_all[selected_groups].repeat_interleave(trials, dim=0)

    device = torch.device(config.device)
    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    nominal = load_and_materialize(config.simulator_config, 1, device, dtype)
    controller_config = dict(load_controller_config(config.controller_config))
    collective_mode = str(
        _node(controller_config, "params").get("collective_mode", "hover")
    )
    if collective_mode not in {"manual", "external_upper"}:
        raise ValueError("DAgger requires manual or external_upper collective mode")
    external_upper = collective_mode == "external_upper"
    action_names = _action_names(controller_config)
    nominal_parameters = _expand_mapping(nominal.parameters, batch_size)
    actual_parameters = _apply_effectiveness_labels(
        nominal_parameters, episode_labels.to(device), config.parameterization
    )
    group_parameters = {
        name: value[::trials] for name, value in actual_parameters.items()
    }
    group_gains = _oracle_gains(
        controller_config, group_parameters, 1.0 / config.control_hz
    )
    episode_gains = group_gains.repeat_interleave(trials, dim=0)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sample_config = replace(config, initial_conditions_per_group=trials)
    attitude, angular_rate = _sample_initial_state(
        batch_size, sample_config, generator, dtype
    )
    initial_state = _expand_mapping(nominal.initial_state, batch_size)
    initial_state["attitude_q_wb"].copy_(attitude.to(device))
    initial_state["angular_velocity_b"].copy_(angular_rate.to(device))
    materialized = replace(
        nominal,
        parameters=actual_parameters,
        initial_state=initial_state,
        sensor_state=_expand_mapping(nominal.sensor_state, batch_size),
    )
    environment = SimulationEnvironment(
        materialized, batch_size, device, dtype, logging_enabled=False
    )
    try:
        oracle_context = ControllerContext(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            control_dt=1.0 / config.control_hz,
            parameters=actual_parameters,
        )
        if external_upper:
            from .external_collective_lqi import ExternalCollectiveLQIController

            oracle = ExternalCollectiveLQIController(
                oracle_context,
                config.controller_config,
                config.simulator_config,
                episode_gains,
                "nonlinear",
            )
        else:
            oracle = create_controller(controller_config, oracle_context)
            oracle.schedule_lqr_gain(episode_gains)
        _set_initial_observation_state(
            environment, oracle.trim.motor_speed, initial_state["angular_velocity_b"]
        )

        student, checkpoint = load_student(checkpoint_path, device)
        student.eval()
        if tuple(checkpoint["action_names"]) != tuple(action_names):
            raise ValueError(
                "student action schema does not match the controller ownership contract"
            )
        mean = checkpoint["normalization"]["mean"].to(device)
        std = checkpoint["normalization"]["std"].to(device)
        full_name_to_index = {
            name: index for index, name in enumerate(OBSERVATION_NAMES)
        }
        checkpoint_names = tuple(checkpoint["observation_names"])
        missing = [
            name for name in checkpoint_names if name not in full_name_to_index
        ]
        if missing:
            raise ValueError(f"checkpoint requests unknown observations: {missing}")
        student_observation_indices = torch.tensor(
            [full_name_to_index[name] for name in checkpoint_names],
            device=device,
            dtype=torch.int64,
        )
        observation_width = len(checkpoint_names)
        initial_previous = torch.as_tensor(
            manifest["initial_previous_command"], device=device, dtype=dtype
        ).reshape(1, 5).expand(batch_size, -1).clone()

        pilot_config = _virtual_pilot(_node(raw, "command_source"))
        from .external_pilot import make_external_pilot

        pilot = make_external_pilot(
            pilot_config, raw, batch_size, device, dtype, config.control_hz
        )
        pilot.generator.manual_seed(seed + 1)
        active = torch.ones(batch_size, device=device, dtype=torch.bool)
        pilot.reset(active)

        previous_command = initial_previous
        student_state = StudentStepState()
        stored_steps = config.episode_steps
        observations = torch.zeros(
            batch_size, stored_steps, observation_width, device=device, dtype=dtype
        )
        teacher_actions = torch.zeros(
            batch_size, stored_steps, len(action_names), device=device, dtype=dtype
        )
        valid_mask = torch.zeros(batch_size, stored_steps, device=device, dtype=torch.bool)
        survival_steps = torch.zeros(batch_size, device=device, dtype=torch.int64)
        for step in range(config.episode_steps):
            truth = environment.observe("truth").values
            sensors = environment.observe("sensor").values
            if pilot_height_mode == "training_truth":
                pilot_height = -truth["position_n"][:, 2:3]
            else:
                pilot_height = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            pilot.step(pilot_height, active)
            command_snapshot = pilot.snapshot()
            attitude_estimate = _yaw_free_attitude(truth["attitude_q_wb"])
            target_attitude = _yaw_free_attitude(command_snapshot.target_attitude_q_wb)
            target_rate = torch.cat(
                (
                    torch.zeros(batch_size, 2, device=device, dtype=dtype),
                    pilot.desired_yaw_rate,
                ),
                dim=1,
            )
            reference = ControllerReference(
                target_position_n=torch.zeros(batch_size, 3, device=device, dtype=dtype),
                target_velocity_n=torch.zeros(batch_size, 3, device=device, dtype=dtype),
                target_attitude_q_wb=target_attitude,
                target_angular_velocity_b=target_rate,
                collective_command=command_snapshot.upper_throttle,
            )
            state = ControllerState(
                position_n=truth["position_n"],
                velocity_n=truth["velocity_n"],
                attitude_q_wb=attitude_estimate,
                angular_velocity_b=truth["angular_velocity_b"],
                linear_acceleration_n=truth["linear_acceleration_n"],
                motor_speed=truth["motor_speed"],
                servo_angle=truth["servo_angle"],
            )
            oracle_action = (
                oracle.step(
                    state,
                    reference,
                    command_snapshot.upper_throttle,
                    active,
                    update_observer=False,
                )
                if external_upper
                else oracle.step(state, reference, active).command
            )
            student_observation = torch.cat(
                (
                    attitude_estimate,
                    sensors["gyro"],
                    sensors["accelerometer"],
                    sensors["motor_speed"],
                    target_attitude,
                    target_rate,
                    command_snapshot.upper_throttle,
                    previous_command,
                ),
                dim=1,
            ).index_select(1, student_observation_indices)
            normalized = (student_observation - mean) / std
            student_command, student_state = step_student(
                student,
                checkpoint,
                normalized,
                student_state,
                full_context=True,
            )
            observations[:, step].copy_(
                torch.where(
                    active[:, None],
                    student_observation,
                    torch.zeros_like(student_observation),
                )
            )
            teacher_actions[:, step].copy_(oracle_action)
            valid_mask[:, step].copy_(active)
            applied_command = (
                compose_external_upper_command(
                    command_snapshot.upper_throttle, student_command
                )
                if external_upper
                else student_command
            )
            if external_upper:
                oracle.observe_applied_command(
                    command_snapshot.upper_throttle, student_command
                )
            previous_command = applied_command
            result = environment.advance(applied_command, active)
            next_truth = environment.observe("truth").values
            tilt = torch.acos(
                (2.0 * truth["attitude_q_wb"][:, 0].square()
                 + 2.0 * truth["attitude_q_wb"][:, 3].square() - 1.0).clamp(-1.0, 1.0)
            )
            active &= result.valid
            active &= torch.isfinite(next_truth["attitude_q_wb"]).all(1)
            active &= torch.isfinite(tilt)
            active &= tilt <= config.convergence.safety_tilt_rad
            active &= truth["angular_velocity_b"].norm(dim=1) <= config.convergence.safety_angular_rate_rad_s
            survival_steps += active.to(torch.int64)

        counts = _save_dagger_shard(
            output_dataset,
            shard_index,
            int(dagger_iteration),
            split_assignment,
            group_ids,
            observations.cpu(),
            teacher_actions.cpu(),
            valid_mask.cpu(),
            episode_labels,
            episode_gains.cpu(),
            label_names,
            checkpoint_names,
            action_names,
        )
        totals = {
            "iteration": shard_index,
            "groups": group_count,
            "episodes": batch_size,
            "valid_steps": int(valid_mask.sum()),
            "survival_fraction_mean": float(survival_steps.to(dtype).mean() / config.episode_steps),
            "survival_fraction_p10": float(
                torch.quantile(
                    survival_steps.to(dtype) / config.episode_steps, 0.10
                )
            ),
            "safe_fraction": float(active.to(torch.float32).mean()),
            "shards": counts,
            "checkpoint": str(checkpoint_path),
        }
        return totals
    finally:
        environment.close()


def run_dagger_iterations(
    base_dataset: Path,
    config_path: Path,
    output_root: Path,
    device: torch.device,
    iterations: int,
    train_groups: int,
    validation_groups: int,
    trials: int,
    epochs: int,
    seed: int,
    exclude_previous_command: bool,
    arch: str,
    context_steps: int | None,
    previous_command_noise_std: float,
    previous_command_reset_prob: float,
    gate_trials: int,
    gate_duration_s: float,
    gate_groups: int,
    start_iteration: int = 0,
    initial_checkpoint: Path | None = None,
    update_epochs: int | None = None,
) -> Mapping[str, Any]:
    """Train -> DAgger-rollout -> closed-loop gate loop."""

    full_names = tuple(OBSERVATION_NAMES)
    if exclude_previous_command:
        observation_names = tuple(
            name for name in full_names if not name.startswith("previous_command.")
        )
    else:
        observation_names = full_names
    merged = output_root / "datasets" / "merged"
    if merged.exists() and any(merged.iterdir()):
        manifest = _base_manifest(merged)
        if tuple(manifest["observation_names"]) != observation_names:
            raise ValueError("merged dataset observation schema differs from requested")
    else:
        prepare_dagger_dataset(base_dataset, merged, observation_names)
    results: list[Mapping[str, Any]] = []
    previous_checkpoint = initial_checkpoint
    if previous_checkpoint is None and start_iteration > 0:
        candidate = output_root / "students" / f"iter_{start_iteration - 1:02d}" / "student.pt"
        if candidate.exists():
            previous_checkpoint = candidate
    for iteration in range(start_iteration, iterations):
        student_dir = output_root / "students" / f"iter_{iteration:02d}"
        if student_dir.exists() and any(student_dir.iterdir()):
            checkpoint = student_dir / "student.pt"
            train_result = {
                "checkpoint": str(checkpoint),
                "resumed_existing": True,
            }
        else:
            iteration_epochs = (
                epochs
                if iteration == 0 and previous_checkpoint is None
                else (update_epochs or epochs)
            )
            train_result = train(
                merged,
                config_path,
                student_dir,
                device,
                epochs_override=iteration_epochs,
                resume_checkpoint=previous_checkpoint,
                exclude_previous_command=exclude_previous_command,
                arch=arch,
                context_steps=context_steps,
                previous_command_noise_std=previous_command_noise_std,
                previous_command_reset_prob=previous_command_reset_prob,
            )
            checkpoint = Path(train_result["checkpoint"])
        previous_checkpoint = checkpoint
        if iteration in _shard_iterations(merged, "train"):
            rollout = {"skipped_existing": True, "iteration": iteration}
        else:
            rollout = generate_dagger_shards(
                config_path,
                base_dataset,
                checkpoint,
                merged,
                train_groups,
                validation_groups,
                trials,
                device,
                seed + 1000 * (iteration + 1),
                dagger_iteration=iteration,
            )
        gate_output = output_root / "gates" / f"gate_iter_{iteration:02d}.json"
        gate = evaluate_closed_loop(
            merged,
            config_path,
            checkpoint,
            gate_output,
            device,
            maximum_groups=gate_groups,
            trials=gate_trials,
            duration_s=gate_duration_s,
            seed=seed + 7 + iteration,
            pilot_height_mode="fixed_target",
            split="validation",
            variants=("oracle_lqi", "gru", "gru_reset"),
        )
        entry = {
            "iteration": iteration,
            "train": train_result,
            "rollout": rollout,
            "gate": {
                "gru_safe_fraction": gate["variants"]["gru"]["safe_fraction"],
                "gru_reset_safe_fraction": gate["variants"]["gru_reset"]["safe_fraction"],
                "oracle_safe_fraction": gate["variants"]["oracle_lqi"]["safe_fraction"],
                "gru_rate_rms_mean": gate["variants"]["gru"]["rate_tracking_rms_rad_s"]["mean"],
                "gru_attitude_rms_mean": gate["variants"]["gru"]["attitude_tracking_rms_rad"]["mean"],
            },
            "gate_report": str(gate_output),
            "checkpoint": str(checkpoint),
            "parent_checkpoint": (
                None
                if train_result.get("parent_checkpoint") is None
                else train_result.get("parent_checkpoint")
            ),
        }
        results.append(entry)
        summary_path = output_root / "dagger_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "base_dataset": str(base_dataset),
                    "config": str(config_path),
                    "arch": arch,
                    "context_steps": context_steps,
                    "exclude_previous_command": exclude_previous_command,
                    "previous_command_noise_std": previous_command_noise_std,
                    "previous_command_reset_prob": previous_command_reset_prob,
                    "iterations": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(json.dumps(entry, indent=2, sort_keys=True))
    # The last rollout is newly aggregated data. Fit one final student after
    # the loop so the deployable checkpoint has actually seen every DAgger
    # shard (iteration checkpoints intentionally precede their own rollout).
    final_dir = output_root / "students" / "final"
    if final_dir.exists() and any(final_dir.iterdir()):
        final_checkpoint = final_dir / "student.pt"
        final_train = {
            "checkpoint": str(final_checkpoint),
            "resumed_existing": True,
        }
    else:
        final_train = train(
            merged,
            config_path,
            final_dir,
            device,
            epochs_override=update_epochs or epochs,
            resume_checkpoint=previous_checkpoint,
            exclude_previous_command=exclude_previous_command,
            arch=arch,
            context_steps=context_steps,
            previous_command_noise_std=previous_command_noise_std,
            previous_command_reset_prob=previous_command_reset_prob,
        )
        final_checkpoint = Path(final_train["checkpoint"])
    final_gate_path = output_root / "gates" / "gate_final.json"
    final_gate = evaluate_closed_loop(
        merged,
        config_path,
        final_checkpoint,
        final_gate_path,
        device,
        maximum_groups=gate_groups,
        trials=gate_trials,
        duration_s=gate_duration_s,
        seed=seed + 100_007,
        pilot_height_mode="fixed_target",
        split="validation",
        variants=("oracle_lqi", "nominal_lqi", "gru", "gru_reset"),
    )
    summary = {
        "arch": arch,
        "iterations": results,
        "final_train": final_train,
        "final_checkpoint": str(final_checkpoint),
        "final_gate": str(final_gate_path),
        "final_gate_summary": final_gate["variants"],
    }
    (output_root / "dagger_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger for oracle-LQI angular-velocity students")
    parser.add_argument("--stage", choices=("prepare", "rollout", "iterate"), required=True)
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--output-dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-groups", type=int, default=96)
    parser.add_argument("--validation-groups", type=int, default=0)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--dagger-iteration", type=int)
    parser.add_argument("--groups", type=int, nargs="+")
    parser.add_argument("--exclude-previous-command", action="store_true")
    parser.add_argument(
        "--arch",
        choices=("gru", "lstm", "tcn", "transformer"),
        default="gru",
    )
    parser.add_argument("--context-steps", type=int)
    parser.add_argument("--previous-command-noise-std", type=float, default=0.0)
    parser.add_argument("--previous-command-reset-prob", type=float, default=0.0)
    parser.add_argument("--gate-trials", type=int, default=4)
    parser.add_argument("--gate-duration-s", type=float, default=6.0)
    parser.add_argument("--gate-groups", type=int, default=24)
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument(
        "--update-epochs",
        type=int,
        help="epochs for warm-started DAgger iterations and the final fit",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dataset).resolve()
    config_path = Path(args.config).resolve()
    device = torch.device(args.device)
    if args.stage == "prepare":
        if args.output_dataset is None:
            parser.error("--output-dataset is required for prepare")
        full_names = tuple(OBSERVATION_NAMES)
        names = (
            tuple(n for n in full_names if not n.startswith("previous_command."))
            if args.exclude_previous_command
            else full_names
        )
        manifest = prepare_dagger_dataset(
            base, Path(args.output_dataset).resolve(), names, overwrite=args.overwrite
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if args.stage == "rollout":
        if args.checkpoint is None or args.output_dataset is None:
            parser.error("--checkpoint and --output-dataset are required for rollout")
        totals = generate_dagger_shards(
            config_path,
            base,
            Path(args.checkpoint).resolve(),
            Path(args.output_dataset).resolve(),
            args.train_groups,
            args.validation_groups,
            args.trials,
            device,
            args.seed,
            args.shard_index,
            args.dagger_iteration,
            selected_group_ids=args.groups,
        )
        print(json.dumps(totals, indent=2, sort_keys=True))
        return
    if args.output_root is None:
        parser.error("--output-root is required for iterate")
    summary = run_dagger_iterations(
        base,
        config_path,
        Path(args.output_root).resolve(),
        device,
        args.iterations,
        args.train_groups,
        args.validation_groups,
        args.trials,
        args.epochs,
        args.seed,
        args.exclude_previous_command,
        args.arch,
        args.context_steps,
        args.previous_command_noise_std,
        args.previous_command_reset_prob,
        args.gate_trials,
        args.gate_duration_s,
        args.gate_groups,
        args.start_iteration,
        None
        if args.initial_checkpoint is None
        else Path(args.initial_checkpoint).resolve(),
        args.update_epochs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
