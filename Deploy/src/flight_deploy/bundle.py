from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from .adapters import AdapterRegistry, default_registry
from .checkpoint import load_checkpoint
from .errors import BundleIntegrityError
from .hashing import sha256_file
from .model import ConvertedPolicy


BUNDLE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


class BundleBuilder:
    def __init__(self, registry: AdapterRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def inspect(
        self,
        checkpoint_path: str | Path,
        *,
        adapter_name: str | None = None,
        verify_checksum: bool = True,
    ) -> Mapping[str, Any]:
        checkpoint, source = load_checkpoint(
            checkpoint_path, verify_checksum=verify_checksum
        )
        adapter = self.registry.select(checkpoint, requested=adapter_name)
        return {**source, **adapter.describe(checkpoint)}

    def build(
        self,
        checkpoint_path: str | Path,
        output_directory: str | Path,
        *,
        adapter_name: str | None = None,
        verify_checksum: bool = True,
        overwrite: bool = False,
    ) -> "DeploymentBundle":
        source_path = Path(checkpoint_path).expanduser().resolve()
        destination = Path(output_directory).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise FileExistsError(f"bundle destination already exists: {destination}")
        checkpoint, source = load_checkpoint(
            source_path, verify_checksum=verify_checksum
        )
        adapter = self.registry.select(checkpoint, requested=adapter_name)
        converted = adapter.convert(checkpoint, source_path=source_path)
        self._validate_converted(converted)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            artifacts = self._write_artifacts(temporary, converted)
            manifest = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "adapter": adapter.name,
                "source": {
                    "checkpoint_sha256": source["sha256"],
                    "checkpoint_checksum_verified": source[
                        "checksum_verified"
                    ],
                    "checkpoint_size_bytes": source["size_bytes"],
                    **dict(converted.source_metadata),
                },
                "interface": {
                    "input": {
                        "name": "observation",
                        "shape": ["batch", converted.input_dim],
                        "dtype": "float32",
                    },
                    "output": {
                        "name": "policy_action",
                        "shape": ["batch", converted.output_dim],
                        "dtype": "float32",
                    },
                    "stateful": converted.stateful,
                    "state_inputs": list(converted.state_inputs),
                    "state_outputs": list(converted.state_outputs),
                },
                "architecture": dict(converted.architecture),
                "contract": dict(converted.contract),
                "artifacts": artifacts,
            }
            (temporary / MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return DeploymentBundle.load(destination)

    @staticmethod
    def _validate_converted(converted: ConvertedPolicy) -> None:
        if converted.stateful:
            raise NotImplementedError(
                "the core bundle schema supports explicit states, but the "
                "TorchScript backend does not yet serialize stateful policies; "
                "install a stateful backend adapter"
            )
        example = torch.zeros(2, converted.input_dim, dtype=torch.float32)
        converted.module.eval()
        with torch.inference_mode():
            output = converted.module(example)
        if not isinstance(output, torch.Tensor) or output.shape != (
            2,
            converted.output_dim,
        ):
            raise ValueError("converted policy violates its declared interface")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("converted policy emits non-finite values")

    @staticmethod
    def _write_artifacts(
        directory: Path, converted: ConvertedPolicy
    ) -> Mapping[str, Mapping[str, Any]]:
        module = converted.module.eval().cpu()
        example = torch.zeros(1, converted.input_dim, dtype=torch.float32)
        with torch.inference_mode():
            exported = torch.export.export(
                module,
                (torch.zeros(2, converted.input_dim, dtype=torch.float32),),
                dynamic_shapes=(
                    {0: torch.export.Dim("batch", min=1)},
                ),
            )
            torch.export.save(exported, directory / "model.pt2")
            traced = torch.jit.trace(module, example, strict=True)
            frozen = torch.jit.freeze(traced)
            frozen.save(str(directory / "model.ts"))
            torch.save(module.state_dict(), directory / "weights.pt")
            expected = module(example)
            actual = frozen(example)
            generator = torch.Generator().manual_seed(0xF11E)
            golden_observation = torch.cat(
                (
                    torch.zeros(1, converted.input_dim),
                    torch.randn(
                        31,
                        converted.input_dim,
                        generator=generator,
                    ),
                )
            )
            golden_action = module(golden_observation)
            torch.save(
                {
                    "schema_version": 1,
                    "observation": golden_observation,
                    "policy_action": golden_action,
                    "absolute_tolerance": 1e-6,
                    "relative_tolerance": 1e-6,
                },
                directory / "golden_vectors.pt",
            )
        maximum_error = float((expected - actual).abs().max().item())
        if maximum_error > 1e-6:
            raise ValueError(
                f"TorchScript conversion error {maximum_error} exceeds tolerance"
            )
        result: dict[str, Mapping[str, Any]] = {}
        for name, kind in (
            ("model.pt2", "torch_export"),
            ("model.ts", "torchscript"),
            ("weights.pt", "pytorch_state_dict"),
            ("golden_vectors.pt", "golden_vectors"),
        ):
            path = directory / name
            result[name] = {
                "kind": kind,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        result["model.ts"] = {
            **result["model.ts"],
            "conversion_max_abs_error": maximum_error,
        }
        return result


class DeploymentBundle:
    def __init__(self, directory: Path, manifest: Mapping[str, Any]) -> None:
        self.directory = directory
        self.manifest = manifest

    @classmethod
    def load(
        cls, path: str | Path, *, verify_integrity: bool = True
    ) -> "DeploymentBundle":
        directory = Path(path).expanduser().resolve()
        manifest_path = directory / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError(f"invalid bundle manifest: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise BundleIntegrityError("bundle manifest root must be a mapping")
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise BundleIntegrityError(
                f"unsupported bundle schema: {manifest.get('schema_version')}"
            )
        bundle = cls(directory, manifest)
        if verify_integrity:
            bundle.verify()
        return bundle

    def verify(self) -> None:
        artifacts = self.manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise BundleIntegrityError("bundle contains no artifacts")
        for name, metadata in artifacts.items():
            if not isinstance(name, str) or not isinstance(metadata, Mapping):
                raise BundleIntegrityError("invalid artifact manifest entry")
            path = self.directory / name
            if not path.is_file():
                raise BundleIntegrityError(f"bundle artifact is missing: {name}")
            expected = metadata.get("sha256")
            if not isinstance(expected, str) or sha256_file(path) != expected:
                raise BundleIntegrityError(f"artifact checksum mismatch: {name}")
        self._verify_golden_vectors()

    def _verify_golden_vectors(self) -> None:
        golden_path = self.directory / "golden_vectors.pt"
        model_path = self.directory / "model.ts"
        if not golden_path.is_file() or not model_path.is_file():
            return
        try:
            golden = torch.load(
                golden_path, map_location="cpu", weights_only=True
            )
            observation = golden["observation"]
            expected = golden["policy_action"]
            module = torch.jit.load(str(model_path), map_location="cpu").eval()
            with torch.inference_mode():
                actual = module(observation)
            torch.testing.assert_close(
                actual,
                expected,
                atol=float(golden["absolute_tolerance"]),
                rtol=float(golden["relative_tolerance"]),
            )
        except Exception as exc:
            raise BundleIntegrityError(
                f"bundle golden-vector verification failed: {exc}"
            ) from exc
