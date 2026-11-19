from __future__ import annotations

import hashlib
import json
from pathlib import Path

from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess

import torch


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_info() -> dict:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"commit": None, "dirty": None}

    if rev.returncode != 0 or status.returncode != 0:
        return {"commit": None, "dirty": None}

    commit = rev.stdout.strip() or None
    dirty = bool(status.stdout.strip())
    return {"commit": commit, "dirty": dirty}


def training_data_identity(
    train_manifest: Path = Path("data/processed/manifests/train.csv"),
    validation_manifest: Path = Path("data/processed/manifests/validation.csv"),
) -> str:
    """Return a deterministic identity for the training + validation manifests.

    This helper is provided because the final evaluator uses the same identity
    to verify that the selected MLflow Run matches the current prepared data.
    Do not hard-code the value.
    """
    paths = (train_manifest, validation_manifest)
    if any(not path.exists() for path in paths):
        return "unavailable"

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_run_manifest(params: dict, device: str, mlflow_run_id: str) -> dict:
    """Build experiment provenance metadata."""
    data_cfg = params.get("data", {})
    train_cfg = params.get("train", {})
    aug_cfg = params.get("augmentation", {})

    return {
        "git": _git_info(),
        "data": {
            "training_data_identity": training_data_identity(),
            "image_size": int(data_cfg["image_size"]),
        },
        "configuration": {
            "image_size": int(data_cfg["image_size"]),
            "batch_size": int(train_cfg["batch_size"]),
            "epochs": int(train_cfg["epochs"]),
            "learning_rate": float(train_cfg["learning_rate"]),
            "dropout": float(train_cfg["dropout"]),
            "augmentation": {
                "enabled": bool(aug_cfg.get("enabled", False)),
                "horizontal_flip": float(aug_cfg.get("horizontal_flip", False)),
                "rotation_degrees": float(aug_cfg.get("rotation_degrees", 0.0)),
            },
        },
        "execution": {
            "mlflow_run_id": str(mlflow_run_id),
            "random_state": int(train_cfg["random_state"]),
            "device": str(device),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "dvc": _package_version("dvc"),
            "mlflow": _package_version("mlflow"),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
        },
    }


def write_run_manifest(params: dict, device: str, mlflow_run_id: str, path: Path) -> dict:
    manifest = build_run_manifest(params, device, mlflow_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
