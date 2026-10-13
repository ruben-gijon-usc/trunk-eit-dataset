import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EXPERIMENTS_CSV = PROJECT_ROOT / "experiments_results.csv"

FIELDNAMES = [
    "timestamp",
    "git_commit",
    "script",
    "model_name",
    "loss_type",
    "train_dataset",
    "test_id_dataset",
    "test_ood_dataset",
    # Hold-out Validation Split Metrics (80/20)
    "val_loss",
    "val_rmse",
    "val_ssim",
    "val_dice",
    "val_iou",
    "val_pe",
    # Test ID Metrics (Sample-level Mean & Std across Test ID)
    "test_id_rmse_mean",
    "test_id_rmse_std",
    "test_id_ssim_mean",
    "test_id_ssim_std",
    "test_id_dice_mean",
    "test_id_dice_std",
    "test_id_iou_mean",
    "test_id_iou_std",
    "test_id_pe_mean",
    "test_id_pe_std",
    # Test OOD Metrics (Sample-level Mean & Std across Test OOD)
    "test_ood_rmse_mean",
    "test_ood_rmse_std",
    "test_ood_ssim_mean",
    "test_ood_ssim_std",
    "test_ood_dice_mean",
    "test_ood_dice_std",
    "test_ood_iou_mean",
    "test_ood_iou_std",
    "test_ood_pe_mean",
    "test_ood_pe_std",
    "hyperparameters",
]


def get_git_info() -> str:
    """Obtains the current git commit hash and checks for uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{commit} (dirty)" if dirty else commit
    except Exception:
        return "unknown"


def log_experiment_result(
    script: str,
    model_name: str,
    loss_type: str,
    train_dataset: str,
    test_id_dataset: str | None,
    test_ood_dataset: str | None,
    val_metrics: dict[str, Any] | None = None,
    test_id_metrics: dict[str, tuple[float, float]] | None = None,
    test_ood_metrics: dict[str, tuple[float, float]] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    csv_path: Path | str = DEFAULT_EXPERIMENTS_CSV,
) -> Path:
    """Appends an experiment run (with validation metrics and sample-level test statistics) to experiments_results.csv."""
    csv_file = Path(csv_path)

    row: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_info(),
        "script": script,
        "model_name": model_name,
        "loss_type": loss_type,
        "train_dataset": str(train_dataset),
        "test_id_dataset": str(test_id_dataset) if test_id_dataset else "",
        "test_ood_dataset": str(test_ood_dataset) if test_ood_dataset else "",
        "hyperparameters": json.dumps(hyperparameters) if hyperparameters else "",
    }

    # Validation Metrics (Hold-out Val Set)
    val_metrics = val_metrics or {}
    for m in ["loss", "rmse", "ssim", "dice", "iou", "pe"]:
        val_key = f"val_{m}"
        raw_val = val_metrics.get(m, val_metrics.get(val_key, None))
        if raw_val is not None:
            if isinstance(raw_val, (tuple, list)):
                raw_val = raw_val[0]
            row[val_key] = f"{float(raw_val):.6f}"
        else:
            row[val_key] = ""

    # Test ID Metrics (sample-level mean, std)
    test_id_metrics = test_id_metrics or {}
    for m in ["rmse", "ssim", "dice", "iou", "pe"]:
        if m in test_id_metrics:
            mean_val, std_val = test_id_metrics[m]
            row[f"test_id_{m}_mean"] = f"{mean_val:.6f}"
            row[f"test_id_{m}_std"] = f"{std_val:.6f}"
        else:
            row[f"test_id_{m}_mean"] = ""
            row[f"test_id_{m}_std"] = ""

    # Test OOD Metrics (sample-level mean, std)
    test_ood_metrics = test_ood_metrics or {}
    for m in ["rmse", "ssim", "dice", "iou", "pe"]:
        if m in test_ood_metrics:
            mean_val, std_val = test_ood_metrics[m]
            row[f"test_ood_{m}_mean"] = f"{mean_val:.6f}"
            row[f"test_ood_{m}_std"] = f"{std_val:.6f}"
        else:
            row[f"test_ood_{m}_mean"] = ""
            row[f"test_ood_{m}_std"] = ""

    file_exists = csv_file.exists() and csv_file.stat().st_size > 0

    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n📊 Experimento registrado exitosamente en: {csv_file.resolve()}")
    return csv_file
