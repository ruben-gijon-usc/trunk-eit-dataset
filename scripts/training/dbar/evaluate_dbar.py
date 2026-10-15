import argparse
import collections
import sys
from pathlib import Path

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY
from src.data.eit_dataset import EITDataset
from src.training.dbar.dbar_solver import DBarSolver
from src.training.experiment_logger import log_experiment_result
from src.training.metrics import EITMetricsEvaluator
from src.training.train_utils import set_seed


def evaluate_dataset(
    solver: DBarSolver, dataset: EITDataset, evaluator: EITMetricsEvaluator
) -> dict[str, tuple[float, float]]:
    metrics_list: dict[str, list[float]] = collections.defaultdict(list)
    n_samples = len(dataset)

    for i in range(n_samples):
        v, target, mask = dataset[i]
        
        # Cargar reconstrucción precalculada si existe (p. ej. generada con --save_dbar)
        idx = getattr(dataset, "valid_indices", [str(i)])[i] if hasattr(dataset, "valid_indices") else str(i)
        recon_path = None
        if hasattr(dataset, "dataset_dir"):
            candidate1 = dataset.dataset_dir / f"reconstructions/dbar_adjacent/{idx}.npy"
            candidate2 = dataset.dataset_dir / f"reconstructions/dbar/{idx}.npy"
            if candidate1.exists():
                recon_path = candidate1
            elif candidate2.exists():
                recon_path = candidate2

        if recon_path is not None:
            pred_arr = np.load(recon_path)
            pred = torch.tensor(pred_arr, dtype=torch.float32)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
        else:
            # Reconstrucción analítica D-bar en tiempo de ejecución
            pred = solver.reconstruct(v, mask=mask)  # (1, 64, 64)

        # Batch dimension for evaluator: (1, 1, 64, 64)
        pred_4d = pred.unsqueeze(0)
        target_4d = target.unsqueeze(0)
        mask_4d = mask.unsqueeze(0)

        metrics_list["rmse"].append(evaluator.rmse(pred_4d, target_4d, mask_4d).item())
        metrics_list["ssim"].append(evaluator.ssim(pred_4d, target_4d, mask_4d).item())
        dice, iou = evaluator.dice_and_iou(pred_4d, target_4d, mask_4d)
        metrics_list["dice"].append(dice.item())
        metrics_list["iou"].append(iou.item())
        metrics_list["pe"].append(evaluator.position_error(pred_4d, target_4d, mask_4d).item())

    summary = {}
    for k, vals in metrics_list.items():
        summary[k] = (float(np.mean(vals)), float(np.std(vals)))
    return summary


def save_qualitative_pdf(solver: DBarSolver, dataset: EITDataset, output_path: Path):
    """
    Saves a lossless vector .pdf visualization for research paper integration (Rule #4).
    Compares Ground Truth, D-Bar Reconstruction, and Error Map.
    """
    if len(dataset) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    v, target, mask = dataset[0]
    idx = getattr(dataset, "valid_indices", ["0"])[0] if hasattr(dataset, "valid_indices") else "0"
    recon_path = None
    if hasattr(dataset, "dataset_dir"):
        candidate1 = dataset.dataset_dir / f"reconstructions/dbar_adjacent/{idx}.npy"
        candidate2 = dataset.dataset_dir / f"reconstructions/dbar/{idx}.npy"
        if candidate1.exists():
            recon_path = candidate1
        elif candidate2.exists():
            recon_path = candidate2

    if recon_path is not None:
        pred = np.load(recon_path)
        if hasattr(pred, "ndim") and pred.ndim == 3:
            pred = pred.squeeze(0)
    else:
        pred = solver.reconstruct(v, mask=mask).squeeze().numpy()

    target_np = target.squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    # Zero background
    pred[mask_np == 0] = np.nan
    target_np[mask_np == 0] = np.nan
    diff = np.abs(pred - target_np)

    im0 = axes[0].imshow(target_np, origin="lower", cmap="viridis", vmin=0, vmax=MAX_CONDUCTIVITY)
    axes[0].set_title("Ground Truth Conductivity")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, origin="lower", cmap="viridis", vmin=0, vmax=MAX_CONDUCTIVITY)
    axes[1].set_title("D-Bar Reconstruction")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, origin="lower", cmap="magma")
    axes[2].set_title("Absolute Error Map")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"📄 Gráfico vectorial guardado para el paper: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación del Método Analítico D-Bar")
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--test_id_dir", type=str, default="dataset/dataset_16e_adjacent_test_id")
    parser.add_argument("--test_ood_dir", type=str, default="dataset/dataset_16e_adjacent_test_ood")
    parser.add_argument(
        "--r_cutoff", type=float, default=3.5, help="Radio de truncamiento en el plano k"
    )
    parser.add_argument("--electrodes", type=int, default=16)
    parser.add_argument(
        "--val_samples", type=int, default=200, help="Muestras de validación para sanity check"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla aleatoria para replicabilidad"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(__file__).resolve().parent

    evaluator = EITMetricsEvaluator(threshold=GROUND_TRUTH_THRESHOLD, data_range=MAX_CONDUCTIVITY)
    solver = DBarSolver(n_electrodes=args.electrodes, r_cutoff=args.r_cutoff, grid_resolution=64)

    print("\n" + "=" * 50)
    print(f"Iniciando Evaluación D-Bar (Direct Analytical Method, R_cutoff={args.r_cutoff})")
    print("=" * 50)

    # 1. Validación sobre subconjunto de train_dir
    val_summary = None
    if (PROJECT_ROOT / args.train_dir).exists():
        train_dataset = EITDataset(
            PROJECT_ROOT / args.train_dir, target_type="grid", flatten_grid=False
        )
        n_eval = min(args.val_samples, len(train_dataset))
        val_sub = Subset(train_dataset, range(n_eval))
        print(f"Evaluando D-Bar en {n_eval} muestras de validación...")
        val_summary = evaluate_dataset(solver, val_sub, evaluator)
        print("\n--- Métricas de Validación ---")
        for k, (m_val, s_val) in val_summary.items():
            print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    # 2. Test ID
    test_id_dataset = None
    test_id_summary = None
    if (PROJECT_ROOT / args.test_id_dir).exists():
        print("\nEvaluando D-Bar en Test ID (Misma Distribución)...")
        test_id_dataset = EITDataset(
            PROJECT_ROOT / args.test_id_dir, target_type="grid", flatten_grid=False
        )
        test_id_summary = evaluate_dataset(solver, test_id_dataset, evaluator)
        print("\n--- Test ID ---")
        for k, (m_val, s_val) in test_id_summary.items():
            print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

        # Guardar gráfico PDF vectorial
        save_qualitative_pdf(
            solver, test_id_dataset, output_dir / "visualizations/dbar_reconstruction_test_id.pdf"
        )

    # 3. Test OOD
    test_ood_dataset = None
    test_ood_summary = None
    if (PROJECT_ROOT / args.test_ood_dir).exists():
        print("\nEvaluando D-Bar en Test OOD (Topologías Complejas)...")
        test_ood_dataset = EITDataset(
            PROJECT_ROOT / args.test_ood_dir, target_type="grid", flatten_grid=False
        )
        test_ood_summary = evaluate_dataset(solver, test_ood_dataset, evaluator)
        print("\n--- Test OOD ---")
        for k, (m_val, s_val) in test_ood_summary.items():
            print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

        # Guardar gráfico PDF vectorial
        save_qualitative_pdf(
            solver, test_ood_dataset, output_dir / "visualizations/dbar_reconstruction_test_ood.pdf"
        )

    # 4. Registrar en experiments_results.csv
    log_experiment_result(
        script="scripts/training/dbar/evaluate_dbar.py",
        model_name="D-Bar (Analytical)",
        loss_type="none",
        train_dataset=args.train_dir,
        test_id_dataset=args.test_id_dir if test_id_summary else None,
        test_ood_dataset=args.test_ood_dir if test_ood_summary else None,
        val_metrics=val_summary,
        test_id_metrics=test_id_summary,
        test_ood_metrics=test_ood_summary,
        hyperparameters={"r_cutoff": args.r_cutoff, "method": "analytical_born_dbar"},
    )


if __name__ == "__main__":
    main()
