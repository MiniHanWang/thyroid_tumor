from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.thyroid_dataset import (
    ThyroidPatientDataset,
    build_transforms,
    collate_patients,
    ensure_dir,
    get_path,
    inspect_dataset,
    load_clinical_df,
    load_config,
    project_root_from,
    save_dataset_summary,
    set_seed,
)
from evaluation.metrics import metrics_from_probs
from evaluation.roc_plot import save_multi_roc
from models.cnn_backbones import BACKBONES
from training.progress import format_duration, log, timed_stage
from training.train_backbone import evaluate_backbone_model, load_backbone_checkpoint, train_backbone_model


def main() -> None:
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Patient-level backbone benchmark for thyroid ultrasound malignancy prediction.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--backbones", nargs="*", default=BACKBONES)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    project_root = project_root_from(__file__)
    cfg = load_config(project_root)
    image_root = get_path(cfg, project_root, "data", "image_root")
    clinical_path = get_path(cfg, project_root, "data", "clinical_file")
    evaluation_dir = ensure_dir(get_path(cfg, project_root, "paths", "evaluation_dir"))
    backbone_model_dir = ensure_dir(get_path(cfg, project_root, "paths", "backbone_model_dir"))

    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg["training"]["lr"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["training"]["num_workers"]

    set_seed(seed)
    log(
        f"[run] backbone benchmark | backbones={len(args.backbones)} epochs={epochs} "
        f"batch_size={batch_size} device={'cpu' if args.cpu_only else 'auto'}"
    )
    with timed_stage("Inspect dataset and load clinical table"):
        summary = inspect_dataset(image_root)
        df = load_clinical_df(clinical_path, image_root)
    summary["matched_patients"] = int(df["patient_id"].nunique())
    summary["matched_images"] = int(df["image_count"].sum())
    save_dataset_summary(summary, evaluation_dir)
    if args.inspect_only:
        return

    if args.max_patients > 0:
        df = df.sort_values("patient_id", key=lambda s: s.astype(int)).head(args.max_patients).copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["malignant"],
    )

    train_ds = ThyroidPatientDataset(train_df, image_root, build_transforms(train=True))
    test_ds = ThyroidPatientDataset(test_df, image_root, build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_patients)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_patients)
    use_gpu = cfg["training"]["device"]["use_gpu"] and not args.cpu_only and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    performance_rows = []
    roc_curves = {}
    for backbone_name in args.backbones:
        backbone_start = time.perf_counter()
        log(f"[backbone-start] {backbone_name}")
        checkpoint_path = backbone_model_dir / f"{backbone_name}_best.pth"
        if checkpoint_path.exists() and not args.force_retrain:
            log(f"[backbone-skip] {backbone_name} | loading existing checkpoint")
            model = load_backbone_checkpoint(backbone_name, str(checkpoint_path), device)
            y_true, y_prob = evaluate_backbone_model(model, test_loader, device)
            metrics = metrics_from_probs(y_true, y_prob)
        else:
            model, y_true, y_prob, metrics = train_backbone_model(
                backbone_name=backbone_name,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            torch.save(model.state_dict(), checkpoint_path)
        performance_rows.append({"Backbone": backbone_name, **metrics})
        roc_curves[backbone_name] = (y_true, y_prob)
        log(
            f"[backbone-done] {backbone_name} | auc={metrics['AUC']:.4f} "
            f"elapsed={format_duration(time.perf_counter() - backbone_start)}"
        )

    performance_df = pd.DataFrame(performance_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    performance_df.to_csv(evaluation_dir / "backbone_performance.csv", index=False)
    save_multi_roc(roc_curves, evaluation_dir / "backbone_roc_comparison.png", "Backbone ROC comparison")

    ranking_df = performance_df[["Backbone", "AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "Recall", "F1"]].copy()
    ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))
    ranking_df.to_csv(evaluation_dir / "backbone_ranking.csv", index=False)

    best = ranking_df.iloc[0]
    best_text = (
        f"Best backbone: {best['Backbone']}\n"
        f"AUC: {best['AUC']:.6f}\n"
        f"Accuracy: {best['Accuracy']:.6f}\n"
        f"Sensitivity: {best['Sensitivity']:.6f}\n"
        f"Specificity: {best['Specificity']:.6f}\n"
    )
    (evaluation_dir / "best_backbone.txt").write_text(best_text, encoding="utf-8")
    log(f"[run-done] backbone benchmark finished in {format_duration(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
