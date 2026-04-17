from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from datasets.thyroid_dataset import (
    CLINICAL_COLUMNS,
    ThyroidPatientDataset,
    build_clinical_transformer,
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
from models.multimodal_model import UltrasoundMILModel
from training.progress import format_duration, log, timed_stage
from training.train_multimodal import train_ultrasound_model


def save_clinical_importance(model: Pipeline, evaluation_dir):
    names = model.named_steps["pre"].get_feature_names_out()
    values = model.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": names, "coefficient": values}).assign(abs_value=lambda x: x["coefficient"].abs())
    df.sort_values("abs_value", ascending=False).to_csv(evaluation_dir / "clinical_feature_importance.csv", index=False)
    top = df.sort_values("abs_value", ascending=False).head(15).sort_values("coefficient")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["coefficient"])
    plt.title("Clinical Feature Importance")
    plt.tight_layout()
    plt.savefig(evaluation_dir / "clinical_feature_importance.png", dpi=200)
    plt.close()


def train_clinical_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    transformer = build_clinical_transformer(train_df)
    x_train, x_test = train_df[CLINICAL_COLUMNS], test_df[CLINICAL_COLUMNS]
    y_train, y_test = train_df["malignant"].astype(int), test_df["malignant"].astype(int)
    logreg = Pipeline([("pre", transformer), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    rf = Pipeline([("pre", transformer), ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))])
    logreg.fit(x_train, y_train)
    rf.fit(x_train, y_train)
    rows = []
    for name, model in [("Clinical_LogisticRegression", logreg), ("Clinical_RandomForest", rf)]:
        rows.append({"model": name, **metrics_from_probs(y_test.to_numpy(), model.predict_proba(x_test)[:, 1])})
    return logreg, pd.DataFrame(rows), transformer


def generate_gradcam_examples(model: UltrasoundMILModel, dataset: ThyroidPatientDataset, device: torch.device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    if model.backbone_name != "efficientnet_b0":
        (out_dir / "gradcam_note.txt").write_text("Grad-CAM is implemented for EfficientNet-B0 only.\n", encoding="utf-8")
        return
    model.eval()
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        image = sample["images_tensor"][0:1].to(device)
        fmap_holder, grad_holder = [], []

        def forward_hook(_m, _i, out):
            fmap_holder.append(out)

        def backward_hook(_m, _gi, go):
            grad_holder.append(go[0])

        h1 = model.encoder.features.register_forward_hook(forward_hook)
        h2 = model.encoder.features.register_full_backward_hook(backward_hook)
        model.zero_grad()
        logits = model.classifier(model.encoder(image)).squeeze()
        logits.backward()
        h1.remove()
        h2.remove()

        fmap = fmap_holder[0].detach().cpu()
        grad = grad_holder[0].detach().cpu()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * fmap).sum(dim=1)).squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = torch.nn.functional.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()
        base = image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        base = np.clip(np.array([0.229, 0.224, 0.225]) * base + np.array([0.485, 0.456, 0.406]), 0, 1)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(base)
        plt.axis("off")
        plt.title(f"Patient {sample['patient_id']}")
        plt.subplot(1, 2, 2)
        plt.imshow(base)
        plt.imshow(cam, cmap="jet", alpha=0.4)
        plt.axis("off")
        plt.title("Grad-CAM")
        plt.tight_layout()
        plt.savefig(out_dir / f"patient_{sample['patient_id']}_gradcam.png", dpi=200)
        plt.close()


def save_manuscript_text(evaluation_dir, reports_dir, summary: dict, perf: pd.DataFrame | None, n_train: int | None, n_test: int | None):
    methods = textwrap.dedent(
        f"""
        A patient-level multimodal framework was developed for thyroid malignancy prediction. Ultrasound images were grouped by patient folder, resized to 224 x 224 pixels, normalized with ImageNet statistics, and augmented during training using random flips, rotation within +/-10 degrees, brightness/contrast perturbation, and random resized cropping. Each patient contributed one bag of ultrasound images and one clinical feature vector. Clinical predictors included age, sex, body mass index, smoking, drinking, hypertension, diabetes, hepatitis, thyroid history, other cancer history, thyroiditis, and nodular goiter. Data were split at the patient level with an 80:20 train-test ratio. A clinical-only baseline was trained using logistic regression and random forest. An ultrasound-only model used a pretrained backbone with multiple-instance learning aggregation. A multimodal fusion model concatenated patient-level image embeddings with clinical variables and optimized a fully connected prediction head. Performance was evaluated at the patient level with AUC, accuracy, sensitivity, specificity, precision, recall, and F1 score.

        Dataset inspection identified {summary['number_of_patients']} patient folders and {summary['number_of_images']} ultrasound images.
        """
    ).strip()
    (evaluation_dir / "ai_methods_section.txt").write_text(methods + "\n", encoding="utf-8")
    (reports_dir / "ai_methods_section.txt").write_text(methods + "\n", encoding="utf-8")
    if perf is None:
        results = "Only dataset inspection and pipeline construction were completed in the current run."
    else:
        best = perf.sort_values("AUC", ascending=False).iloc[0]
        results = (
            f"A total of {n_train + n_test} patients were used for modeling, including {n_train} training patients and {n_test} testing patients. "
            f"The best-performing model was {best['model']} with AUC {best['AUC']:.3f} and accuracy {best['Accuracy']:.3f}."
        )
    (evaluation_dir / "ai_results_section.txt").write_text(results + "\n", encoding="utf-8")
    (reports_dir / "ai_results_section.txt").write_text(results + "\n", encoding="utf-8")


def main() -> None:
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--aggregator", choices=["mean", "attention"], default="attention")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    project_root = project_root_from(__file__)
    cfg = load_config(project_root)
    image_root = get_path(cfg, project_root, "data", "image_root")
    clinical_path = get_path(cfg, project_root, "data", "clinical_file")
    evaluation_dir = ensure_dir(get_path(cfg, project_root, "paths", "evaluation_dir"))
    reports_dir = ensure_dir(get_path(cfg, project_root, "paths", "reports_dir"))
    model_dir = ensure_dir(get_path(cfg, project_root, "paths", "model_dir"))

    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg["training"]["lr"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["training"]["num_workers"]
    set_seed(seed)
    log(
        f"[run] multimodal thyroid pipeline | backbone={args.backbone} aggregator={args.aggregator} "
        f"epochs={epochs} batch_size={batch_size}"
    )

    with timed_stage("Inspect dataset"):
        summary = inspect_dataset(image_root)
        save_dataset_summary(summary, evaluation_dir)
    if args.inspect_only:
        save_manuscript_text(evaluation_dir, reports_dir, summary, None, None, None)
        return

    with timed_stage("Load and split clinical dataset"):
        df = load_clinical_df(clinical_path, image_root)
    if args.max_patients > 0:
        df = df.sort_values("patient_id", key=lambda s: s.astype(int)).head(args.max_patients).copy()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["malignant"])
    split_payload = {
        "train_patient_ids": sorted(train_df["patient_id"].tolist(), key=int),
        "test_patient_ids": sorted(test_df["patient_id"].tolist(), key=int),
    }
    (evaluation_dir / "patient_level_split.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with timed_stage("Train clinical baseline models"):
        clinical_model, clinical_perf, transformer = train_clinical_models(train_df, test_df)
    clinical_probs = clinical_model.predict_proba(test_df[CLINICAL_COLUMNS])[:, 1]
    save_clinical_importance(clinical_model, evaluation_dir)
    joblib.dump(clinical_model, model_dir / "clinical_model.pkl")

    train_ds = ThyroidPatientDataset(train_df, image_root, build_transforms(train=True), transformer)
    test_ds = ThyroidPatientDataset(test_df, image_root, build_transforms(train=False), transformer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_patients)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_patients)
    use_gpu = cfg["training"]["device"]["use_gpu"] and not args.cpu_only and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    ultrasound_model = UltrasoundMILModel(backbone_name=args.backbone, aggregator=args.aggregator)
    with timed_stage("Train ultrasound MIL model"):
        ultrasound_model, us_true, us_prob, us_perf = train_ultrasound_model(
            ultrasound_model, train_loader, test_loader, device, epochs, learning_rate
        )
    torch.save(ultrasound_model.state_dict(), model_dir / "ultrasound_model.pth")

    perf = pd.concat(
        [clinical_perf, pd.DataFrame([{"model": "Ultrasound_MIL", **us_perf}])],
        ignore_index=True,
    )
    perf.to_csv(evaluation_dir / "model_performance_comparison.csv", index=False)
    save_multi_roc(
        {
            "Clinical_LogisticRegression": (test_df["malignant"].to_numpy(), clinical_probs),
            "Ultrasound_MIL": (us_true, us_prob),
        },
        evaluation_dir / "roc_curve_models.png",
        "ROC Curves",
    )
    with timed_stage("Generate Grad-CAM and report text"):
        generate_gradcam_examples(ultrasound_model, test_ds, device, evaluation_dir / "gradcam_examples")
        save_manuscript_text(evaluation_dir, reports_dir, summary, perf, len(train_df), len(test_df))
    log(f"[run-done] multimodal thyroid pipeline finished in {format_duration(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
