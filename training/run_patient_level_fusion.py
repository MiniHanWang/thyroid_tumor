from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
from evaluation.roc_plot import save_single_roc
from models.multimodal_model import UltrasoundMILModel
from training.progress import format_duration, log, timed_stage
from training.run_multimodal_thyroid_pipeline import generate_gradcam_examples, save_clinical_importance, train_clinical_models
from training.train_multimodal import extract_patient_embeddings, train_fusion_model, train_ultrasound_model


def save_shap_outputs(model: Pipeline, train_df: pd.DataFrame, evaluation_dir):
    import shap

    explainability_dir = evaluation_dir / "explainability"
    explainability_dir.mkdir(parents=True, exist_ok=True)
    try:
        preprocessor = model.named_steps["pre"]
        classifier = model.named_steps["clf"]
        transformed = preprocessor.transform(train_df[CLINICAL_COLUMNS])
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.LinearExplainer(classifier, transformed)
        shap_values = explainer.shap_values(transformed)
        plt.figure()
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(explainability_dir / "clinical_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        mean_abs = np.abs(shap_values).mean(axis=0)
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(explainability_dir / "clinical_shap_importance.csv", index=False)
    except Exception as exc:
        (explainability_dir / "clinical_shap_not_available.txt").write_text(f"{exc}\n", encoding="utf-8")


def save_merged_dataset(patient_ids, clinical_df, image_embeddings, evaluation_dir):
    embedding_cols = [f"image_embedding_{i:04d}" for i in range(image_embeddings.shape[1])]
    image_df = pd.DataFrame(image_embeddings, columns=embedding_cols)
    image_df.insert(0, "patient_id", patient_ids)
    image_df.to_csv(evaluation_dir / "patient_image_embeddings.csv", index=False)
    merged = clinical_df.merge(image_df, on="patient_id", how="inner")
    merged.to_csv(evaluation_dir / "fusion_merged_dataset.csv", index=False)
    return merged


def save_manuscript_sections(evaluation_dir, reports_dir, summary, train_size, test_size, fusion_metrics):
    methods = textwrap.dedent(
        f"""
        A patient-level multimodal fusion framework was developed to integrate structured clinical variables with ultrasound-derived image embeddings for thyroid malignancy prediction. Ultrasound images were organized by patient identifier, and all images from a given patient were aggregated into a single patient-level representation using a convolutional neural network with multi-image aggregation. Clinical variables included age, sex, body mass index, smoking, drinking, hypertension, diabetes, hepatitis, thyroid history, other cancer history, thyroiditis, and nodular goiter. Clinical and imaging data were merged by patient_id to ensure one row per patient.

        The multimodal cohort contained {summary['number_of_patients']} patients and {summary['number_of_images']} ultrasound images overall. Patients were split at the patient level using stratified sampling with an 80:20 training-test ratio based on malignancy label. Continuous clinical variables, including age and body mass index, were normalized using training-set statistics. Patient-level image embeddings extracted from the ultrasound model were standardized before fusion. The fusion neural network consisted of an image branch mapping image embeddings to 256 hidden units, a clinical branch mapping clinical features to 32 hidden units, concatenation of both branches, and a fusion head comprising a 128-unit dense layer with ReLU activation and dropout of 0.3, followed by a single sigmoid output. The model was optimized using binary cross-entropy loss and Adam with a learning rate of 1e-4. Performance was evaluated at the patient level using AUC, accuracy, sensitivity, specificity, precision, recall, and F1-score.
        """
    ).strip()
    results = textwrap.dedent(
        f"""
        The multimodal fusion dataset was successfully merged by patient_id, yielding one patient-level record per case with clinical variables, aggregated image embeddings, and malignancy labels. In the current run, {train_size} patients were assigned to the training cohort and {test_size} patients to the testing cohort. The fusion model achieved an AUC of {fusion_metrics['AUC']:.3f}, accuracy of {fusion_metrics['Accuracy']:.3f}, sensitivity of {fusion_metrics['Sensitivity']:.3f}, specificity of {fusion_metrics['Specificity']:.3f}, precision of {fusion_metrics['Precision']:.3f}, recall of {fusion_metrics['Recall']:.3f}, and F1-score of {fusion_metrics['F1']:.3f}.

        Comparative analysis across the clinical-only, ultrasound-only, and fusion models is summarized in the model comparison table. Explainability outputs were also generated, including SHAP analysis for clinical variables and Grad-CAM examples for representative ultrasound images.
        """
    ).strip()
    for path in [evaluation_dir / "fusion_methods_section.txt", reports_dir / "fusion_methods_section.txt"]:
        path.write_text(methods + "\n", encoding="utf-8")
    for path in [evaluation_dir / "fusion_results_section.txt", reports_dir / "fusion_results_section.txt"]:
        path.write_text(results + "\n", encoding="utf-8")


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
        f"[run] patient-level fusion | backbone={args.backbone} aggregator={args.aggregator} "
        f"epochs={epochs} batch_size={batch_size}"
    )

    with timed_stage("Inspect dataset and load clinical data"):
        summary = inspect_dataset(image_root)
        save_dataset_summary(summary, evaluation_dir)
        df = load_clinical_df(clinical_path, image_root)
    if args.max_patients > 0:
        df = df.sort_values("patient_id", key=lambda s: s.astype(int)).head(args.max_patients).copy()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["malignant"])
    with timed_stage("Train clinical baseline"):
        clinical_model, clinical_perf_df, transformer = train_clinical_models(train_df, test_df)
    clinical_probs = clinical_model.predict_proba(test_df[CLINICAL_COLUMNS])[:, 1]
    save_clinical_importance(clinical_model, evaluation_dir)
    save_shap_outputs(clinical_model, train_df, evaluation_dir)
    joblib.dump(clinical_model, model_dir / "clinical_model.pkl")

    train_ds = ThyroidPatientDataset(train_df, image_root, build_transforms(train=True), transformer)
    test_ds = ThyroidPatientDataset(test_df, image_root, build_transforms(train=False), transformer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_patients)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_patients)
    embed_loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_patients)
    embed_loader_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_patients)
    use_gpu = cfg["training"]["device"]["use_gpu"] and not args.cpu_only and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    ultrasound_model = UltrasoundMILModel(backbone_name=args.backbone, aggregator=args.aggregator)
    with timed_stage("Train ultrasound model"):
        ultrasound_model, us_labels, us_probs, us_metrics = train_ultrasound_model(
            ultrasound_model, train_loader, test_loader, device, epochs, learning_rate
        )
    torch.save(ultrasound_model.state_dict(), model_dir / "ultrasound_model.pth")

    with timed_stage("Extract patient embeddings"):
        train_ids, train_embeddings, train_embed_labels = extract_patient_embeddings(ultrasound_model, embed_loader_train, device)
        test_ids, test_embeddings, test_embed_labels = extract_patient_embeddings(ultrasound_model, embed_loader_test, device)
    scaler = StandardScaler().fit(train_embeddings)
    train_embeddings_scaled = scaler.transform(train_embeddings)
    test_embeddings_scaled = scaler.transform(test_embeddings)
    joblib.dump(scaler, model_dir / "image_embedding_scaler.pkl")
    train_clinical = np.asarray(train_ds.clinical_matrix, dtype=np.float32)
    test_clinical = np.asarray(test_ds.clinical_matrix, dtype=np.float32)
    save_merged_dataset(train_ids + test_ids, pd.concat([train_df, test_df], ignore_index=True), np.vstack([train_embeddings_scaled, test_embeddings_scaled]), evaluation_dir)

    with timed_stage("Train fusion model"):
        fusion_model, fusion_probs, fusion_metrics = train_fusion_model(
            train_embeddings_scaled,
            train_clinical,
            train_embed_labels,
            test_embeddings_scaled,
            test_clinical,
            test_embed_labels,
            device,
            epochs,
            learning_rate,
            batch_size,
        )
    torch.save(fusion_model.state_dict(), model_dir / "multimodal_model.pth")
    pd.DataFrame([{"model": "Fusion_Model", **fusion_metrics}]).to_csv(evaluation_dir / "fusion_model_performance.csv", index=False)
    save_single_roc(test_embed_labels, fusion_probs, evaluation_dir / "fusion_roc_curve.png", "Fusion model ROC curve", "Fusion model")

    best_clinical = clinical_perf_df.sort_values("AUC", ascending=False).iloc[0]
    comparison = pd.DataFrame(
        [
            {"Model": f"Clinical model ({best_clinical['model']})", "AUC": best_clinical["AUC"], "Accuracy": best_clinical["Accuracy"], "Sensitivity": best_clinical["Sensitivity"], "Specificity": best_clinical["Specificity"]},
            {"Model": "Ultrasound model", "AUC": us_metrics["AUC"], "Accuracy": us_metrics["Accuracy"], "Sensitivity": us_metrics["Sensitivity"], "Specificity": us_metrics["Specificity"]},
            {"Model": "Fusion model", "AUC": fusion_metrics["AUC"], "Accuracy": fusion_metrics["Accuracy"], "Sensitivity": fusion_metrics["Sensitivity"], "Specificity": fusion_metrics["Specificity"]},
        ]
    )
    comparison.to_csv(evaluation_dir / "model_comparison.csv", index=False)
    with timed_stage("Generate explainability outputs and manuscript sections"):
        generate_gradcam_examples(ultrasound_model, test_ds, device, evaluation_dir / "explainability")
        save_manuscript_sections(evaluation_dir, reports_dir, summary, len(train_df), len(test_df), fusion_metrics)
    log(f"[run-done] patient-level fusion finished in {format_duration(time.perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
