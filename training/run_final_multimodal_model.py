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
import pandas as pd
import torch
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
from evaluation.roc_plot import save_single_roc
from models.multimodal_model import FinalMultimodalModel, WeakSupervisedReasoningModel
from training.progress import format_duration, log, timed_stage
from training.run_multimodal_thyroid_pipeline import generate_gradcam_examples, save_clinical_importance
from training.run_patient_level_fusion import save_shap_outputs
from training.train_multimodal import train_final_multimodal_model


def parse_best_backbone(evaluation_dir):
    path = evaluation_dir / "best_backbone.txt"
    if not path.exists():
        return "efficientnet_b0"
    first = path.read_text(encoding="utf-8").splitlines()[0].strip()
    return first.split(":", 1)[1].strip() if ":" in first else first


def model_artifact_prefix(model_type: str) -> str:
    return f"final_multimodal_{model_type}"


def save_manuscript_sections(evaluation_dir, reports_dir, artifact_prefix, backbone_name, summary, train_size, test_size, metrics):
    methods = textwrap.dedent(
        f"""
        A final patient-level multimodal model was developed for thyroid malignancy prediction by integrating ultrasound images and clinical variables. Ultrasound images were organized by patient identifier, and all images from the same patient were processed jointly to avoid image-level leakage. The selected image backbone from the benchmark stage was {backbone_name}. Each image was resized to 224 x 224 pixels, normalized using ImageNet mean and standard deviation, and augmented during training with horizontal flipping, random rotation within +/-10 degrees, and brightness adjustment.

        Patient-level image modeling was performed using a sequence-based framework. Image embeddings extracted by the selected backbone were projected into a hidden dimension of 768 and passed through a Transformer encoder comprising 2 layers and 8 attention heads to generate context-aware image representations. Attention-based multiple instance learning was then used to compute a weighted patient-level image representation from the full image sequence. Clinical variables included age, sex, body mass index, smoking, drinking, hypertension, diabetes, hepatitis, thyroid history, other cancer history, thyroiditis, and nodular goiter. Patients were split at the patient level into training and testing cohorts using an 80:20 stratified design based on malignancy status.

        The patient-level image representation was fused with a processed clinical feature vector and passed to a prediction head for binary malignancy classification. Model training used binary cross-entropy loss and Adam optimizer with a learning rate of 1e-4. Performance was evaluated at the patient level using area under the receiver operating characteristic curve, accuracy, sensitivity, specificity, precision, recall, and F1-score.
        """
    ).strip()
    results = textwrap.dedent(
        f"""
        Dataset inspection identified {summary['number_of_patients']} patient folders and {summary['number_of_images']} ultrasound images. After matching the clinical spreadsheet to patient folders and excluding patients without images, the final multimodal analysis included {train_size + test_size} patients, of whom {train_size} were assigned to the training cohort and {test_size} to the testing cohort.

        In the current run, the final multimodal model achieved an AUC of {metrics['AUC']:.3f}, accuracy of {metrics['Accuracy']:.3f}, sensitivity of {metrics['Sensitivity']:.3f}, specificity of {metrics['Specificity']:.3f}, precision of {metrics['Precision']:.3f}, recall of {metrics['Recall']:.3f}, and F1-score of {metrics['F1']:.3f}. SHAP analysis was generated for the clinical component, and Grad-CAM examples were produced for representative ultrasound images.
        """
    ).strip()
    for path in [evaluation_dir / f"{artifact_prefix}_methods_section.txt", reports_dir / f"{artifact_prefix}_methods_section.txt"]:
        path.write_text(methods + "\n", encoding="utf-8")
    for path in [evaluation_dir / f"{artifact_prefix}_results_section.txt", reports_dir / f"{artifact_prefix}_results_section.txt"]:
        path.write_text(results + "\n", encoding="utf-8")


def main() -> None:
    run_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Final patient-level multimodal thyroid malignancy model.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--backbone", default="")
    args = parser.parse_args()

    project_root = project_root_from(__file__)
    cfg = load_config(project_root)
    image_root = get_path(cfg, project_root, "data", "image_root")
    clinical_path = get_path(cfg, project_root, "data", "clinical_file")
    evaluation_dir = ensure_dir(get_path(cfg, project_root, "paths", "evaluation_dir"))
    reports_dir = ensure_dir(get_path(cfg, project_root, "paths", "reports_dir"))
    model_dir = ensure_dir(get_path(cfg, project_root, "paths", "model_dir"))
    backbone_name = args.backbone or parse_best_backbone(evaluation_dir)
    model_type = cfg["model"].get("model_type", "final_multimodal")
    artifact_prefix = model_artifact_prefix(model_type)

    seed = args.seed if args.seed is not None else cfg["training"]["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["training"]["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg["training"]["lr"]
    num_workers = args.num_workers if args.num_workers is not None else cfg["training"]["num_workers"]
    set_seed(seed)
    log(
        f"[run] final multimodal model | model_type={model_type} backbone={backbone_name} "
        f"epochs={epochs} batch_size={batch_size}"
    )

    with timed_stage("Inspect dataset and load clinical data"):
        summary = inspect_dataset(image_root)
        save_dataset_summary(summary, evaluation_dir)
        df = load_clinical_df(clinical_path, image_root)
    if args.max_patients > 0:
        df = df.sort_values("patient_id", key=lambda s: s.astype(int)).head(args.max_patients).copy()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["malignant"])
    split_payload = {
        "train_patient_ids": sorted(train_df["patient_id"].tolist(), key=int),
        "test_patient_ids": sorted(test_df["patient_id"].tolist(), key=int),
    }
    (evaluation_dir / f"{artifact_prefix}_patient_split.json").write_text(json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    clinical_transformer = build_clinical_transformer(train_df)
    train_ds = ThyroidPatientDataset(train_df, image_root, build_transforms(train=True), clinical_transformer)
    test_ds = ThyroidPatientDataset(test_df, image_root, build_transforms(train=False), clinical_transformer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_patients)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_patients)
    use_gpu = cfg["training"]["device"]["use_gpu"] and not args.cpu_only and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    if model_type == "weak_supervised_reasoning":
        model = WeakSupervisedReasoningModel(
            backbone_name=backbone_name,
            clinical_dim=train_ds.clinical_matrix.shape[1],
            hidden_dim=cfg["model"]["hidden_dim"],
            transformer_layers=cfg["model"]["transformer_layers"],
            transformer_heads=cfg["model"]["transformer_heads"],
            local_top_k=cfg["model"].get("local_top_k", 2),
            use_cls_token=cfg["model"].get("use_cls_token", False),
        )
    else:
        model = FinalMultimodalModel(
            backbone_name=backbone_name,
            clinical_dim=train_ds.clinical_matrix.shape[1],
            hidden_dim=cfg["model"]["hidden_dim"],
            transformer_layers=cfg["model"]["transformer_layers"],
            transformer_heads=cfg["model"]["transformer_heads"],
        )
    with timed_stage("Train final multimodal model"):
        loss_cfg = dict(cfg.get("loss", {}))
        loss_cfg["local_top_k"] = cfg["model"].get("local_top_k", 2)
        model, y_true, y_prob, metrics = train_final_multimodal_model(
            model,
            train_loader,
            test_loader,
            device,
            epochs,
            learning_rate,
            loss_cfg=loss_cfg,
        )
    torch.save(model.state_dict(), model_dir / f"{artifact_prefix}_model.pth")
    joblib.dump(clinical_transformer, model_dir / f"{artifact_prefix}_clinical_transformer.pkl")

    pd.DataFrame([{"model": model_type, "backbone": backbone_name, **metrics}]).to_csv(
        evaluation_dir / f"{artifact_prefix}_performance.csv", index=False
    )
    save_single_roc(
        y_true,
        y_prob,
        evaluation_dir / f"{artifact_prefix}_roc_curve.png",
        f"Final multimodal ROC curve ({model_type})",
        f"Final multimodal model ({model_type})",
    )

    comparison_path = evaluation_dir / "final_model_comparison.csv"
    comparison_cols = ["Model", "AUC", "Accuracy", "Sensitivity", "Specificity"]
    comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame(columns=comparison_cols)
    row = {
        "Model": f"{model_type} ({backbone_name})",
        "AUC": metrics["AUC"],
        "Accuracy": metrics["Accuracy"],
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
    }
    comparison_df = comparison_df[comparison_df["Model"] != row["Model"]]
    comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
    comparison_df.to_csv(comparison_path, index=False)

    with timed_stage("Generate explainability outputs and manuscript sections"):
        clinical_model = Pipeline([("pre", clinical_transformer), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        clinical_model.fit(train_df[CLINICAL_COLUMNS], train_df["malignant"].astype(int))
        save_clinical_importance(clinical_model, evaluation_dir)
        save_shap_outputs(clinical_model, train_df, evaluation_dir)
        generate_gradcam_examples(model_to_ultrasound_wrapper(model, backbone_name), test_ds, device, evaluation_dir / "explainability")
        save_manuscript_sections(
            evaluation_dir,
            reports_dir,
            artifact_prefix,
            backbone_name,
            summary,
            len(train_df),
            len(test_df),
            metrics,
        )
    log(f"[run-done] final multimodal model finished in {format_duration(time.perf_counter() - run_start)}")


def model_to_ultrasound_wrapper(model: FinalMultimodalModel | WeakSupervisedReasoningModel, backbone_name: str):
    class Wrapper(torch.nn.Module):
        def __init__(self, final_model: FinalMultimodalModel) -> None:
            super().__init__()
            self.encoder = final_model.encoder
            classifier_in_features = getattr(final_model.projector, "in_features", final_model.projector.out_features)
            self.classifier = torch.nn.Linear(classifier_in_features, 1)
            self.backbone_name = backbone_name
            if hasattr(self.encoder, "features"):
                self.encoder.features = self.encoder.features

    return Wrapper(model)


if __name__ == "__main__":
    main()
