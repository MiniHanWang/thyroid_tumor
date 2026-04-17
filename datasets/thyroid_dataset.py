from __future__ import annotations

import json
import random
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLINICAL_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "smoking_bin",
    "drinking_bin",
    "hypertension",
    "diabetes",
    "hepatitis",
    "thyroid_history",
    "other_cancer_history",
    "thyroiditis_flag",
    "nodular_goiter_flag",
]


def project_root_from(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[1]


def load_config(project_root: Path) -> dict[str, Any]:
    with (project_root / "configs" / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(cfg: dict[str, Any], project_root: Path, section: str, key: str) -> Path:
    return project_root / cfg[section][key]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def image_count(patient_dir: Path) -> int:
    return sum(1 for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def iter_patient_dirs(image_root: Path) -> list[Path]:
    patient_dirs = [p for p in image_root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(patient_dirs, key=lambda p: int(p.name))


def inspect_dataset(image_root: Path) -> dict[str, Any]:
    patient_map = {}
    counts = []
    for patient_dir in iter_patient_dirs(image_root):
        files = sorted([p.name for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        patient_map[patient_dir.name] = {"image_count": len(files), "images": files}
        counts.append(len(files))
    return {
        "number_of_patients": len(patient_map),
        "number_of_images": int(sum(counts)),
        "images_per_patient": patient_map,
        "stats": {
            "min": int(min(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
            "mean": float(np.mean(counts)) if counts else 0.0,
            "median": float(np.median(counts)) if counts else 0.0,
            "patients_with_zero_images": int(sum(c == 0 for c in counts)),
        },
    }


def save_dataset_summary(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def load_clinical_df(clinical_path: Path, image_root: Path) -> pd.DataFrame:
    if clinical_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(clinical_path)
    else:
        df = pd.read_csv(clinical_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str).str.strip()
    elif "id" in df.columns:
        df["patient_id"] = df["id"].astype(str).str.strip()
    else:
        raise ValueError("Clinical file must contain patient_id or id.")
    if "malignant" not in df.columns:
        raise ValueError("Clinical file must contain malignant.")
    if "final_analysis_include" in df.columns:
        df = df[df["final_analysis_include"].fillna(0).astype(int) == 1].copy()
    counts = {p.name: image_count(p) for p in iter_patient_dirs(image_root)}
    df["image_count"] = df["patient_id"].map(counts).fillna(0).astype(int)
    df = df[df["image_count"] > 0].copy()
    df["patient_dir"] = df["patient_id"].map(lambda x: str(image_root / str(x)))
    df = df[df["patient_id"].map(lambda x: (image_root / str(x)).exists())].copy()
    df["malignant"] = df["malignant"].astype(float).astype(int)
    return df


def build_transforms(train: bool) -> transforms.Compose:
    ops: list[Any] = [transforms.Resize((224, 224))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    return transforms.Compose(ops)


def build_clinical_transformer(train_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in CLINICAL_COLUMNS if c != "sex"]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["sex"],
            ),
        ]
    ).fit(train_df[CLINICAL_COLUMNS])


class ThyroidPatientDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        image_transform: transforms.Compose,
        clinical_transformer: ColumnTransformer | None = None,
        *,
        drop_invalid_patients: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_root = image_root
        self.image_transform = image_transform
        self.df = self._validate_dataframe(self.df, drop_invalid_patients=drop_invalid_patients)
        self.patient_image_paths = self._build_patient_image_paths(self.df)
        self.clinical_matrix = None
        if clinical_transformer is not None:
            matrix = clinical_transformer.transform(self.df[CLINICAL_COLUMNS])
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            self.clinical_matrix = np.asarray(matrix, dtype=np.float32)

    def _validate_dataframe(self, dataframe: pd.DataFrame, *, drop_invalid_patients: bool) -> pd.DataFrame:
        valid_indices: list[int] = []
        invalid_rows: list[tuple[str, str]] = []
        for row in dataframe.itertuples():
            patient_id = str(getattr(row, "patient_id")).strip()
            patient_dir = self.image_root / patient_id
            if not patient_dir.exists():
                invalid_rows.append((patient_id, f"missing directory: {patient_dir}"))
                continue
            has_image = any(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in patient_dir.iterdir()
            )
            if not has_image:
                invalid_rows.append((patient_id, f"no supported image files in: {patient_dir}"))
                continue
            valid_indices.append(row.Index)

        if invalid_rows and not drop_invalid_patients:
            patient_id, reason = invalid_rows[0]
            raise FileNotFoundError(f"Invalid patient sample {patient_id}: {reason}")

        if invalid_rows:
            preview = ", ".join(f"{patient_id} ({reason})" for patient_id, reason in invalid_rows[:5])
            warnings.warn(
                f"Dropping {len(invalid_rows)} invalid patient rows before dataset construction: {preview}",
                stacklevel=2,
            )

        return dataframe.loc[valid_indices].reset_index(drop=True).copy()

    def _build_patient_image_paths(self, dataframe: pd.DataFrame) -> list[list[Path]]:
        patient_image_paths: list[list[Path]] = []
        for row in dataframe.itertuples():
            patient_id = str(getattr(row, "patient_id")).strip()
            patient_dir = self.image_root / patient_id
            image_paths = [
                path
                for path in sorted(patient_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not image_paths:
                raise FileNotFoundError(f"Patient {patient_id} has no cached image paths in {patient_dir}")
            patient_image_paths.append(image_paths)
        return patient_image_paths

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_tensor(self, path: Path, patient_id: str) -> torch.Tensor:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                # Read bytes explicitly to avoid sporadic Windows path/file-handle issues
                # observed during long training runs.
                with path.open("rb") as f:
                    image_bytes = f.read()
                with Image.open(BytesIO(image_bytes)) as img:
                    rgb_img = img.convert("RGB")
                    return self.image_transform(rgb_img)
            except (OSError, FileNotFoundError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                break
        raise OSError(
            f"Failed to load image for patient {patient_id} from {path}: {last_error}"
        ) from last_error

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        patient_id = row["patient_id"]
        tensors = []
        load_errors: list[str] = []
        for path in self.patient_image_paths[index]:
            try:
                tensors.append(self._load_image_tensor(path, patient_id))
            except OSError as exc:
                load_errors.append(str(exc))
        if not tensors:
            error_preview = "; ".join(load_errors[:3]) if load_errors else "no cached image paths"
            raise ValueError(f"Patient {patient_id} has no readable images. Errors: {error_preview}")
        if load_errors:
            warnings.warn(
                f"Patient {patient_id}: skipped {len(load_errors)} unreadable images during loading.",
                stacklevel=2,
            )
        item = {
            "patient_id": patient_id,
            "images_tensor": torch.stack(tensors, dim=0),
            "label": torch.tensor(float(row["malignant"]), dtype=torch.float32),
        }
        if self.clinical_matrix is not None:
            item["clinical_vector"] = torch.tensor(self.clinical_matrix[index], dtype=torch.float32)
        return item


def collate_patients(batch: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "patient_id": [b["patient_id"] for b in batch],
        "images_tensor": [b["images_tensor"] for b in batch],
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }
    if "clinical_vector" in batch[0]:
        payload["clinical_vector"] = torch.stack([b["clinical_vector"] for b in batch], dim=0)
    return payload
