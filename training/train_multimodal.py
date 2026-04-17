from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import metrics_from_probs
from models.multimodal_model import FinalMultimodalModel, FusionMLP, UltrasoundMILModel, WeakSupervisedReasoningModel
from training.progress import log, progress, remaining_time


@torch.no_grad()
def evaluate_ultrasound_model(model: UltrasoundMILModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    for batch in progress(loader, total=len(loader), desc="Eval ultrasound", leave=False):
        images = [x.to(device) for x in batch["images_tensor"]]
        logits, _, _ = model(images)
        y_true.append(batch["label"].numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def train_ultrasound_model(
    model: UltrasoundMILModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[UltrasoundMILModel, np.ndarray, np.ndarray, dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -math.inf
    best_state = None
    epoch_start = time.perf_counter()
    epoch_bar = progress(range(1, epochs + 1), total=epochs, desc="Train ultrasound", leave=True)
    for epoch_idx in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = progress(train_loader, total=len(train_loader), desc=f"Ultrasound epoch {epoch_idx}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(batch_bar, start=1):
            images = [x.to(device) for x in batch["images_tensor"]]
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, _, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if hasattr(batch_bar, "set_postfix"):
                batch_bar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
        y_true, y_prob = evaluate_ultrasound_model(model, test_loader, device)
        auc = metrics_from_probs(y_true, y_prob)["AUC"]
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        eta = remaining_time(epoch_start, epoch_idx, epochs)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(auc=f"{auc:.4f}", best=f"{best_auc:.4f}", eta=eta)
        log(
            f"[epoch {epoch_idx}/{epochs}] stage=ultrasound "
            f"train_loss={running_loss / max(len(train_loader), 1):.4f} auc={auc:.4f} best_auc={best_auc:.4f} eta={eta}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    y_true, y_prob = evaluate_ultrasound_model(model, test_loader, device)
    return model, y_true, y_prob, metrics_from_probs(y_true, y_prob)


@torch.no_grad()
def extract_patient_embeddings(model: UltrasoundMILModel, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    patient_ids, embeddings, labels = [], [], []
    for batch in progress(loader, total=len(loader), desc="Extract embeddings", leave=False):
        images = [tensor.to(device) for tensor in batch["images_tensor"]]
        _, _, pooled = model(images)
        patient_ids.extend(batch["patient_id"])
        embeddings.append(pooled.cpu().numpy())
        labels.append(batch["label"].numpy())
    return patient_ids, np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


class FusionTabularDataset(torch.utils.data.Dataset):
    def __init__(self, image_embeddings: np.ndarray, clinical_features: np.ndarray, labels: np.ndarray) -> None:
        self.image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
        self.clinical_features = torch.tensor(clinical_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image_embedding": self.image_embeddings[index],
            "clinical_features": self.clinical_features[index],
            "label": self.labels[index],
        }


def train_fusion_model(
    train_embeddings: np.ndarray,
    train_clinical: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_clinical: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[FusionMLP, np.ndarray, dict[str, float]]:
    model = FusionMLP(train_embeddings.shape[1], train_clinical.shape[1]).to(device)
    train_ds = FusionTabularDataset(train_embeddings, train_clinical, train_labels)
    test_ds = FusionTabularDataset(test_embeddings, test_clinical, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -math.inf
    best_state = None
    epoch_start = time.perf_counter()
    epoch_bar = progress(range(1, epochs + 1), total=epochs, desc="Train fusion", leave=True)
    for epoch_idx in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = progress(train_loader, total=len(train_loader), desc=f"Fusion epoch {epoch_idx}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(batch_bar, start=1):
            optimizer.zero_grad()
            logits = model(batch["image_embedding"].to(device), batch["clinical_features"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if hasattr(batch_bar, "set_postfix"):
                batch_bar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
        probs, labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in progress(test_loader, total=len(test_loader), desc="Eval fusion", leave=False):
                probs.append(torch.sigmoid(model(batch["image_embedding"].to(device), batch["clinical_features"].to(device))).cpu().numpy())
                labels.append(batch["label"].numpy())
        y_prob = np.concatenate(probs)
        y_true = np.concatenate(labels)
        auc = metrics_from_probs(y_true, y_prob)["AUC"]
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        eta = remaining_time(epoch_start, epoch_idx, epochs)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(auc=f"{auc:.4f}", best=f"{best_auc:.4f}", eta=eta)
        log(
            f"[epoch {epoch_idx}/{epochs}] stage=fusion "
            f"train_loss={running_loss / max(len(train_loader), 1):.4f} auc={auc:.4f} best_auc={best_auc:.4f} eta={eta}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in progress(test_loader, total=len(test_loader), desc="Final eval fusion", leave=False):
            probs.append(torch.sigmoid(model(batch["image_embedding"].to(device), batch["clinical_features"].to(device))).cpu().numpy())
    y_prob = np.concatenate(probs)
    return model, y_prob, metrics_from_probs(test_labels, y_prob)


def _extract_final_logit(output):
    if isinstance(output, dict):
        return output["final_logit"]
    return output[0]


def _compute_attention_regularization(
    attention_weights_all: list[torch.Tensor],
    target_entropy: float,
) -> torch.Tensor:
    penalties = []
    for weights in attention_weights_all:
        if weights.numel() <= 1:
            continue
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum()
        normalized_entropy = entropy / math.log(weights.numel())
        penalties.append((normalized_entropy - target_entropy) ** 2)
    if not penalties:
        device = attention_weights_all[0].device if attention_weights_all else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    return torch.stack(penalties).mean()


def _compute_local_weak_loss(
    local_logits_all: list[torch.Tensor],
    labels: torch.Tensor,
    local_top_k: int,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for local_logits, label in zip(local_logits_all, labels):
        if local_logits.numel() == 0:
            continue
        if float(label.item()) >= 0.5:
            k = min(local_top_k, local_logits.numel())
            top_logits = torch.topk(local_logits, k=k).values
            target = torch.ones_like(top_logits)
            losses.append(criterion(top_logits, target))
        else:
            target = torch.zeros_like(local_logits)
            losses.append(criterion(local_logits, target))
    if not losses:
        device = labels.device
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def _compute_final_multimodal_loss(model, output, labels: torch.Tensor, loss_cfg: dict | None) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    if not isinstance(output, dict):
        loss = criterion(output[0], labels)
        return loss, {"total": float(loss.item())}

    loss_cfg = loss_cfg or {}
    global_weight = float(loss_cfg.get("global_weight", 0.5))
    fusion_weight = float(loss_cfg.get("fusion_weight", 1.0))
    local_weight = float(loss_cfg.get("local_weight", 0.3))
    attention_reg_weight = float(loss_cfg.get("attention_reg_weight", 0.05))
    attention_entropy_target = float(loss_cfg.get("attention_entropy_target", 0.6))
    local_top_k = int(loss_cfg.get("local_top_k", getattr(model, "local_top_k", 2)))

    global_loss = criterion(output["global_logit"], labels)
    fusion_loss = criterion(output["final_logit"], labels)
    local_loss = _compute_local_weak_loss(output["local_logits"], labels, local_top_k)
    attention_loss = _compute_attention_regularization(output["attention_weights"], attention_entropy_target)
    total_loss = (
        global_weight * global_loss
        + fusion_weight * fusion_loss
        + local_weight * local_loss
        + attention_reg_weight * attention_loss
    )
    return total_loss, {
        "total": float(total_loss.item()),
        "global": float(global_loss.item()),
        "fusion": float(fusion_loss.item()),
        "local": float(local_loss.item()),
        "attention": float(attention_loss.item()),
    }


@torch.no_grad()
def evaluate_final_multimodal_model(model: FinalMultimodalModel | WeakSupervisedReasoningModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    for batch in progress(loader, total=len(loader), desc="Eval final multimodal", leave=False):
        images = [x.to(device) for x in batch["images_tensor"]]
        output = model(images, batch["clinical_vector"].to(device))
        logits = _extract_final_logit(output)
        y_true.append(batch["label"].numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def train_final_multimodal_model(
    model: FinalMultimodalModel | WeakSupervisedReasoningModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    loss_cfg: dict | None = None,
) -> tuple[FinalMultimodalModel | WeakSupervisedReasoningModel, np.ndarray, np.ndarray, dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_auc = -math.inf
    best_state = None
    epoch_start = time.perf_counter()
    epoch_bar = progress(range(1, epochs + 1), total=epochs, desc="Train final multimodal", leave=True)
    for epoch_idx in epoch_bar:
        model.train()
        running_loss = 0.0
        running_global = 0.0
        running_fusion = 0.0
        running_local = 0.0
        running_attention = 0.0
        batch_bar = progress(train_loader, total=len(train_loader), desc=f"Final epoch {epoch_idx}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(batch_bar, start=1):
            images = [x.to(device) for x in batch["images_tensor"]]
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            output = model(images, batch["clinical_vector"].to(device))
            loss, parts = _compute_final_multimodal_loss(model, output, labels, loss_cfg)
            loss.backward()
            optimizer.step()
            running_loss += parts["total"]
            running_global += parts.get("global", 0.0)
            running_fusion += parts.get("fusion", 0.0)
            running_local += parts.get("local", 0.0)
            running_attention += parts.get("attention", 0.0)
            if hasattr(batch_bar, "set_postfix"):
                batch_bar.set_postfix(
                    loss=f"{running_loss / batch_idx:.4f}",
                    fusion=f"{running_fusion / batch_idx:.4f}" if running_fusion else "n/a",
                )
        y_true, y_prob = evaluate_final_multimodal_model(model, test_loader, device)
        auc = metrics_from_probs(y_true, y_prob)["AUC"]
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        eta = remaining_time(epoch_start, epoch_idx, epochs)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(auc=f"{auc:.4f}", best=f"{best_auc:.4f}", eta=eta)
        log(
            f"[epoch {epoch_idx}/{epochs}] stage=final_multimodal "
            f"train_loss={running_loss / max(len(train_loader), 1):.4f} "
            f"global={running_global / max(len(train_loader), 1):.4f} "
            f"fusion={running_fusion / max(len(train_loader), 1):.4f} "
            f"local={running_local / max(len(train_loader), 1):.4f} "
            f"attn={running_attention / max(len(train_loader), 1):.4f} "
            f"auc={auc:.4f} best_auc={best_auc:.4f} eta={eta}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    y_true, y_prob = evaluate_final_multimodal_model(model, test_loader, device)
    return model, y_true, y_prob, metrics_from_probs(y_true, y_prob)
