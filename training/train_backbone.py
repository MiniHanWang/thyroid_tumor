from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import metrics_from_probs
from models.multimodal_model import PatientBackboneClassifier
from training.progress import log, progress, remaining_time


@torch.no_grad()
def evaluate_backbone_model(model: PatientBackboneClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    for batch in progress(loader, total=len(loader), desc="Eval", leave=False):
        images = [x.to(device) for x in batch["images_tensor"]]
        logits, _ = model(images)
        y_true.append(batch["label"].numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def load_backbone_checkpoint(
    backbone_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> PatientBackboneClassifier:
    model = PatientBackboneClassifier(backbone_name).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def train_backbone_model(
    backbone_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[PatientBackboneClassifier, np.ndarray, np.ndarray, dict[str, float]]:
    model = PatientBackboneClassifier(backbone_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -math.inf
    best_state = None
    epoch_start = time.perf_counter()
    epoch_bar = progress(range(1, epochs + 1), total=epochs, desc=f"Train {backbone_name}", leave=True)
    for epoch_idx in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = progress(train_loader, total=len(train_loader), desc=f"{backbone_name} epoch {epoch_idx}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(batch_bar, start=1):
            images = [x.to(device) for x in batch["images_tensor"]]
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if hasattr(batch_bar, "set_postfix"):
                batch_bar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
        y_true, y_prob = evaluate_backbone_model(model, test_loader, device)
        auc = metrics_from_probs(y_true, y_prob)["AUC"]
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        eta = remaining_time(epoch_start, epoch_idx, epochs)
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(auc=f"{auc:.4f}", best=f"{best_auc:.4f}", eta=eta)
        log(
            f"[epoch {epoch_idx}/{epochs}] backbone={backbone_name} "
            f"train_loss={running_loss / max(len(train_loader), 1):.4f} auc={auc:.4f} best_auc={best_auc:.4f} eta={eta}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    y_true, y_prob = evaluate_backbone_model(model, test_loader, device)
    return model, y_true, y_prob, metrics_from_probs(y_true, y_prob)
