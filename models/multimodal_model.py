from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_backbones import build_backbone


class MeanPooling(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.full(
            (embeddings.shape[0],),
            1.0 / embeddings.shape[0],
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        return embeddings.mean(dim=0), weights


class AttentionMIL(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 256) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(embeddings).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=0)
        return pooled, weights


def normalize_backbone_output(embeddings: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Normalize torchvision backbone outputs to [N, feature_dim]."""
    if embeddings.ndim == 2 and embeddings.shape[-1] == feature_dim:
        return embeddings
    if embeddings.ndim > 2:
        # CNN backbones may return [N, C, H, W] when the classifier head is removed.
        if embeddings.shape[1] == feature_dim:
            embeddings = F.adaptive_avg_pool2d(embeddings, output_size=1).flatten(1)
        else:
            embeddings = embeddings.flatten(1)
    if embeddings.ndim == 2 and embeddings.shape[-1] == feature_dim:
        return embeddings
    raise RuntimeError(
        f"Backbone output shape {tuple(embeddings.shape)} is incompatible with feature_dim={feature_dim}."
    )


class PatientBackboneClassifier(nn.Module):
    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, feature_dim = build_backbone(backbone_name)
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, patient_images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = []
        for images in patient_images:
            image_embeddings = normalize_backbone_output(self.encoder(images), self.feature_dim)
            pooled.append(image_embeddings.mean(dim=0))
        patient_embeddings = torch.stack(pooled, dim=0)
        logits = self.classifier(patient_embeddings).squeeze(-1)
        return logits, patient_embeddings


class UltrasoundMILModel(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", aggregator: str = "attention") -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, self.feature_dim = build_backbone(backbone_name)
        self.aggregator = MeanPooling() if aggregator == "mean" else AttentionMIL(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, patient_images: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        pooled = []
        weights_all = []
        for images in patient_images:
            image_embeddings = normalize_backbone_output(self.encoder(images), self.feature_dim)
            patient_embedding, weights = self.aggregator(image_embeddings)
            pooled.append(patient_embedding)
            weights_all.append(weights)
        pooled_tensor = torch.stack(pooled, dim=0)
        logits = self.classifier(pooled_tensor).squeeze(-1)
        return logits, weights_all, pooled_tensor


class FusionMLP(nn.Module):
    def __init__(self, image_dim: int, clinical_dim: int) -> None:
        super().__init__()
        self.image_branch = nn.Sequential(nn.Linear(image_dim, 256), nn.ReLU())
        self.clinical_branch = nn.Sequential(nn.Linear(clinical_dim, 32), nn.ReLU())
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, image_embedding: torch.Tensor, clinical_features: torch.Tensor) -> torch.Tensor:
        image_hidden = self.image_branch(image_embedding)
        clinical_hidden = self.clinical_branch(clinical_features)
        fused = torch.cat([image_hidden, clinical_hidden], dim=1)
        return self.fusion_head(fused).squeeze(-1)


class FinalMultimodalModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        clinical_dim: int,
        hidden_dim: int = 768,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, feature_dim = build_backbone(backbone_name)
        self.projector = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.mil = AttentionMIL(hidden_dim)
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def encode_patient(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = normalize_backbone_output(self.encoder(images), self.projector.in_features)
        projected = self.projector(embeddings)
        contextual = self.transformer(projected.unsqueeze(0)).squeeze(0)
        pooled, weights = self.mil(contextual)
        return pooled, weights

    def forward(
        self, patient_images: list[torch.Tensor], clinical_features: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        patient_embeddings = []
        attention_weights = []
        for images in patient_images:
            pooled, weights = self.encode_patient(images)
            patient_embeddings.append(pooled)
            attention_weights.append(weights)
        image_repr = torch.stack(patient_embeddings, dim=0)
        clinical_repr = self.clinical_branch(clinical_features)
        logits = self.fusion_head(torch.cat([image_repr, clinical_repr], dim=1)).squeeze(-1)
        return logits, attention_weights, image_repr


class WeakSupervisedReasoningModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        clinical_dim: int,
        hidden_dim: int = 768,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        local_top_k: int = 2,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.local_top_k = max(int(local_top_k), 1)
        self.use_cls_token = use_cls_token
        self.encoder, feature_dim = build_backbone(backbone_name)
        self.projector = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)
        self.local_classifier = nn.Linear(hidden_dim, 1)
        self.attention_head = nn.Linear(hidden_dim, 1)
        self.global_classifier = nn.Linear(hidden_dim, 1)
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def encode_patient(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = normalize_backbone_output(self.encoder(images), self.projector.in_features)
        tokens = self.projector(embeddings).unsqueeze(0)
        if self.use_cls_token:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        contextual = self.transformer(tokens).squeeze(0)
        if self.use_cls_token:
            global_repr = contextual[0]
            local_tokens = contextual[1:]
        else:
            local_tokens = contextual
            global_repr = local_tokens.mean(dim=0)
        local_logits = self.local_classifier(local_tokens).squeeze(-1)
        attention_scores = self.attention_head(local_tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=0)
        local_repr = torch.sum(local_tokens * attention_weights.unsqueeze(-1), dim=0)
        global_logit = self.global_classifier(global_repr).squeeze(-1)
        return {
            "global_repr": global_repr,
            "local_repr": local_repr,
            "global_logit": global_logit,
            "local_logits": local_logits,
            "attention_weights": attention_weights,
        }

    def forward(self, patient_images: list[torch.Tensor], clinical_features: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        global_reprs = []
        local_reprs = []
        global_logits = []
        local_logits_all = []
        attention_weights_all = []
        for images in patient_images:
            encoded = self.encode_patient(images)
            global_reprs.append(encoded["global_repr"])
            local_reprs.append(encoded["local_repr"])
            global_logits.append(encoded["global_logit"])
            local_logits_all.append(encoded["local_logits"])
            attention_weights_all.append(encoded["attention_weights"])
        global_repr = torch.stack(global_reprs, dim=0)
        local_repr = torch.stack(local_reprs, dim=0)
        global_logit = torch.stack(global_logits, dim=0)
        clinical_repr = self.clinical_branch(clinical_features)
        final_logit = self.fusion_head(torch.cat([global_repr, local_repr, clinical_repr], dim=1)).squeeze(-1)
        return {
            "final_logit": final_logit,
            "global_logit": global_logit,
            "local_logits": local_logits_all,
            "attention_weights": attention_weights_all,
            "global_repr": global_repr,
            "local_repr": local_repr,
            "clinical_repr": clinical_repr,
        }
