from __future__ import annotations

import torch.nn as nn
from torchvision import models


BACKBONES = [
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
]


def _load_with_fallback(builder, weights_enum_name: str | None):
    weights = None
    if weights_enum_name is not None:
        try:
            weights = getattr(models, weights_enum_name).DEFAULT
        except Exception:
            weights = None
    try:
        return builder(weights=weights)
    except Exception:
        return builder(weights=None)


def build_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    if backbone_name == "resnet18":
        model = _load_with_fallback(models.resnet18, "ResNet18_Weights")
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if backbone_name == "resnet50":
        model = _load_with_fallback(models.resnet50, "ResNet50_Weights")
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if backbone_name == "densenet121":
        model = _load_with_fallback(models.densenet121, "DenseNet121_Weights")
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if backbone_name == "efficientnet_b0":
        model = _load_with_fallback(models.efficientnet_b0, "EfficientNet_B0_Weights")
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if backbone_name == "mobilenet_v3_large":
        model = _load_with_fallback(models.mobilenet_v3_large, "MobileNet_V3_Large_Weights")
        feature_dim = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if backbone_name == "convnext_tiny":
        model = _load_with_fallback(models.convnext_tiny, "ConvNeXt_Tiny_Weights")
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if backbone_name == "vit_b_16":
        model = _load_with_fallback(models.vit_b_16, "ViT_B_16_Weights")
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        return model, feature_dim
    if backbone_name == "swin_t":
        model = _load_with_fallback(models.swin_t, "Swin_T_Weights")
        feature_dim = model.head.in_features
        model.head = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Unsupported backbone: {backbone_name}")
