import torch
import torch.nn as nn
from torchvision import models


def build_model(cfg):
    """
    Builds a ResNet-50 with ImageNet weights and replaces the head for CUB-200.
    When cfg['model']['freeze_backbone'] is True all parameters except the
    new classification head are frozen (linear-probe mode).
    """
    num_classes = cfg["dataset"]["num_classes"]
    freeze      = cfg["model"]["freeze_backbone"]

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model   = models.resnet50(weights=weights)

    # ── freeze backbone (linear-probe mode) ──
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # ── replace head: 2048 → num_classes ──
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_trainable_params(model):
    """Returns only parameters that require gradients."""
    return [p for p in model.parameters() if p.requires_grad]
