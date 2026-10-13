"""Backbone classifier, staged unfreezing, the E6 fusion head and the Deep CORAL loss."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.dim, num_classes))

    def forward(self, x, return_features: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        return (logits, f) if return_features else logits


def _stages(backbone: nn.Module) -> list[nn.Module]:
    """Backbone pieces from input to output, so the last k are the 'top blocks'."""
    if hasattr(backbone, "blocks") and hasattr(backbone, "conv_head"):      # EfficientNet / MobileNet
        return [nn.ModuleList([backbone.conv_stem, backbone.bn1]), *backbone.blocks,
                nn.ModuleList([backbone.conv_head, backbone.bn2])]
    if hasattr(backbone, "layer4"):                                           # ResNet
        return [nn.ModuleList([backbone.conv1, backbone.bn1]), backbone.layer1, backbone.layer2,
                backbone.layer3, backbone.layer4]
    return [backbone]


def set_trainable(model: Classifier, unfreeze: int) -> None:
    """unfreeze = 0: head only; k > 0: head + top k stages (EfficientNet counts the
    final conv_head as one stage); -1: everything."""
    for p in model.backbone.parameters():
        p.requires_grad = unfreeze == -1
    if unfreeze > 0:
        for stage in _stages(model.backbone)[-unfreeze:]:
            for p in stage.parameters():
                p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


def train_mode(model: nn.Module) -> None:
    """model.train(), but BatchNorm layers with frozen weights stay in eval mode so their
    running statistics are not overwritten."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def param_groups(model: Classifier, lr_head: float, lr_backbone: float, weight_decay: float):
    groups = [{"params": [p for p in model.head.parameters()], "lr": lr_head}]
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    if bb and lr_backbone > 0:
        groups.append({"params": bb, "lr": lr_backbone})
    for g in groups:
        g["weight_decay"] = weight_decay
    return groups


def coral_loss(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Sun & Saenko (2016): squared Frobenius distance between feature covariances / 4d^2."""
    d = fs.size(1)
    cs = torch.cov(fs.float().T)
    ct = torch.cov(ft.float().T)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


class FusionMLP(nn.Module):
    """E6 head on [CNN embedding ‖ texture features] (or either alone)."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def target_layer(model: Classifier) -> nn.Module:
    """Last conv block, for Grad-CAM."""
    bb = model.backbone
    if hasattr(bb, "conv_head"):
        return bb.conv_head
    if hasattr(bb, "layer4"):
        return bb.layer4[-1]
    raise ValueError("unknown backbone for Grad-CAM")
