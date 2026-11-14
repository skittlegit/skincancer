"""SkinNet: our own convolutional network, trained from random weights. No pretrained
weights are used anywhere in the SkinNet classifier.

The one exception is the reference comparison of idea P (DECISIONS.md, section 19): a
torchvision ResNet-18, from ImageNet weights and from random weights, chosen with
`model.arch: resnet18` and `model.weights: imagenet`. It is never a checkpoint."""
from __future__ import annotations

import torch.nn as nn


def _conv_bn(cin: int, cout: int, k: int = 3, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout)]


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection. When the block halves the resolution or
    changes width, the skip is average-pool + 1x1 convolution."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.body = nn.Sequential(*_conv_bn(cin, cout, 3, stride), nn.ReLU(inplace=True), *_conv_bn(cout, cout))
        self.skip = None
        if stride != 1 or cin != cout:
            pool = [nn.AvgPool2d(stride)] if stride != 1 else []
            self.skip = nn.Sequential(*pool, *_conv_bn(cin, cout, 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + (x if self.skip is None else self.skip(x)))


class SkinNet(nn.Module):
    """Stem of three 3x3 convolutions and a max-pool, then four stages of residual blocks.
    Each stage after the first halves the resolution and doubles the width. Global average
    pooling gives the embedding. Defaults (width 32, two blocks per stage): 17 convolution
    layers on the main path, 2.8M parameters, a 256-d embedding, 224 px in -> 7x7 feature map."""

    def __init__(self, width: int = 32, depths=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(*_conv_bn(3, width // 2, 3, 2), nn.ReLU(inplace=True),
                                  *_conv_bn(width // 2, width // 2), nn.ReLU(inplace=True),
                                  *_conv_bn(width // 2, width), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        stages, cin = [], width
        for i, depth in enumerate(depths):
            cout = width * 2 ** i
            stages.append(nn.Sequential(*[ResBlock(cin if j == 0 else cout, cout, 2 if (j == 0 and i > 0) else 1)
                                          for j in range(depth)]))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.num_features = cin
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():   # each block starts as the identity, which steadies early training
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.body[-1].weight)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x.mean((2, 3))


class ResNet18(nn.Module):
    """torchvision ResNet-18 without its ImageNet classifier: 512-d embedding. Only for the
    idea P reference comparison. imagenet=True loads torchvision's IMAGENET1K_V1 weights."""

    def __init__(self, imagenet: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if imagenet else None)
        net.fc = nn.Identity()
        self.net = net
        self.num_features = 512

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    """Backbone (SkinNet unless arch says otherwise) + dropout + one linear layer."""

    def __init__(self, num_classes: int, width: int = 32, depths=(2, 2, 2, 2), drop: float = 0.3,
                 arch: str = "skinnet", imagenet: bool = False):
        super().__init__()
        if arch == "skinnet":
            if imagenet:
                raise ValueError("SkinNet has no ImageNet weights")
            self.backbone = SkinNet(width, depths)
        elif arch == "resnet18":
            self.backbone = ResNet18(imagenet)
        else:
            raise ValueError(f"unknown model.arch {arch!r}")
        self.dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.dim, num_classes))

    def forward(self, x, return_features: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        return (logits, f) if return_features else logits


def target_layer(model: Classifier) -> nn.Module:
    """Last residual block, for Grad-CAM."""
    if isinstance(model.backbone, ResNet18):
        return model.backbone.net.layer4[-1]
    return model.backbone.stages[-1][-1]
