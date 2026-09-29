"""VGG19 feature extractor and Gram matrices (Gatys et al., 2016)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torchvision import models

# Positions of the activations we need inside torchvision's ``vgg19().features``.
LAYER_INDEX = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu5_1": 29,
}
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
CONTENT_LAYERS = ("relu4_2",)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGFeatures(nn.Module):
    """Frozen VGG19 truncated after the deepest requested layer.

    Takes images in ``[0, 1]`` (ImageNet normalisation happens inside) and returns
    ``{layer_name: activation}`` for the requested layers.
    """

    def __init__(self, layers: Iterable[str] = STYLE_LAYERS + CONTENT_LAYERS, pretrained: bool = True):
        super().__init__()
        layers = tuple(layers)
        unknown = set(layers) - LAYER_INDEX.keys()
        if unknown:
            raise ValueError(f"Unknown VGG19 layers: {sorted(unknown)}")

        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights).features
        last = max(LAYER_INDEX[name] for name in layers)
        self.body = vgg[: last + 1]
        # torchvision's ReLUs are in-place, which would overwrite activations we keep.
        for module in self.body:
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.requires_grad_(False)
        self.eval()
        self._names = {LAYER_INDEX[name]: name for name in layers}
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i in self._names:
                out[self._names[i]] = x
        return out


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Channel-to-channel correlations, normalised by the number of elements.

    Normalising by ``C * H * W`` makes the style loss independent of image size,
    so the style image and content image do not need matching resolutions.
    """
    b, c, h, w = features.shape
    flat = features.reshape(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)
