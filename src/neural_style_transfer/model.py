"""VGG19 feature extractor and Gram matrices (Gatys et al., 2016).

Why a network trained to *classify* photos can describe *style*
-----------------------------------------------------------------
VGG19 is a convolutional neural network trained on ImageNet to recognise 1,000
kinds of objects. To do that it learned a stack of filters: early layers respond to
simple things like edges, colours and small textures, and deeper layers respond to
larger patterns and object parts. Style transfer reuses those learned filters as a
ready-made way to *measure* images. Nothing is trained here.

- **Content** is read from a deep layer (``relu4_2``). Its activations say which
  large patterns appear and roughly where, but not their exact pixels.
- **Style** is read from five layers, shallow to deep (``relu1_1`` ... ``relu5_1``),
  through Gram matrices (see :func:`gram_matrix`). They record which features tend
  to fire together, for example "swirly strokes appear alongside dark blue", and
  throw away *where* they fire. That is texture: colours and brush strokes, not layout.

Layer names follow the paper: ``relu4_2`` is the ReLU after the 2nd convolution in
VGG19's 4th block.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torchvision import models

# Positions of the activations we need inside torchvision's ``vgg19().features``,
# which is a flat list of conv, ReLU and pooling layers.
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

# VGG19 was trained on images normalised with these per-channel statistics, so
# inputs must be normalised the same way for its filters to respond as intended.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VGGFeatures(nn.Module):
    """Frozen VGG19 that returns the activations of selected layers.

    The network is cut after the deepest requested layer (the classifier head and
    later layers are never needed) and its weights are frozen, because only the
    input image is optimised.

    Args:
        layers: Names from :data:`LAYER_INDEX` to return.
        pretrained: Load ImageNet weights (downloaded once, 548 MB, into the PyTorch
            cache). ``False`` gives random weights, which is only useful in tests.

    Raises:
        ValueError: If a layer name is not in :data:`LAYER_INDEX`.
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
        # Buffers move with the module on .to(device) but are not trainable parameters.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run images through the network and collect the requested activations.

        Args:
            x: Images of shape ``[B, 3, H, W]`` with values in ``[0, 1]``.

        Returns:
            ``{layer_name: activation}``, each activation of shape ``[B, C, h, w]``
            where ``C`` is that layer's number of filters (64 to 512).
        """
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i in self._names:
                out[self._names[i]] = x
        return out


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Summarise a layer's activations as channel-to-channel correlations.

    Each of the ``C`` channels is a map of where one filter fired. Flattening the maps
    and multiplying them together gives a ``C x C`` matrix whose entry ``(i, j)`` is
    large when filters ``i`` and ``j`` fire in the same places. Position is summed
    away, so the result describes texture rather than layout.

    Normalising by ``C * H * W`` makes the value independent of image size, so the
    style and content images do not need matching resolutions.

    Args:
        features: Activations of shape ``[B, C, H, W]``.

    Returns:
        Gram matrices of shape ``[B, C, C]``.
    """
    b, c, h, w = features.shape
    flat = features.reshape(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)
