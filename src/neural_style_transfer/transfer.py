"""Single-image style transfer: optimise the pixels of one image with L-BFGS."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .device import get_best_device, synchronize
from .model import CONTENT_LAYERS, STYLE_LAYERS, VGGFeatures, gram_matrix

ImageSource = Union[str, Path, bytes, Image.Image]


@dataclass(frozen=True)
class StyleConfig:
    size: int = 512  # longest side of the output, in pixels
    steps: int = 300  # L-BFGS iterations per image
    content_weight: float = 1.0
    style_weight: float = 1e6


def load_image(source: ImageSource, max_size: int) -> torch.Tensor:
    """Load an image as a ``[1, 3, H, W]`` tensor in ``[0, 1]``.

    The longest side is scaled to ``max_size`` and the aspect ratio is kept.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, bytes):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(source)
    image = image.convert("RGB")

    scale = max_size / max(image.size)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    return TF.to_tensor(image).unsqueeze(0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``[1, 3, H, W]`` tensor in ``[0, 1]`` back to a PIL image."""
    return TF.to_pil_image(tensor.detach().squeeze(0).clamp(0, 1).cpu())


@dataclass
class StylizeResult:
    image: Image.Image
    seconds: float
    steps: int
    evaluations: int  # forward+backward passes through VGG19
    loss_history: list[float]


class StyleTransfer:
    """VGG19 plus the precomputed Gram matrices of one style image.

    Building this is the expensive part (loading VGG19 and moving it to the
    accelerator), so construct it once and call :meth:`stylize` for each content image.
    """

    def __init__(
        self,
        style_image: ImageSource,
        config: StyleConfig = StyleConfig(),
        device: str | torch.device | None = None,
        pretrained: bool = True,
    ):
        self.config = config
        self.device = device if isinstance(device, torch.device) else get_best_device(device)
        self.features = VGGFeatures(STYLE_LAYERS + CONTENT_LAYERS, pretrained=pretrained).to(self.device)

        style = load_image(style_image, config.size).to(self.device)
        with torch.no_grad():
            feats = self.features(style)
            self.style_grams = {name: gram_matrix(feats[name]) for name in STYLE_LAYERS}

    def stylize(self, content_image: ImageSource) -> StylizeResult:
        cfg = self.config
        start = time.perf_counter()

        content = load_image(content_image, cfg.size).to(self.device)
        with torch.no_grad():
            feats = self.features(content)
            content_targets = {name: feats[name] for name in CONTENT_LAYERS}

        # Start from the content image; this converges much faster than from noise.
        image = content.clone().requires_grad_(True)
        # One .step() runs up to `steps` L-BFGS iterations (the default max_iter=20
        # would silently multiply the work when step() is called in a loop).
        optimizer = torch.optim.LBFGS([image], max_iter=cfg.steps)
        losses: list[torch.Tensor] = []

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            feats = self.features(image)
            content_loss = sum(F.mse_loss(feats[n], content_targets[n]) for n in CONTENT_LAYERS)
            style_loss = sum(F.mse_loss(gram_matrix(feats[n]), self.style_grams[n]) for n in STYLE_LAYERS)
            loss = cfg.content_weight * content_loss + cfg.style_weight * style_loss
            loss.backward()
            losses.append(loss.detach())
            return loss.detach()  # L-BFGS reads gradients from image.grad; it only needs the value

        optimizer.step(closure)
        steps = optimizer.state[optimizer._params[0]]["n_iter"]

        result = to_pil(image)  # copies to host, which also waits for the device
        synchronize(self.device)
        return StylizeResult(
            image=result,
            seconds=time.perf_counter() - start,
            steps=steps,
            evaluations=len(losses),
            loss_history=torch.stack(losses).cpu().tolist(),
        )
