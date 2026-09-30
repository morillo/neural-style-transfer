"""Single-image style transfer: optimise the pixels of one image with L-BFGS.

This is the method of Gatys et al., "Image Style Transfer Using Convolutional Neural
Networks" (CVPR 2016). Unlike most deep learning, nothing here is trained. The
network (VGG19, see ``model.py``) stays frozen, and the *image itself* is the thing
being optimised:

1. Run the style image through VGG19 once and record its Gram matrices (its
   "texture statistics") at five layers. These are the style targets.
2. Run the content photo through VGG19 once and record its activations at a deep
   layer. This is the content target: what is in the picture and where.
3. Start the output image as a copy of the content photo.
4. Repeatedly measure how far the output is from both targets (the *loss*), use
   backpropagation to get the gradient of that loss with respect to every pixel,
   and let the L-BFGS optimiser nudge the pixels to reduce it.
5. After a few hundred steps the output keeps the photo's layout but has the
   painting's colours and brush strokes.

Because each image is its own optimisation, one image costs hundreds of VGG19
forward and backward passes. That is why ``pipeline.py`` spreads images across
workers.
"""

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
"""Anything :func:`load_image` accepts: a file path, the file's raw bytes, or a PIL image."""


@dataclass(frozen=True)
class StyleConfig:
    """Settings for one style transfer run.

    Attributes:
        size: Longest side of the output image in pixels. Cost grows with the pixel
            count, so doubling ``size`` makes each image roughly 3-4x slower.
        steps: L-BFGS iterations per image. Each iteration is about one VGG19 forward
            and backward pass. Around 300 gives good results; fewer is faster but
            less stylised.
        content_weight: How strongly the output must keep the photo's structure.
        style_weight: How strongly the output must match the painting's textures.
            Only the ratio to ``content_weight`` matters: raise it (e.g. ``1e7``) for
            a bolder painterly look, lower it (e.g. ``1e5``) to stay closer to the photo.
    """

    size: int = 512
    steps: int = 300
    content_weight: float = 1.0
    style_weight: float = 1e6


def load_image(source: ImageSource, max_size: int) -> torch.Tensor:
    """Load an image and convert it to the tensor format the model expects.

    Args:
        source: A file path, the file's raw bytes, or a PIL image.
        max_size: The longest side of the result in pixels. The aspect ratio is kept.

    Returns:
        A float tensor of shape ``[1, 3, H, W]`` (batch of one, RGB channels, height,
        width) with pixel values in ``[0, 1]``.
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
    """Convert a ``[1, 3, H, W]`` tensor in ``[0, 1]`` back to a PIL image.

    Values outside ``[0, 1]`` (the optimiser can overshoot slightly) are clipped.
    """
    return TF.to_pil_image(tensor.detach().squeeze(0).clamp(0, 1).cpu())


@dataclass
class StylizeResult:
    """The output of :meth:`StyleTransfer.stylize`.

    Attributes:
        image: The stylised image.
        seconds: Wall-clock time for this image, including loading it.
        steps: L-BFGS iterations actually run (can be fewer than requested if the
            optimiser converges early).
        evaluations: VGG19 forward and backward passes. L-BFGS may evaluate more than
            once per step, so this is the true measure of compute used.
        loss_history: Total loss after each evaluation, for plotting convergence.
    """

    image: Image.Image
    seconds: float
    steps: int
    evaluations: int
    loss_history: list[float]


class StyleTransfer:
    """Stylises content images in the style of one painting.

    Construction is the expensive part: it loads VGG19 (548 MB), moves it to the
    accelerator and computes the style image's Gram matrices. Build one instance and
    call :meth:`stylize` for each content image. The Ray workers in ``pipeline.py``
    each hold one instance for exactly this reason.

    Args:
        style_image: The painting whose style to apply (path, bytes or PIL image).
        config: Image size, number of steps and loss weights.
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, a ``torch.device``, or ``None`` to
            pick the fastest available.
        pretrained: Use ImageNet-trained VGG19 weights. Only tests set this to
            ``False``, to avoid downloading the weights.

    Example:
        >>> engine = StyleTransfer("samples/style/starry_night.jpg")
        >>> result = engine.stylize("samples/content/victorian_house.jpg")
        >>> result.image.save("house.png")
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

        # Style targets never change, so compute them once. no_grad() skips the
        # bookkeeping PyTorch would otherwise keep for backpropagation.
        style = load_image(style_image, config.size).to(self.device)
        with torch.no_grad():
            feats = self.features(style)
            self.style_grams = {name: gram_matrix(feats[name]) for name in STYLE_LAYERS}

    def stylize(self, content_image: ImageSource) -> StylizeResult:
        """Apply the style to one content image.

        Args:
            content_image: The photo to stylise (path, bytes or PIL image).

        Returns:
            The stylised image plus timing and convergence details.
        """
        cfg = self.config
        start = time.perf_counter()

        content = load_image(content_image, cfg.size).to(self.device)
        with torch.no_grad():
            feats = self.features(content)
            content_targets = {name: feats[name] for name in CONTENT_LAYERS}

        # The pixels of `image` are the parameters being optimised, so it needs
        # gradients. Starting from the content photo converges much faster than
        # starting from random noise.
        image = content.clone().requires_grad_(True)
        # L-BFGS is a quasi-Newton optimiser: it estimates curvature from recent
        # gradients, which suits this smooth, full-batch problem better than Adam.
        # One .step() runs up to `steps` iterations (the default max_iter=20 would
        # silently multiply the work if step() were called in a loop).
        optimizer = torch.optim.LBFGS([image], max_iter=cfg.steps)
        losses: list[torch.Tensor] = []

        def closure() -> torch.Tensor:
            # L-BFGS calls this whenever it needs the loss and gradient at the current pixels.
            optimizer.zero_grad()
            feats = self.features(image)
            # Content loss: are the deep features (what is where) still those of the photo?
            content_loss = sum(F.mse_loss(feats[n], content_targets[n]) for n in CONTENT_LAYERS)
            # Style loss: do the texture statistics (Gram matrices) match the painting's, at every scale?
            style_loss = sum(F.mse_loss(gram_matrix(feats[n]), self.style_grams[n]) for n in STYLE_LAYERS)
            loss = cfg.content_weight * content_loss + cfg.style_weight * style_loss
            loss.backward()  # fills image.grad: how each pixel should change to lower the loss
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
