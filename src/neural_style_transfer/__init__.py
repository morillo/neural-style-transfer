"""Neural style transfer with PyTorch, distributed across workers with Ray Data.

Modules, in the order a newcomer might read them:

- ``model``: VGG19 feature extraction and Gram matrices (how images are measured).
- ``transfer``: the per-image optimisation (how one image is stylised).
- ``pipeline``: Ray Data actors that stylise many images in parallel.
- ``device``: CUDA / Apple MPS / CPU selection.
- ``cli``: the ``nst`` command.

See ``docs/how-it-works.md`` for a guided walkthrough.
"""

from .device import get_best_device
from .model import VGGFeatures, gram_matrix
from .pipeline import run_style_transfer
from .transfer import StyleConfig, StyleTransfer, StylizeResult, load_image, to_pil

__version__ = "0.2.0"

__all__ = [
    "StyleConfig",
    "StyleTransfer",
    "StylizeResult",
    "VGGFeatures",
    "get_best_device",
    "gram_matrix",
    "load_image",
    "run_style_transfer",
    "to_pil",
]
