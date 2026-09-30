"""Choosing where PyTorch runs: an NVIDIA GPU (CUDA), an Apple GPU (MPS) or the CPU."""

from __future__ import annotations

import torch


def get_best_device(preferred: str | None = None) -> torch.device:
    """Return the device to run on.

    Args:
        preferred: ``"cuda"``, ``"mps"``, ``"cpu"`` (or e.g. ``"cuda:1"``) to force a
            device. ``None`` or ``"auto"`` picks the fastest one available, in the
            order CUDA -> Apple MPS -> CPU.

    Returns:
        The ``torch.device`` to move models and tensors to.

    Raises:
        RuntimeError: If a GPU type was requested that this machine does not have.
    """
    if preferred and preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait until all queued work on ``device`` has finished.

    GPU calls return immediately and run in the background, so without this a timer
    could stop before the GPU is actually done.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
