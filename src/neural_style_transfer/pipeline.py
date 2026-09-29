"""Distributed batch style transfer with Ray Data.

Every content image is an independent optimisation problem (hundreds of VGG19
forward/backward passes), so the batch is embarrassingly parallel. The pipeline is:

    read_binary_files (CPU tasks)  ->  pool of StyleTransferWorker actors (one per accelerator slot)

Actors, not tasks, do the heavy lifting so VGG19 is loaded and moved to the
accelerator once per worker instead of once per image.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch

from .transfer import StyleConfig, StyleTransfer

logger = logging.getLogger(__name__)

# Ray schedules `num_gpus` against CUDA (and ROCm) devices only, so an Apple GPU is
# invisible to it. We advertise it as a custom resource instead, which lets Ray cap
# how many workers share the one Apple GPU exactly as it would for a CUDA device.
MPS_RESOURCE = "mps"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def init_ray(**kwargs: Any) -> None:
    """Start (or connect to) Ray, registering the Apple GPU as a custom resource."""
    if ray.is_initialized():
        return
    if "address" not in kwargs and torch.backends.mps.is_available():
        kwargs.setdefault("resources", {})[MPS_RESOURCE] = 1
    ray.init(**kwargs)


def detect_cluster_device(resources: dict[str, float]) -> str:
    """Pick the accelerator from what the *cluster* has, not from the driver process."""
    if resources.get("GPU", 0) > 0:
        return "cuda"
    if resources.get(MPS_RESOURCE, 0) > 0:
        return "mps"
    return "cpu"


@dataclass
class ResourcePlan:
    """How many actors to start and what each one reserves from Ray."""

    workers: int
    num_cpus: float
    num_gpus: float = 0
    resources: dict[str, float] = field(default_factory=dict)
    torch_threads: int | None = None

    def map_batches_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"concurrency": self.workers, "num_cpus": self.num_cpus}
        if self.num_gpus:
            kwargs["num_gpus"] = self.num_gpus
        if self.resources:
            kwargs["resources"] = self.resources
        return kwargs


def plan_resources(
    device: str,
    cluster: dict[str, float],
    workers: int | None = None,
    accelerator_fraction: float = 1.0,
) -> ResourcePlan:
    """Translate "N workers, each using this share of an accelerator" into Ray requests.

    ``accelerator_fraction=0.5`` packs two workers onto each GPU, which helps when one
    image is too small to saturate the device.
    """
    if not 0 < accelerator_fraction <= 1:
        raise ValueError("accelerator_fraction must be in (0, 1]")

    if device in ("cuda", "mps"):
        key = "GPU" if device == "cuda" else MPS_RESOURCE
        available = cluster.get(key, 0)
        if available <= 0:
            raise ValueError(f"Device {device!r} requested but the Ray cluster has no {key!r} resource")
        capacity = int(available / accelerator_fraction + 1e-9)
        workers = workers or capacity
        if workers > capacity:
            raise ValueError(
                f"{workers} workers x {accelerator_fraction} {key} needs "
                f"{workers * accelerator_fraction:g} but the cluster has {available:g}"
            )
        if device == "cuda":
            return ResourcePlan(workers=workers, num_cpus=1, num_gpus=accelerator_fraction)
        return ResourcePlan(workers=workers, num_cpus=1, resources={MPS_RESOURCE: accelerator_fraction})

    # CPU: split the cores between workers and pin PyTorch's thread pool to that share,
    # otherwise every worker spawns one thread per core and they fight each other.
    # One core stays free for Ray Data's read tasks and the driver.
    usable = max(1, int(cluster.get("CPU", 1)) - 1)
    workers = workers or 1
    if workers > usable:
        raise ValueError(f"{workers} CPU workers requested but only {usable} cores are usable")
    threads = usable // workers
    return ResourcePlan(workers=workers, num_cpus=threads, torch_threads=threads)


class StyleTransferWorker:
    """Ray Data actor: holds VGG19 and the style Grams, stylises one image per call."""

    def __init__(
        self,
        style_image: bytes,
        config: StyleConfig,
        output_dir: str,
        device: str,
        pretrained: bool = True,
        torch_threads: int | None = None,
    ):
        if torch_threads:
            torch.set_num_threads(torch_threads)
        self.engine = StyleTransfer(style_image, config, device=device, pretrained=pretrained)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rows = []
        for data, path in zip(batch["bytes"], batch["path"]):
            row = {"path": path, "output_path": "", "status": "ok", "seconds": 0.0, "steps": 0, "final_loss": np.nan}
            try:
                result = self.engine.stylize(bytes(data))
                output = self.output_dir / f"{Path(path).stem}_stylized.png"
                result.image.save(output)
                row.update(
                    output_path=str(output),
                    seconds=result.seconds,
                    steps=result.steps,
                    final_loss=result.loss_history[-1],
                )
            except Exception as exc:  # one bad image must not take down the whole batch
                logger.exception("Failed to stylise %s", path)
                row["status"] = f"error: {type(exc).__name__}: {exc}"
            row["device"] = str(self.engine.device)
            row["worker"] = self.worker_id
            rows.append(row)
        return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def find_images(inputs: list[str | Path]) -> list[str]:
    """Expand directories into the image files they contain."""
    paths: list[str] = []
    for item in map(Path, inputs):
        if item.is_dir():
            paths.extend(str(p) for p in sorted(item.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            paths.append(str(item))
    return paths


def run_style_transfer(
    content: list[str | Path],
    style_image: str | Path,
    output_dir: str | Path = "outputs",
    config: StyleConfig = StyleConfig(),
    device: str | None = None,
    workers: int | None = None,
    accelerator_fraction: float = 1.0,
    pretrained: bool = True,
) -> list[dict[str, Any]]:
    """Stylise every content image with a pool of Ray actors.

    Returns one record per input (sorted by path) with the output file, timing and
    which worker processed it.
    """
    paths = find_images(content)
    if not paths:
        raise ValueError("No content images found")
    stems = [Path(p).stem for p in paths]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        raise ValueError(f"Outputs are named after the input file, so these names clash: {duplicates}")

    init_ray()
    cluster = ray.cluster_resources()
    if device in (None, "auto"):
        device = detect_cluster_device(cluster)
    plan = plan_resources(device, cluster, workers, accelerator_fraction)
    logger.info("Device %s, %s", device, plan)

    # The style image travels with the actor constructor, so workers on other
    # nodes do not need access to the driver's filesystem.
    style_bytes = Path(style_image).read_bytes()

    dataset = ray.data.read_binary_files(paths, include_paths=True, override_num_blocks=len(paths))
    stylized = dataset.map_batches(
        StyleTransferWorker,
        fn_constructor_kwargs={
            "style_image": style_bytes,
            "config": config,
            "output_dir": str(Path(output_dir).resolve()),
            "device": device,
            "pretrained": pretrained,
            "torch_threads": plan.torch_threads,
        },
        batch_size=1,
        **plan.map_batches_kwargs(),
    )
    # By default Ray Data queues up to 4 batches on each actor. One image is tens of
    # seconds of work, so queueing lets the first actor to start grab the whole batch
    # while the others sit idle. One in-flight image per actor keeps them all busy.
    stylized.context.max_tasks_in_flight_per_actor = 1
    records = [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()} for row in stylized.take_all()
    ]
    return sorted(records, key=lambda r: r["path"])
