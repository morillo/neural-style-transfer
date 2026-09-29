"""Throughput benchmark: images/minute for different devices and worker layouts.

    python scripts/benchmark.py                      # sensible defaults for this machine
    python scripts/benchmark.py --configs cuda:1 cuda:2@0.5 cuda:4@0.25

Each config is ``device:workers[@accelerator_fraction]``. Results are appended to
benchmarks/<hardware>.md (a Markdown table) and benchmarks/<hardware>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import ray
import torch

from neural_style_transfer.pipeline import MPS_RESOURCE, init_ray, run_style_transfer
from neural_style_transfer.transfer import StyleConfig

ROOT = Path(__file__).resolve().parent.parent


def default_configs(cluster: dict[str, float]) -> list[str]:
    configs = ["cpu:1", "cpu:2", "cpu:4"]
    if cluster.get(MPS_RESOURCE):
        configs += ["mps:1", "mps:2@0.5"]
    gpus = int(cluster.get("GPU", 0))
    if gpus:
        configs += [f"cuda:{gpus}", f"cuda:{2 * gpus}@0.5", f"cuda:{4 * gpus}@0.25"]
    return configs


def parse_config(spec: str) -> tuple[str, int, float]:
    device, _, rest = spec.partition(":")
    workers, _, fraction = rest.partition("@")
    return device, int(workers or 1), float(fraction or 1.0)


def hardware_name() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        return chip.strip() or platform.machine()
    return platform.processor() or platform.machine()


def make_workload(n: int, directory: Path) -> list[str]:
    """Copy the sample images round-robin until there are ``n`` distinct files."""
    samples = sorted((ROOT / "samples" / "content").glob("*.jpg"))
    paths = []
    for i, src in zip(range(n), itertools.cycle(samples)):
        dst = directory / f"{i:03d}_{src.name}"
        shutil.copy(src, dst)
        paths.append(str(dst))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", help="e.g. cpu:1 mps:2@0.5 cuda:4@0.5")
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--style", default=str(ROOT / "samples" / "style" / "starry_night.jpg"))
    parser.add_argument("--results-dir", default=str(ROOT / "benchmarks"))
    args = parser.parse_args()

    init_ray()
    cluster = ray.cluster_resources()
    configs = args.configs or default_configs(cluster)
    config = StyleConfig(size=args.size, steps=args.steps)
    hardware = hardware_name()
    print(f"{hardware} | torch {torch.__version__} | ray {ray.__version__} | {configs}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in").mkdir()
        content = make_workload(args.images, tmp / "in")
        # Warm-up: downloads the VGG19 weights once so no run pays for it.
        run_style_transfer(content[:1], args.style, tmp / "warmup", StyleConfig(size=64, steps=1), device="cpu")

        for spec in configs:
            device, workers, fraction = parse_config(spec)
            start = time.perf_counter()
            records = run_style_transfer(
                content, args.style, tmp / spec.replace(":", "_"), config, device, workers, fraction
            )
            wall = time.perf_counter() - start
            failed = [r for r in records if r["status"] != "ok"]
            if failed:
                raise RuntimeError(f"{spec}: {len(failed)} images failed, e.g. {failed[0]['status']}")
            row = {
                "config": spec,
                "device": device,
                "workers": workers,
                "accelerator_fraction": fraction,
                "images": len(records),
                "wall_seconds": round(wall, 1),
                "images_per_minute": round(60 * len(records) / wall, 2),
                "median_seconds_per_image": round(statistics.median(r["seconds"] for r in records), 1),
                "distinct_workers": len({r["worker"] for r in records}),
            }
            rows.append(row)
            print(json.dumps(row))

    baseline = rows[0]["images_per_minute"]
    header = (
        f"### {hardware}\n\n"
        f"{args.images} images, {args.size}px longest side, {args.steps} L-BFGS steps each. "
        f"torch {torch.__version__}, ray {ray.__version__}. "
        "Wall clock includes starting the Ray actors and loading VGG19.\n\n"
        "| Config | Workers | Share of accelerator per worker | Wall clock (s) | Images/min "
        "| Speed-up | Median s/image (per worker) |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        f"| `{r['config']}` | {r['workers']} | {r['accelerator_fraction'] if r['device'] != 'cpu' else '-'} "
        f"| {r['wall_seconds']} | {r['images_per_minute']} | {r['images_per_minute'] / baseline:.1f}x "
        f"| {r['median_seconds_per_image']} |"
        for r in rows
    ]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    name = re.sub(r"[^a-z0-9]+", "-", hardware.lower()).strip("-")
    (results_dir / f"{name}.md").write_text(header + "\n".join(lines) + "\n")
    (results_dir / f"{name}.json").write_text(
        json.dumps({"hardware": hardware, "size": args.size, "steps": args.steps, "results": rows}, indent=2) + "\n"
    )
    print("\n" + header + "\n".join(lines))


if __name__ == "__main__":
    main()
