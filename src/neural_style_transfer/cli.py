"""Command line entry point: ``nst --style STYLE CONTENT [CONTENT ...]``."""

from __future__ import annotations

import argparse
import logging
import os
import time

from .pipeline import run_style_transfer
from .transfer import StyleConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nst",
        description="Distributed neural style transfer (Gatys et al.) with PyTorch and Ray Data.",
    )
    parser.add_argument("content", nargs="+", help="content images and/or directories of images")
    parser.add_argument("-s", "--style", required=True, help="style image, e.g. samples/style/starry_night.jpg")
    parser.add_argument("-o", "--output-dir", default="outputs")
    parser.add_argument(
        "--size", type=int, default=StyleConfig.size, help="longest side in pixels (default: %(default)s)"
    )
    parser.add_argument(
        "--steps", type=int, default=StyleConfig.steps, help="L-BFGS iterations per image (default: %(default)s)"
    )
    parser.add_argument("--style-weight", type=float, default=StyleConfig.style_weight)
    parser.add_argument("--content-weight", type=float, default=StyleConfig.content_weight)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--workers", type=int, help="number of Ray actors (default: one per accelerator, or 1 on CPU)")
    parser.add_argument(
        "--accelerator-fraction",
        type=float,
        default=1.0,
        help="share of a GPU each worker reserves; 0.5 puts two workers on each GPU (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = StyleConfig(
        size=args.size, steps=args.steps, content_weight=args.content_weight, style_weight=args.style_weight
    )
    start = time.perf_counter()
    try:
        records = run_style_transfer(
            args.content,
            args.style,
            output_dir=args.output_dir,
            config=config,
            device=args.device,
            workers=args.workers,
            accelerator_fraction=args.accelerator_fraction,
        )
    except ValueError as exc:
        parser.error(str(exc))
    elapsed = time.perf_counter() - start

    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n{'image':<40} {'device':<6} {'seconds':>8}  output")
    for r in records:
        name = r["path"].rsplit("/", 1)[-1]
        detail = os.path.relpath(r["output_path"]) if r["status"] == "ok" else r["status"]
        print(f"{name:<40} {r['device']:<6} {r['seconds']:>8.1f}  {detail}")
    print(
        f"\n{len(ok)}/{len(records)} images in {elapsed:.1f}s wall clock "
        f"({60 * len(ok) / elapsed:.1f} images/min, including worker start-up)"
    )
    if len(ok) != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
