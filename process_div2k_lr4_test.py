from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_synthetic_sci import IMAGE_EXTENSIONS, make_sample
from unet_rectify.geometry import homography_from_quad, matrix_to_list, order_quad_points, warp_perspective
from unet_rectify.learning.model_predictor import UNetCornerPredictor


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DIV2K X4 inclined LR inputs from HR images and rectify them "
            "with the trained U-Net corner model."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="DIV2K valid HR folder.")
    parser.add_argument("--output-lr", type=Path, required=True, help="Folder for rectified LR images.")
    parser.add_argument("--output-incline", type=Path, required=True, help="Folder for inclined LR input images.")
    parser.add_argument("--metadata-dir", type=Path, default=None, help="Optional folder for per-image metadata.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained U-Net corner checkpoint.")
    parser.add_argument("--downscale-factor", type=float, default=4.0, help="LR degradation factor.")
    parser.add_argument("--input-size", type=int, default=224, help="U-Net input size.")
    parser.add_argument("--base-channels", type=int, default=None, help="Override U-Net base channels.")
    parser.add_argument("--device", type=str, default="cuda", help="U-Net device.")
    parser.add_argument("--seed", type=int, default=99, help="Seed for deterministic test inclination.")
    parser.add_argument("--no-camera-degradation", action="store_true", help="Disable blur/noise/illumination degradation.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = iter_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    args.output_lr.mkdir(parents=True, exist_ok=True)
    args.output_incline.mkdir(parents=True, exist_ok=True)
    if args.metadata_dir is not None:
        args.metadata_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    predictor = UNetCornerPredictor(
        args.checkpoint,
        input_size=args.input_size,
        device=args.device,
        base_channels=args.base_channels,
    )

    for index, image_path in enumerate(images, start=1):
        clean = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if clean is None:
            print(f"[skip] Could not read {image_path}")
            continue

        clean_h, clean_w = clean.shape[:2]
        stem = image_path.stem
        incline_path = args.output_incline / f"{stem}.png"
        lr_path = args.output_lr / f"{stem}.png"
        metadata_path = args.metadata_dir / f"{stem}.json" if args.metadata_dir is not None else None
        if not args.overwrite and incline_path.exists() and lr_path.exists():
            print(f"[skip] {stem}: outputs already exist")
            continue

        inclined, synth_meta = make_sample(
            clean,
            clean_w,
            clean_h,
            rng,
            downscale_factor=args.downscale_factor,
            apply_camera_degradation=not args.no_camera_degradation,
        )
        cv2.imwrite(str(incline_path), inclined)

        quad_raw, confidence, peaks = predictor.predict(inclined)
        quad = order_quad_points(quad_raw)
        target_size = tuple(int(v) for v in synth_meta["target_size"])
        h_matrix = homography_from_quad(quad, *target_size)
        rectified = warp_perspective(inclined, h_matrix, target_size)
        cv2.imwrite(str(lr_path), rectified)

        if metadata_path is not None:
            metadata = {
                "input_hr": str(image_path),
                "inclined_lr": str(incline_path),
                "rectified_lr": str(lr_path),
                "source_canvas_size": [int(clean_w), int(clean_h)],
                "target_size": [int(target_size[0]), int(target_size[1])],
                "predicted_quad": quad.astype(float).tolist(),
                "confidence": float(confidence),
                "corner_peak_scores": [float(v) for v in peaks],
                "homography": matrix_to_list(h_matrix),
                "synthetic": synth_meta,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[{index}/{len(images)}] {image_path.name} -> {lr_path.name} "
            f"inclined={inclined.shape[1]}x{inclined.shape[0]} "
            f"target={target_size[0]}x{target_size[1]} conf={confidence:.3f}"
        )


if __name__ == "__main__":
    main()
