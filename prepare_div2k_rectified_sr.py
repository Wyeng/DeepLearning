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
    if path is None:
        return []
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_by_factor(image: np.ndarray, factor: float) -> np.ndarray:
    h, w = image.shape[:2]
    out_w = max(1, int(round(w / factor)))
    out_h = max(1, int(round(h / factor)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)


def scaled_size(size: tuple[int, int], factor: float) -> tuple[int, int]:
    w, h = size
    return max(1, int(round(w / factor))), max(1, int(round(h / factor)))


def rectify_with_unet(
    image: np.ndarray,
    predictor: UNetCornerPredictor,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    quad_raw, confidence, peaks = predictor.predict(image)
    quad = order_quad_points(quad_raw)
    h_matrix = homography_from_quad(quad, *target_size)
    rectified = warp_perspective(image, h_matrix, target_size)
    meta = {
        "target_size": [int(target_size[0]), int(target_size[1])],
        "predicted_quad": quad.astype(float).tolist(),
        "confidence": float(confidence),
        "corner_peak_scores": [float(v) for v in peaks],
        "homography": matrix_to_list(h_matrix),
    }
    return rectified, meta


def rectify_with_quad(
    image: np.ndarray,
    source_quad: np.ndarray,
    target_size: tuple[int, int],
    method: str,
) -> tuple[np.ndarray, dict]:
    h_matrix = homography_from_quad(source_quad, *target_size)
    rectified = warp_perspective(image, h_matrix, target_size)
    meta = {
        "method": method,
        "target_size": [int(target_size[0]), int(target_size[1])],
        "source_quad": np.asarray(source_quad, dtype=np.float32).astype(float).tolist(),
        "homography": matrix_to_list(h_matrix),
    }
    return rectified, meta


def process_split(
    split: str,
    input_dir: Path,
    output_root: Path,
    predictor: UNetCornerPredictor | None,
    args: argparse.Namespace,
    seed_offset: int,
) -> None:
    images = iter_images(input_dir)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images found for split={split}: {input_dir}")

    split_root = output_root / split
    dirs = {
        "gt": split_root / "gt",
        "lq": split_root / "lq",
        "incline_hr": split_root / "incline_hr",
        "incline_lq": split_root / "incline_lq",
        "meta": split_root / "meta",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)

    for index, image_path in enumerate(images, start=1):
        stem = image_path.stem
        gt_path = dirs["gt"] / f"{stem}.png"
        lq_path = dirs["lq"] / f"{stem}.png"
        incline_lq_path = dirs["incline_lq"] / f"{stem}.png"
        incline_hr_path = dirs["incline_hr"] / f"{stem}.png"
        if not args.overwrite and gt_path.exists() and lq_path.exists() and incline_lq_path.exists():
            print(f"[skip] {split}/{stem}: outputs already exist")
            continue

        clean = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if clean is None:
            print(f"[skip] Could not read {image_path}")
            continue

        clean_h, clean_w = clean.shape[:2]
        canvas_w = int(args.canvas_width) if args.canvas_width > 0 else clean_w
        canvas_h = int(args.canvas_height) if args.canvas_height > 0 else clean_h

        inclined_hr, synth_meta = make_sample(
            clean,
            canvas_w,
            canvas_h,
            rng,
            downscale_factor=1.0,
            apply_camera_degradation=not args.no_camera_degradation,
        )
        hr_target_size = tuple(int(v) for v in synth_meta["target_size"])
        if args.rectify_mode == "synthetic":
            source_quad_hr = np.asarray(synth_meta["source_quad"], dtype=np.float32)
            gt_rectified, gt_meta = rectify_with_quad(
                inclined_hr,
                source_quad_hr,
                hr_target_size,
                method="synthetic_homography",
            )
        else:
            if predictor is None:
                raise RuntimeError("U-Net predictor is required when --rectify-mode=unet")
            gt_rectified, gt_meta = rectify_with_unet(inclined_hr, predictor, hr_target_size)

        inclined_lq = resize_by_factor(inclined_hr, args.scale_factor)
        lq_target_size = scaled_size(hr_target_size, args.scale_factor)
        if args.rectify_mode == "synthetic":
            scale_xy = np.array(
                [
                    inclined_lq.shape[1] / inclined_hr.shape[1],
                    inclined_lq.shape[0] / inclined_hr.shape[0],
                ],
                dtype=np.float32,
            )
            source_quad_lq = source_quad_hr * scale_xy[None, :]
            lq_rectified, lq_meta = rectify_with_quad(
                inclined_lq,
                source_quad_lq,
                lq_target_size,
                method="synthetic_homography",
            )
        else:
            if predictor is None:
                raise RuntimeError("U-Net predictor is required when --rectify-mode=unet")
            lq_rectified, lq_meta = rectify_with_unet(inclined_lq, predictor, lq_target_size)

        cv2.imwrite(str(gt_path), gt_rectified)
        cv2.imwrite(str(lq_path), lq_rectified)
        cv2.imwrite(str(incline_lq_path), inclined_lq)
        if args.save_incline:
            cv2.imwrite(str(incline_hr_path), inclined_hr)

        metadata = {
            "input_hr": str(image_path),
            "gt": str(gt_path),
            "lq": str(lq_path),
            "incline_lq": str(incline_lq_path),
            "incline_hr": str(incline_hr_path) if args.save_incline else None,
            "scale_factor": float(args.scale_factor),
            "canvas_size": [int(canvas_w), int(canvas_h)],
            "synthetic": synth_meta,
            "gt_rectify": gt_meta,
            "lq_rectify": lq_meta,
        }
        (dirs["meta"] / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[{split} {index}/{len(images)}] {image_path.name} "
            f"GT={gt_rectified.shape[1]}x{gt_rectified.shape[0]} "
            f"LQ={lq_rectified.shape[1]}x{lq_rectified.shape[0]} "
            f"rectify={args.rectify_mode} "
            f"conf(gt/lq)={gt_meta.get('confidence', 1.0):.3f}/{lq_meta.get('confidence', 1.0):.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare paired DIV2K data for 4x SR: GT is HR after incline+U-Net rectification; "
            "LQ is the same inclined image downscaled by 4 and then U-Net rectified."
        )
    )
    parser.add_argument("--train-hr", type=Path, required=True, help="DIV2K train HR image directory.")
    parser.add_argument("--val-hr", type=Path, required=True, help="DIV2K valid HR image directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output paired dataset root.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained U-Net corner checkpoint.")
    parser.add_argument(
        "--rectify-mode",
        choices=["synthetic", "unet"],
        default="synthetic",
        help="Use synthetic homography labels for generated data, or U-Net corner prediction.",
    )
    parser.add_argument("--scale-factor", type=float, default=4.0, help="SR scale factor.")
    parser.add_argument("--canvas-width", type=int, default=0, help="0 keeps each image's original width.")
    parser.add_argument("--canvas-height", type=int, default=0, help="0 keeps each image's original height.")
    parser.add_argument("--input-size", type=int, default=224, help="U-Net input size.")
    parser.add_argument("--base-channels", type=int, default=None, help="Override U-Net base channels.")
    parser.add_argument("--device", type=str, default="cuda", help="U-Net device.")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic inclination seed.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke-test limit per split.")
    parser.add_argument("--no-camera-degradation", action="store_true", help="Disable synthetic blur/noise/illumination.")
    parser.add_argument("--save-incline", action="store_true", help="Also save intermediate inclined HR images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing paired images.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = None
    if args.rectify_mode == "unet":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required when --rectify-mode=unet")
        predictor = UNetCornerPredictor(
            args.checkpoint,
            input_size=args.input_size,
            device=args.device,
            base_channels=args.base_channels,
        )
    process_split("train", args.train_hr, args.output_root, predictor, args, seed_offset=0)
    process_split("val", args.val_hr, args.output_root, predictor, args, seed_offset=100000)


if __name__ == "__main__":
    main()
