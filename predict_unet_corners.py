from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict SCI corners with a trained U-Net model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target-aspect", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from unet_rectify.geometry import (
        draw_quad,
        estimate_target_size,
        homography_from_quad,
        matrix_to_list,
        parse_aspect,
        warp_perspective,
    )
    from unet_rectify.learning.model_predictor import UNetCornerPredictor

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")

    predictor = UNetCornerPredictor(args.checkpoint, input_size=args.input_size, device=args.device)
    quad, confidence, peaks = predictor.predict(image)
    target_size = estimate_target_size(quad, target_aspect=parse_aspect(args.target_aspect))
    h_matrix = homography_from_quad(quad, *target_size)
    rectified = warp_perspective(image, h_matrix, target_size)

    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / f"{args.image.stem}_quad.png"), draw_quad(image, quad))
    cv2.imwrite(str(args.output / f"{args.image.stem}_rectified.png"), rectified)
    metadata = {
        "quad": quad.astype(float).tolist(),
        "target_size": [int(target_size[0]), int(target_size[1])],
        "homography": matrix_to_list(h_matrix),
        "confidence": float(confidence),
        "corner_peak_scores": peaks,
    }
    (args.output / f"{args.image.stem}_pred_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
