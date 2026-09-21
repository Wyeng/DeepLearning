from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def random_quad(width: int, height: int, rng: random.Random) -> np.ndarray:
    margin_x = width * rng.uniform(0.08, 0.18)
    margin_y = height * rng.uniform(0.08, 0.18)
    base = np.array(
        [
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [width - margin_x, height - margin_y],
            [margin_x, height - margin_y],
        ],
        dtype=np.float32,
    )
    jitter_x = width * rng.uniform(0.02, 0.14)
    jitter_y = height * rng.uniform(0.02, 0.14)
    offsets = np.array(
        [
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
        ],
        dtype=np.float32,
    )
    quad = base + offsets
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return quad


def add_camera_degradation(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)

    if rng.random() < 0.7:
        sigma = rng.uniform(0.3, 1.2)
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    if rng.random() < 0.8:
        noise = np.random.normal(0.0, rng.uniform(1.0, 5.0), out.shape).astype(np.float32)
        out += noise

    if rng.random() < 0.5:
        h, _ = out.shape[:2]
        period = rng.uniform(4.0, 12.0)
        strength = rng.uniform(2.0, 8.0)
        yy = np.arange(h, dtype=np.float32).reshape(h, 1)
        stripe = np.sin(yy / period * np.pi * 2.0) * strength
        out += stripe[:, :, None]

    if rng.random() < 0.45:
        h, w = out.shape[:2]
        overlay = np.zeros_like(out)
        center = (rng.randint(0, w - 1), rng.randint(0, h - 1))
        radius = rng.randint(max(16, min(w, h) // 8), max(24, min(w, h) // 3))
        color = tuple(float(rng.uniform(60, 140)) for _ in range(3))
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (0, 0), radius / 3)
        out = cv2.addWeighted(out, 1.0, overlay, rng.uniform(0.08, 0.22), 0)

    if rng.random() < 0.75:
        out = out * rng.uniform(0.85, 1.15) + rng.uniform(-10, 12)

    return np.clip(out, 0, 255).astype(np.uint8)


def make_sample(clean: np.ndarray, out_width: int, out_height: int, rng: random.Random) -> tuple[np.ndarray, dict]:
    clean_h, clean_w = clean.shape[:2]
    canvas = np.full((out_height, out_width, 3), rng.randint(15, 45), dtype=np.uint8)

    src_quad = np.array(
        [[0, 0], [clean_w - 1, 0], [clean_w - 1, clean_h - 1], [0, clean_h - 1]],
        dtype=np.float32,
    )
    dst_quad = random_quad(out_width, out_height, rng)
    h_forward = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(clean, h_forward, (out_width, out_height), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(
        np.full((clean_h, clean_w), 255, dtype=np.uint8),
        h_forward,
        (out_width, out_height),
        flags=cv2.INTER_NEAREST,
    )
    canvas[mask > 0] = warped[mask > 0]
    canvas = add_camera_degradation(canvas, rng)

    h_inverse = cv2.getPerspectiveTransform(dst_quad, src_quad)
    label = {
        "source_quad": dst_quad.astype(float).tolist(),
        "target_size": [int(clean_w), int(clean_h)],
        "homography_to_clean": h_inverse.astype(float).tolist(),
        "homography_from_clean": h_forward.astype(float).tolist(),
    }
    return canvas, label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic perspective SCI data for U-Net corner training.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of clean SCI images.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset folder.")
    parser.add_argument("--count", type=int, default=500, help="Number of samples.")
    parser.add_argument("--width", type=int, default=1600, help="Synthetic camera image width.")
    parser.add_argument("--height", type=int, default=1000, help="Synthetic camera image height.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = iter_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    for idx in range(args.count):
        image_path = rng.choice(images)
        clean = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if clean is None:
            continue
        sample, label = make_sample(clean, args.width, args.height, rng)
        stem = f"{idx:06d}"
        image_out = args.output / f"{stem}.png"
        label_out = args.output / f"{stem}.json"
        cv2.imwrite(str(image_out), sample)
        label["clean_image"] = str(image_path)
        label_out.write_text(json.dumps(label, indent=2), encoding="utf-8")
        if idx % 50 == 0:
            print(f"[{idx}/{args.count}] {image_out}")


if __name__ == "__main__":
    main()
