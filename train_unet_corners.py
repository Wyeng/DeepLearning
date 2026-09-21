from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a U-Net corner heatmap model for SCI rectification.")
    parser.add_argument("--data", type=Path, required=True, help="Synthetic dataset folder containing PNG/JSON pairs.")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint/output folder.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coord-loss-weight", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    from unet_rectify.learning import (
        SyntheticCornerDataset,
        UNetCornerNet,
        decode_heatmap_corners,
        softargmax_heatmap_corners,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticCornerDataset(args.data, input_size=args.input_size, max_samples=args.max_samples)
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * args.val_ratio)) if len(indices) > 1 else 0
    val_indices = indices[:val_count]
    train_indices = indices[val_count:] if val_count else indices

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()) if val_count else Subset(dataset, train_indices[:1].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNetCornerNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = []
    best_error = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["heatmap"].to(device, non_blocking=True)
            quads = batch["quad"].to(device, non_blocking=True)

            logits = model(images)
            heatmap_loss = F.binary_cross_entropy_with_logits(logits, targets)
            pred_soft = softargmax_heatmap_corners(logits)
            coord_loss = F.smooth_l1_loss(pred_soft / args.input_size, quads / args.input_size)
            loss = heatmap_loss + args.coord_loss_weight * coord_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        val_losses = []
        corner_errors = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["heatmap"].to(device, non_blocking=True)
                quads = batch["quad"].to(device, non_blocking=True)

                logits = model(images)
                heatmap_loss = F.binary_cross_entropy_with_logits(logits, targets)
                pred_soft = softargmax_heatmap_corners(logits)
                coord_loss = F.smooth_l1_loss(pred_soft / args.input_size, quads / args.input_size)
                loss = heatmap_loss + args.coord_loss_weight * coord_loss
                pred = decode_heatmap_corners(logits)
                error = torch.linalg.norm(pred - quads, dim=-1).mean()
                val_losses.append(float(loss.detach().cpu()))
                corner_errors.append(float(error.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        corner_error = float(np.mean(corner_errors)) if corner_errors else 0.0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "corner_error_px": corner_error,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} corner_error_px={corner_error:.2f}"
        )

        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        checkpoint = {
            "model": model.state_dict(),
            "args": serializable_args,
            "history": history,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if corner_error < best_error:
            best_error = corner_error
            torch.save(checkpoint, args.output / "best.pt")

    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
