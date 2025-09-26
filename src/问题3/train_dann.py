"""Stage 2: DANN adaptation using the pretrained source model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import json
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import ROOT
    from .data import create_dataloaders
    from .models import DANNModel
    from .paths import ensure_output_dirs
    from .train_utils import evaluate_dann, sanitize_config, set_seed, train_dann_epoch
    from .visualization import plot_domain_losses, plot_dual_axis_training
except ImportError:
    from paths import ROOT
    from data import create_dataloaders
    from models import DANNModel
    from paths import ensure_output_dirs
    from train_utils import evaluate_dann, sanitize_config, set_seed, train_dann_epoch
    from visualization import plot_domain_losses, plot_dual_axis_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DANN for Problem 3 domain adaptation")
    parser.add_argument("--csv-source", type=Path, default=ROOT / "data" / "processed" / "94feature.csv")
    parser.add_argument("--csv-target", type=Path, default=ROOT / "data" / "processed" / "16feature.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-domain", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figs-dir", type=Path, default=None)
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help="Optional explicit path to pretrained source checkpoint",
    )
    return parser.parse_args()


def _build_loaders(
    csv_source: Path,
    csv_target: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_loader, val_loader, target_loader, class_weights = create_dataloaders(
        source_csv=csv_source,
        target_csv=csv_target,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        seed=seed,
    )
    return train_loader, val_loader, target_loader, class_weights


def _load_pretrained(pretrained_path: Path | None, processed_dir: Path) -> dict:
    candidate = pretrained_path or (processed_dir / "source_pretrained_model.pth")
    if not candidate.is_file():
        raise FileNotFoundError(f"Pretrained source model not found at {candidate}")
    checkpoint = torch.load(candidate, map_location="cpu")
    return checkpoint["model_state"] if "model_state" in checkpoint else checkpoint


def run_training(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir, figs_dir = ensure_output_dirs(args.out_dir, args.figs_dir)

    train_loader, val_loader, target_loader, class_weights = _build_loaders(
        csv_source=args.csv_source,
        csv_target=args.csv_target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = DANNModel().to(device)
    pretrained_state = _load_pretrained(args.pretrained_path, processed_dir)
    model.load_source_weights(pretrained_state)

    label_weights = class_weights.to(device)
    label_criterion = nn.CrossEntropyLoss(weight=label_weights)
    domain_criterion = nn.CrossEntropyLoss()
    lr_domain = args.lr_domain if args.lr_domain is not None else args.lr * 2.0
    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters(), "lr": args.lr},
            {"params": model.label_classifier.parameters(), "lr": args.lr},
            {"params": model.domain_classifier.parameters(), "lr": lr_domain},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    history_label_loss: List[float] = []
    total_steps = max(1, len(train_loader) * args.epochs)
    current_step = 0
    history_domain_loss: List[float] = []
    history_train_acc: List[float] = []
    history_val_loss: List[float] = []
    history_val_acc: List[float] = []
    history_records: List[dict] = []
    best_path = processed_dir / "dann_model.pth"
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_stats, current_step = train_dann_epoch(
            model,
            train_loader,
            target_loader,
            label_criterion,
            domain_criterion,
            optimizer,
            device,
            start_step=current_step,
            total_steps=total_steps,
            alpha_max=args.alpha_max,
        )
        val_stats = evaluate_dann(model, val_loader, label_criterion, device)

        history_label_loss.append(train_stats.losses["label"])
        history_domain_loss.append(train_stats.losses.get("domain", 0.0))
        history_train_acc.append(train_stats.accuracy)
        history_val_loss.append(val_stats.losses["label"])
        history_val_acc.append(val_stats.accuracy)

        history_records.append(
            {
                "epoch": epoch,
                "train_label_loss": float(train_stats.losses["label"]),
                "train_domain_loss": float(train_stats.losses.get("domain", 0.0)),
                "train_acc": float(train_stats.accuracy),
                "val_loss": float(val_stats.losses["label"]),
                "val_acc": float(val_stats.accuracy),
                "alpha": float(args.alpha_max * (2.0 / (1.0 + math.exp(-10.0 * (epoch / max(1, args.epochs))))) - 1.0),
            }
        )

        print(
            f"Epoch {epoch:02d} | train label {train_stats.losses['label']:.4f} "
            f"domain {train_stats.losses['domain']:.4f} acc {train_stats.accuracy:.4f} | "
            f"val loss {val_stats.losses['label']:.4f} acc {val_stats.accuracy:.4f}"
        )

        if val_stats.accuracy > best_val_acc:
            best_val_acc = val_stats.accuracy
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": sanitize_config(vars(args)),
            }
            torch.save(checkpoint, best_path)

        scheduler.step()

    plot_dual_axis_training(
        {"train": history_label_loss, "val": history_val_loss},
        {"train": history_train_acc, "val": history_val_acc},
        figs_dir / "dann_training_curves.png",
        "DANN label classification performance",
    )

    plot_domain_losses(
        label_losses=history_label_loss,
        domain_losses=history_domain_loss,
        save_path=figs_dir / "dann_domain_losses.png",
        title="DANN label vs domain losses",
    )

    history_path = processed_dir / "dann_training_history.json"
    history_path.write_text(json.dumps(history_records, indent=2))
    return best_path


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
