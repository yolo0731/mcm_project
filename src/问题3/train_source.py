"""Stage 1: train a source-domain classifier for Problem 3."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import json

import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import ROOT
    from .data import (
        BearingDataset,
        compute_class_weights,
        load_metadata,
        make_balanced_sampler,
        stratified_train_val_split,
    )
    from .models import SourceClassifier
    from .paths import ensure_output_dirs
    from .train_utils import evaluate_source, sanitize_config, set_seed, train_source_epoch
    from .visualization import plot_dual_axis_training
except ImportError:
    from paths import ROOT
    from data import (
        BearingDataset,
        compute_class_weights,
        load_metadata,
        make_balanced_sampler,
        stratified_train_val_split,
    )
    from models import SourceClassifier
    from paths import ensure_output_dirs
    from train_utils import evaluate_source, sanitize_config, set_seed, train_source_epoch
    from visualization import plot_dual_axis_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-domain classifier for Problem 3")
    parser.add_argument("--csv-source", type=Path, default=ROOT / "data" / "processed" / "94feature.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory to store model checkpoints")
    parser.add_argument("--figs-dir", type=Path, default=None, help="Directory to store figures")
    return parser.parse_args()


def build_loaders(
    csv_source: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    source_meta = load_metadata(csv_source)
    train_meta, val_meta = stratified_train_val_split(source_meta, val_ratio=val_ratio)
    train_dataset = BearingDataset(train_meta, domain_label=0)
    val_dataset = BearingDataset(val_meta, domain_label=0, cache_signals=False)
    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_weights = compute_class_weights(train_dataset.class_counts())
    return train_loader, val_loader, class_weights


def run_training(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir, figs_dir = ensure_output_dirs(args.out_dir, args.figs_dir)

    train_loader, val_loader, class_weights = build_loaders(
        csv_source=args.csv_source,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
    )

    model = SourceClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_acc = 0.0
    history_losses: Dict[str, List[float]] = {"train": [], "val": []}
    history_acc: Dict[str, List[float]] = {"train": [], "val": []}
    history_records: List[Dict[str, float]] = []
    best_path = processed_dir / "source_pretrained_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_source_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = evaluate_source(model, val_loader, criterion, device)
        history_losses["train"].append(train_stats.losses["label"])
        history_losses["val"].append(val_stats.losses["label"])
        history_acc["train"].append(train_stats.accuracy)
        history_acc["val"].append(val_stats.accuracy)

        print(
            f"Epoch {epoch:02d} | train loss {train_stats.losses['label']:.4f} "
            f"acc {train_stats.accuracy:.4f} | val loss {val_stats.losses['label']:.4f} "
            f"acc {val_stats.accuracy:.4f}"
        )

        history_records.append(
            {
                "epoch": epoch,
                "train_loss": float(train_stats.losses["label"]),
                "val_loss": float(val_stats.losses["label"]),
                "train_acc": float(train_stats.accuracy),
                "val_acc": float(val_stats.accuracy),
            }
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
        history_losses,
        history_acc,
        figs_dir / "source_training_curves.png",
        "Source-domain training curves",
    )

    history_path = processed_dir / "source_training_history.json"
    history_path.write_text(json.dumps(history_records, indent=2))
    return best_path


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
