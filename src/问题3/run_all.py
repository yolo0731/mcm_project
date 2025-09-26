"""End-to-end runner covering source training, DANN adaptation, inference, and t-SNE plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import ROOT
    from . import train_dann, train_source, inference, tsne_analysis
    from .paths import ensure_output_dirs
except ImportError:
    # Fallback for direct execution
    from paths import ROOT
    import train_dann, train_source, inference, tsne_analysis
    from paths import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Problem 3 pipeline")
    parser.add_argument("--csv-source", type=Path, default=ROOT / "data" / "processed" / "94feature.csv")
    parser.add_argument("--csv-target", type=Path, default=ROOT / "data" / "processed" / "16feature.csv")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figs-dir", type=Path, default=None)
    parser.add_argument("--epochs-src", type=int, default=10)
    parser.add_argument("--epochs-dann", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-src", type=float, default=1e-3)
    parser.add_argument("--lr-dann", type=float, default=1e-4)
    parser.add_argument("--lr-dann-domain", type=float, default=None)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-tsne-samples", type=int, default=4000)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    processed_dir, figs_dir = ensure_output_dirs(args.out_dir, args.figs_dir)

    # Stage 1: source training
    source_args = argparse.Namespace(
        csv_source=args.csv_source,
        epochs=getattr(args, 'epochs_source', getattr(args, 'epochs_src', 15)),
        batch_size=args.batch_size,
        lr=getattr(args, 'lr_source', getattr(args, 'lr_src', 1e-3)),
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=getattr(args, 'num_workers', 0),
        val_ratio=getattr(args, 'val_ratio', 0.2),
        out_dir=processed_dir,
        figs_dir=figs_dir,
    )
    source_path = train_source.run_training(source_args)

    # Stage 2: DANN adaptation
    dann_args = argparse.Namespace(
        csv_source=args.csv_source,
        csv_target=args.csv_target,
        epochs=args.epochs_dann,
        batch_size=args.batch_size,
        lr=getattr(args, 'lr_dann', 5e-4),
        lr_domain=getattr(args, 'lr_domain', getattr(args, 'lr_dann_domain', 2e-3)),
        alpha_max=args.alpha_max,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=getattr(args, 'num_workers', 0),
        val_ratio=getattr(args, 'val_ratio', 0.2),
        out_dir=processed_dir,
        figs_dir=figs_dir,
        pretrained_path=source_path,
    )
    dann_path = train_dann.run_training(dann_args)

    if not getattr(args, 'skip_inference', False):
        infer_args = argparse.Namespace(
            csv_target=args.csv_target,
            model_path=dann_path,
            batch_size=args.batch_size,
            num_workers=getattr(args, 'num_workers', 0),
            seed=args.seed,
            out_dir=processed_dir,
            figs_dir=figs_dir,
            top_ratio=getattr(args, 'top_ratio', 0.75),
            temperature=getattr(args, 'temperature', 1.2),
        )
        inference.run_inference(infer_args)

    if not getattr(args, 'skip_tsne', False):
        tsne_analysis.plot_tsne(
            model_type="dann",
            csv_source=args.csv_source,
            csv_target=args.csv_target,
            model_path=dann_path,
            figs_dir=figs_dir,
            max_samples=getattr(args, 'max_tsne_samples', 4000),
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=getattr(args, 'num_workers', 0),
        )


def main(external_args=None) -> None:
    if external_args is not None:
        args = external_args
    else:
        args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
