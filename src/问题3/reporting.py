"""Problem 3 visualisation toolkit (essential outputs only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import ROOT
    from .data import INDEX_TO_LABEL
    from .paths import ensure_output_dirs
    from .visualization import save_prediction_table
except ImportError:
    from paths import ROOT
    from data import INDEX_TO_LABEL
    from paths import ensure_output_dirs
    from visualization import save_prediction_table

sns.set_theme(style="whitegrid", palette="deep")


def _load_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"History file missing: {path}")
    return pd.DataFrame(json.loads(path.read_text()))


def _load_predictions(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = processed_dir / "problem3_target_predictions.csv"
    prob_path = processed_dir / "problem3_target_probabilities.csv"
    if not pred_path.is_file() or not prob_path.is_file():
        raise FileNotFoundError("Run inference first to produce prediction CSVs.")
    return pd.read_csv(pred_path), pd.read_csv(prob_path)


def _format(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def generate_essentials(
    processed_dir: Path,
    figs_dir: Path,
) -> None:
    processed_dir, figs_dir = ensure_output_dirs(processed_dir, figs_dir)

    source_hist = _load_history(processed_dir / "source_training_history.json")
    dann_hist = _load_history(processed_dir / "dann_training_history.json")
    preds_df, prob_df = _load_predictions(processed_dir)

    figs_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(source_hist["epoch"], source_hist["train_loss"], marker="o", label="Train")
    ax.plot(source_hist["epoch"], source_hist["val_loss"], marker="s", label="Val")
    _format(ax, "Source pretraining loss", "Epoch", "Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figs_dir / "stage1_loss_curve.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(source_hist["epoch"], source_hist["train_acc"], marker="o", label="Train")
    ax.plot(source_hist["epoch"], source_hist["val_acc"], marker="s", label="Val")
    _format(ax, "Source pretraining accuracy", "Epoch", "Accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figs_dir / "stage1_accuracy_curve.png", dpi=300)
    plt.close(fig)

    # Stage 2 (DANN) training
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dann_hist["epoch"], dann_hist["train_label_loss"], marker="o", label="Train label")
    ax.plot(dann_hist["epoch"], dann_hist["val_loss"], marker="s", label="Val")
    _format(ax, "DANN classification loss", "Epoch", "Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figs_dir / "dann_training_curves.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dann_hist["epoch"], dann_hist["train_domain_loss"], marker="o", label="Domain loss")
    _format(ax, "DANN domain losses", "Epoch", "Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figs_dir / "dann_domain_losses.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dann_hist["epoch"], dann_hist["train_acc"], marker="o", label="Train acc")
    ax.plot(dann_hist["epoch"], dann_hist["val_acc"], marker="s", label="Val acc")
    _format(ax, "DANN accuracy curves", "Epoch", "Accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figs_dir / "dann_accuracy_curves.png", dpi=300)
    plt.close(fig)

    # 学习效果对比
    stage1_final = source_hist.iloc[-1]
    stage2_best = dann_hist.loc[dann_hist["val_acc"].idxmax()]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Stage 1", "Stage 2"], [stage1_final["val_acc"], stage2_best["val_acc"]], color=["#4c72b0", "#dd8452"])
    _format(ax, "Validation accuracy comparison", "Stage", "Accuracy")
    fig.tight_layout()
    fig.savefig(figs_dir / "domain_adaptation_comparison.png", dpi=300)
    plt.close(fig)

    # 目标域分析
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(preds_df["confidence"], bins=10, kde=True, color="#4c72b0", ax=ax)
    _format(ax, "Confidence histogram", "Confidence", "Count")
    fig.tight_layout()
    fig.savefig(figs_dir / "confidence_histogram.png", dpi=300)
    plt.close(fig)

    counts = preds_df["predicted_label"].value_counts().reindex(INDEX_TO_LABEL.values())
    fig, ax = plt.subplots(figsize=(6, 5))
    counts.plot(kind="bar", color="#55a868", ax=ax)
    _format(ax, "Fault statistics", "Label", "Files")
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(figs_dir / "fault_statistics_bar.png", dpi=300)
    plt.close(fig)

    prob_cols = [f"prob_{label}" for label in INDEX_TO_LABEL.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(prob_df[prob_cols], annot=True, fmt=".2f", cmap="YlGnBu",
                yticklabels=prob_df["filename"], xticklabels=INDEX_TO_LABEL.values(), ax=ax)
    ax.set_title("Target file probability heatmap")
    ax.set_xlabel("Class")
    ax.set_ylabel("File")
    fig.tight_layout()
    fig.savefig(figs_dir / "probability_heatmap.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="filename", y="confidence", data=preds_df, color="#4c72b0", ax=ax)
    ax.set_ylim(0, 1)
    _format(ax, "File-level confidence", "File", "Confidence")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.tight_layout()
    fig.savefig(figs_dir / "sample_confidence_bars.png", dpi=300)
    plt.close(fig)

    save_prediction_table(
        preds_df,
        figs_dir / "target_predictions_full.png",
        figs_dir / "target_predictions_full.pdf",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate essential Problem 3 visualisations")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed" / "问题3")
    parser.add_argument("--figs-dir", type=Path, default=ROOT / "figs" / "问题3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_essentials(args.processed_dir, args.figs_dir)


if __name__ == "__main__":
    main()
