"""Visualization helpers for Problem 3 outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def plot_training_curves(
    history: Dict[str, Sequence[float]],
    save_path: Path,
    title: str,
    ylabel: str = "Loss",
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(next(iter(history.values()))) + 1)
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        plt.plot(epochs, values, marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_dual_axis_training(
    losses: Dict[str, Sequence[float]],
    accuracies: Dict[str, Sequence[float]],
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(next(iter(losses.values()))) + 1)
    plt.figure(figsize=(9, 5))
    ax1 = plt.gca()
    for key, values in losses.items():
        ax1.plot(epochs, values, marker="o", label=f"{key} loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    for key, values in accuracies.items():
        ax2.plot(epochs, values, marker="s", linestyle="--", label=f"{key} acc")
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis="y")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labels + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_domain_losses(
    label_losses: Sequence[float],
    domain_losses: Sequence[float],
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(label_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, label_losses, marker="o", label="Label loss")
    plt.plot(epochs, domain_losses, marker="s", label="Domain loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    label_palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
    domain_styles = {0: {"marker": "o", "label": "Source"}, 1: {"marker": "^", "label": "Target"}}
    for label_idx in np.unique(labels):
        mask_label = labels == label_idx
        for domain_idx, style in domain_styles.items():
            mask = mask_label & (domains == domain_idx)
            if not np.any(mask):
                continue
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                marker=style["marker"],
                label=f"{style['label']} - class {label_idx}",
                s=32,
                alpha=0.7,
                color=label_palette[int(label_idx) % len(label_palette)],
            )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_distribution(counts: Dict[str, int], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=values, color="steelblue")
    plt.xlabel("Predicted class")
    plt.ylabel("Number of files")
    plt.title("Target-domain prediction distribution")
    for idx, value in enumerate(values):
        plt.text(idx, value + 0.05, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_prediction_table(
    predictions: pd.DataFrame,
    save_png: Path,
    save_pdf: Path,
) -> None:
    save_png.parent.mkdir(parents=True, exist_ok=True)
    columns = ['filename', 'predicted_label', 'confidence']
    table_df = predictions[columns].copy()
    cell_text = [
        [row['filename'], row['predicted_label'], f"{row['confidence']:.3f}"]
        for _, row in table_df.iterrows()
    ]
    fig_height = 0.5 + 0.3 * max(1, len(cell_text))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["File", "Predicted label", "Confidence"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    plt.title("Problem 3 target predictions", pad=20)
    plt.tight_layout()
    fig.savefig(save_png, dpi=300)
    fig.savefig(save_pdf, dpi=300)
    plt.close(fig)
