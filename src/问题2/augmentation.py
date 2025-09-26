from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from problem2_config import PipelineConfig
from utils import print_class_distribution, save_figure


@dataclass
class AugmentationArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_augmented: np.ndarray
    y_train_augmented: np.ndarray
    X_train_augmented_scaled: np.ndarray
    X_test_scaled: np.ndarray
    scaler: StandardScaler
    class_counts_original: Dict[str, int]
    class_counts_augmented: Dict[str, int]


def split_and_augment(
    X_selected: pd.DataFrame,
    y_encoded: np.ndarray,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> AugmentationArtifacts:
    print("=== Data Splitting (8:2) and Data Augmentation ===")
    class_counts = pd.Series(y_encoded).value_counts().sort_index()
    print("Original class distribution:")
    print_class_distribution(class_counts.tolist(), class_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_encoded,
        test_size=config.test_size,
        random_state=config.train_test_split_seed,
        stratify=y_encoded,
    )

    print("\nOriginal split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    test_counts = pd.Series(y_test).value_counts().sort_index()
    print("\nTest set class distribution:")
    print_class_distribution(test_counts.reindex(range(len(class_names)), fill_value=0).tolist(), class_names)

    print("\n=== Data Augmentation: Sliding Window ===")
    X_train_augmented, y_train_augmented = _sliding_window_augmentation(
        X_train,
        y_train,
        window_size=config.augmentation_window,
        stride=config.augmentation_stride,
        noise_level=config.augmentation_noise,
    )

    print(f"Training set after augmentation: {X_train_augmented.shape[0]} samples")
    print(f"Augmentation factor: {X_train_augmented.shape[0] / X_train.shape[0]:.1f}x")

    augmented_counts = pd.Series(y_train_augmented).value_counts().sort_index()
    print("\nAugmented training set class distribution:")
    for idx, class_name in enumerate(class_names):
        original_count = int((y_train == idx).sum())
        augmented_count = int(augmented_counts.get(idx, 0))
        factor = (augmented_count / original_count) if original_count else 0
        print(f"  {class_name}: {augmented_count} samples (original: {original_count}, factor: {factor:.1f}x)")

    print("\n=== Standardization ===")
    scaler = StandardScaler()
    X_train_augmented_scaled = scaler.fit_transform(X_train_augmented)
    X_test_scaled = scaler.transform(X_test)

    mean_range = (X_train_augmented_scaled.mean(axis=0).min(), X_train_augmented_scaled.mean(axis=0).max())
    std_range = (X_train_augmented_scaled.std(axis=0).min(), X_train_augmented_scaled.std(axis=0).max())
    print(
        f"Post-standardization augmented training set mean range: [{mean_range[0]:.6f}, {mean_range[1]:.6f}]"
    )
    print(f"Post-standardization augmented training set std range: [{std_range[0]:.6f}, {std_range[1]:.6f}]")
    print("✅ Data splitting and augmentation completed")

    class_counts_original = {
        class_names[i]: int(class_counts.get(i, 0))
        for i in range(len(class_names))
    }
    class_counts_augmented = {
        class_names[i]: int(augmented_counts.get(i, 0))
        for i in range(len(class_names))
    }

    _plot_augmentation_figures(
        X_train,
        X_train_augmented,
        X_test,
        y_train,
        y_train_augmented,
        class_names,
        config,
    )
    _plot_augmentation_table(
        X_train,
        X_train_augmented,
        X_test,
        y_train,
        y_train_augmented,
        class_names,
        config,
    )

    return AugmentationArtifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_augmented=X_train_augmented,
        y_train_augmented=y_train_augmented,
        X_train_augmented_scaled=X_train_augmented_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        class_counts_original=class_counts_original,
        class_counts_augmented=class_counts_augmented,
    )


def _sliding_window_augmentation(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    window_size: float,
    stride: float,
    noise_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    X_augmented = [X.values]
    y_augmented = [y]

    n_features = X.shape[1]
    window_length = max(1, int(n_features * window_size))
    step_size = max(1, int(n_features * stride))

    for start_idx in range(0, n_features - window_length + 1, step_size):
        end_idx = start_idx + window_length
        X_window = np.zeros_like(X.values)
        X_window[:, start_idx:end_idx] = X.iloc[:, start_idx:end_idx].values
        noise = np.random.normal(0, noise_level, X_window.shape)
        X_window += noise
        X_augmented.append(X_window)
        y_augmented.append(y)

    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)
    return X_final, y_final


def _plot_augmentation_figures(
    X_train: pd.DataFrame,
    X_train_augmented: np.ndarray,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_train_augmented: np.ndarray,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    categories = ["Original Training", "Augmented Training", "Test Set"]
    counts = [X_train.shape[0], X_train_augmented.shape[0], X_test.shape[0]]
    colors = ["#FF9800", "#4CAF50", "#2196F3"]
    ax1 = axes[0, 0]
    bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
    ax1.set_title("Dataset Size Comparison")
    ax1.set_ylabel("Number of Samples")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, count + 1, f"{count}", ha="center", va="bottom", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[0, 1]
    x = np.arange(len(class_names))
    width = 0.35
    original_counts = [(y_train == i).sum() for i in range(len(class_names))]
    augmented_counts = [(y_train_augmented == i).sum() for i in range(len(class_names))]
    bars1 = ax2.bar(x - width / 2, original_counts, width, label="Original Training", color="#FF9800", alpha=0.8)
    bars2 = ax2.bar(x + width / 2, augmented_counts, width, label="Augmented Training", color="#4CAF50", alpha=0.8)
    ax2.set_title("Class Distribution: Original vs Augmented")
    ax2.set_ylabel("Number of Samples")
    ax2.set_xlabel("Classes")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[1, 0]
    augmentation_factors = [
        (augmented_counts[i] / original_counts[i]) if original_counts[i] else 0
        for i in range(len(class_names))
    ]
    bars = ax3.bar(class_names, augmentation_factors, color="#9C27B0", alpha=0.8)
    ax3.set_title("Augmentation Factor by Class")
    ax3.set_ylabel("Augmentation Factor")
    ax3.set_xlabel("Classes")
    for bar, factor in zip(bars, augmentation_factors):
        ax3.text(bar.get_x() + bar.get_width() / 2, factor + 0.1, f"{factor:.1f}x", ha="center", va="bottom", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = axes[1, 1]
    original_std = X_train.std(axis=0).mean()
    augmented_std = pd.DataFrame(X_train_augmented).std(axis=0).mean()
    test_std = X_test.std(axis=0).mean()
    stats_categories = ["Original Training", "Augmented Training", "Test Set"]
    std_values = [original_std, augmented_std, test_std]
    bars = ax4.bar(stats_categories, std_values, color=["#FF9800", "#4CAF50", "#2196F3"], alpha=0.8)
    ax4.set_title("Feature Variability Comparison")
    ax4.set_ylabel("Mean Standard Deviation")
    for bar, std_val in zip(bars, std_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, std_val + 0.1, f"{std_val:.3f}", ha="center", va="bottom", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = config.figure_dir / "data_augmentation_analysis.png"
    save_figure(fig, fig_path)
    print(f"✅ Data augmentation visualization saved to: {fig_path}")


def _plot_augmentation_table(
    X_train: pd.DataFrame,
    X_train_augmented: np.ndarray,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_train_augmented: np.ndarray,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    aug_summary_data = [
        ["Original Training Set", f"{X_train.shape[0]} samples"],
        ["Augmented Training Set", f"{X_train_augmented.shape[0]} samples"],
        ["Test Set (unchanged)", f"{X_test.shape[0]} samples"],
        ["Augmentation Method", "Sliding Window + Gaussian Noise"],
        ["Window Size", "70% of features"],
        ["Stride", "30% of features"],
        ["Noise Level", "2% Gaussian noise"],
        [
            "Overall Augmentation Factor",
            f"{X_train_augmented.shape[0] / X_train.shape[0]:.1f}x",
        ],
    ]

    for idx, class_name in enumerate(class_names):
        original_count = int((y_train == idx).sum())
        augmented_count = int((y_train_augmented == idx).sum())
        factor = (augmented_count / original_count) if original_count else 0
        aug_summary_data.append(
            [f"{class_name} Class", f"{original_count} → {augmented_count} ({factor:.1f}x)"]
        )

    table = ax.table(
        cellText=aug_summary_data,
        colLabels=["Augmentation Property", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for i in range(2):
        table[(0, i)].set_facecolor("#9C27B0")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(aug_summary_data) + 1):
        for j in range(2):
            if i <= 8:
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#F3E5F5")
                else:
                    table[(i, j)].set_facecolor("#FFFFFF")
            else:
                table[(i, j)].set_facecolor("#E1BEE7")

    ax.set_title("Data Augmentation Summary", fontsize=16, fontweight="bold", pad=20)

    fig_path = config.figure_dir / "data_augmentation_table.png"
    save_figure(fig, fig_path)
    print(f"✅ Data augmentation table saved to: {fig_path}")

