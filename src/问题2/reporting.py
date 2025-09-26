from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize

from augmentation import AugmentationArtifacts
from data_processing import DataBundle
from feature_analysis import FeatureArtifacts
from modeling import ModelArtifacts
from problem2_config import PipelineConfig
from utils import safe_display, save_figure


def build_results_dataframe(results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "CV_Accuracy": [results[m]["cv_mean"] for m in results.keys()],
            "CV_Std": [results[m]["cv_std"] for m in results.keys()],
            "Test_Accuracy": [results[m]["test_accuracy"] for m in results.keys()],
            "Precision": [results[m]["test_precision"] for m in results.keys()],
            "Recall": [results[m]["test_recall"] for m in results.keys()],
            "F1_Score": [results[m]["test_f1"] for m in results.keys()],
        }
    )
    return df.sort_values("F1_Score", ascending=False).reset_index(drop=True)


def save_results_tables(results_df: pd.DataFrame, config: PipelineConfig) -> None:
    safe_display(results_df, title="Model performance summary:")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    path_primary = config.data_dir / "model_results.csv"
    results_df.to_csv(path_primary, index=False)
    print(f"\n✅ Results saved to: {path_primary}")

    # Maintain compatibility with notebook export naming
    path_augmented = config.data_dir / "model_results_augmented.csv"
    results_df.to_csv(path_augmented, index=False)
    print(f"✅ Results saved to: {path_augmented}")


def export_best_model_report(
    best_model: str,
    results: Dict[str, Dict[str, object]],
    augmentation_data: AugmentationArtifacts,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> Path:
    y_test = augmentation_data.y_test
    best_predictions = results[best_model]["predictions"]
    report = classification_report(y_test, best_predictions, target_names=class_names, output_dict=True)
    print(f"\n{best_model} Detailed Classification Report:")
    print("(Model trained on augmented data, tested on original test set)")
    print(classification_report(y_test, best_predictions, target_names=class_names))

    report_df = pd.DataFrame(report).transpose()
    path = config.data_dir / "detailed_classification_report_augmented.csv"
    report_df.to_csv(path)
    return path


def plot_model_performance_table(results_df: pd.DataFrame, config: PipelineConfig) -> Path:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("tight")
    ax.axis("off")

    table_data = []
    for _, row in results_df.iterrows():
        table_data.append(
            [
                row["Model"],
                f"{row['CV_Accuracy']:.4f} ± {row['CV_Std']:.4f}",
                f"{row['Test_Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1_Score']:.4f}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "CV Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for row_idx in range(1, len(table_data) + 1):
        for col_idx in range(len(table_data[0])):
            if row_idx % 2 == 0:
                table[(row_idx, col_idx)].set_facecolor("#F5F5F5")
            else:
                table[(row_idx, col_idx)].set_facecolor("#FFFFFF")

    ax.set_title("Model Performance Comparison Table", fontsize=16, fontweight="bold", pad=20)

    path = config.figure_dir / "model_performance_table.png"
    save_figure(fig, path)
    print(f"✅ Model performance table saved to: {path}")
    return path


def plot_detailed_analysis(
    results_df: pd.DataFrame,
    results: Dict[str, Dict[str, object]],
    best_model: str,
    data_bundle: DataBundle,
    feature_artifacts: FeatureArtifacts,
    augmentation_data: AugmentationArtifacts,
    model_artifacts: ModelArtifacts,
    config: PipelineConfig,
) -> Path:
    y_test = augmentation_data.y_test
    class_names = tuple(data_bundle.class_names)

    fig = plt.figure(figsize=(20, 16))

    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    metrics = ["Test_Accuracy", "Precision", "Recall", "F1_Score"]
    x = np.arange(len(results_df))
    width = 0.2
    for idx, metric in enumerate(metrics):
        ax1.bar(x + idx * width, results_df[metric], width, label=metric.replace("_", " "))

    ax1.set_xlabel("Models")
    ax1.set_ylabel("Score")
    ax1.set_title("Model Performance Comparison")
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(results_df["Model"], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cross-validation Stability
    ax2 = plt.subplot(3, 4, 2)
    models_with_cv = [row["Model"] for _, row in results_df.iterrows() if row["CV_Std"] > 0]
    ax2.errorbar(
        range(len(models_with_cv)),
        [results[m]["cv_mean"] for m in models_with_cv],
        yerr=[results[m]["cv_std"] for m in models_with_cv],
        fmt="o-",
        capsize=5,
    )
    ax2.set_xlabel("Models")
    ax2.set_ylabel("CV Accuracy")
    ax2.set_title("Cross-Validation Stability")
    ax2.set_xticks(range(len(models_with_cv)))
    ax2.set_xticklabels(models_with_cv, rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Best Model Confusion Matrix
    ax3 = plt.subplot(3, 4, 3)
    best_predictions = results[best_model]["predictions"]
    cm = confusion_matrix(y_test, best_predictions)
    im = ax3.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax3.set_title(f"Confusion Matrix - {best_model}")
    tick_marks = np.arange(len(class_names))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(class_names)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(class_names)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax3.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    # 4. ROC Curves
    ax4 = plt.subplot(3, 4, 4)
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    proba = results[best_model]["probabilities"]
    colors = plt.cm.get_cmap("tab10", len(class_names))
    for idx in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], proba[:, idx])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, lw=2, label=f"ROC curve {class_names[idx]} (AUC = {roc_auc:.2f})", color=colors(idx))
    ax4.plot([0, 1], [0, 1], "k--", lw=2)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title(f"ROC Curves - {best_model}")
    ax4.legend(loc="lower right", fontsize=8)

    # 5-8. Class-wise Performance
    for idx, class_name in enumerate(class_names):
        ax = plt.subplot(3, 4, 5 + idx)
        class_mask = y_test == idx
        class_acc = []
        model_labels = []
        if class_mask.sum() > 0:
            for model_name, info in results.items():
                pred = info["predictions"]
                class_acc.append(accuracy_score(y_test[class_mask], pred[class_mask]))
                model_labels.append(model_name)
        ax.bar(range(len(class_acc)), class_acc)
        ax.set_title(f"Class {class_name} Accuracy")
        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels([label.split()[0] for label in model_labels], rotation=45)
        ax.grid(True, alpha=0.3)
        if class_acc:
            max_idx = int(np.argmax(class_acc))
            ax.bar(max_idx, class_acc[max_idx], color="red", alpha=0.7)

    # 9-12. Feature importance per model
    feature_models = ["Random Forest", "XGBoost", "LightGBM", "Gradient Boosting"]
    top_features = feature_artifacts.top_features
    for idx, model_name in enumerate(feature_models):
        ax = plt.subplot(3, 4, 9 + idx)
        model = model_artifacts.trained_models.get(model_name)
        if model is None or not hasattr(model, "feature_importances_"):
            ax.axis("off")
            ax.set_title(f"{model_name} (n/a)")
            continue
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        ax.bar(range(len(indices)), importances[indices])
        ax.set_title(f"Feature Importance - {model_name}")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_xticks(range(len(indices)))
        labels = [top_features[i] if i < len(top_features) else f"f{i}" for i in indices]
        ax.set_xticklabels([lbl.split("_")[-1] for lbl in labels], rotation=45)

    fig.tight_layout()
    path = config.figure_dir / "detailed_analysis.png"
    save_figure(fig, path)
    print("✅ Detailed visualization analysis completed")
    print(f"✅ Detailed analysis figure saved to: {path}")
    return path


def plot_comprehensive_analysis(
    results_df: pd.DataFrame,
    best_model: str,
    results: Dict[str, Dict[str, object]],
    feature_artifacts: FeatureArtifacts,
    augmentation_data: AugmentationArtifacts,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> Path:
    y_test = augmentation_data.y_test
    best_predictions = results[best_model]["predictions"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax1 = axes[0, 0]
    x = np.arange(len(results_df))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, results_df["Test_Accuracy"], width, label="Test Accuracy", alpha=0.8, color="#2196F3")
    bars2 = ax1.bar(x + width / 2, results_df["F1_Score"], width, label="F1 Score", alpha=0.8, color="#FF9800")
    ax1.set_title("Model Performance Comparison\n(Trained on Augmented Data)", fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df["Model"], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars1, results_df["Test_Accuracy"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, f1 in zip(bars2, results_df["F1_Score"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, f1 + 0.01, f"{f1:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2 = axes[0, 1]
    categories = ["Original\nTraining", "Augmented\nTraining", "Test Set\n(Unchanged)"]
    counts = [augmentation_data.X_train.shape[0], augmentation_data.X_train_augmented.shape[0], augmentation_data.X_test.shape[0]]
    colors = ["#FF9800", "#4CAF50", "#2196F3"]
    bars = ax2.bar(categories, counts, color=colors, alpha=0.8)
    ax2.set_title("Dataset Size After Augmentation", fontweight="bold")
    ax2.set_ylabel("Number of Samples")
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, count + 5, f"{count}", ha="center", va="bottom", fontweight="bold")
        if count == augmentation_data.X_train_augmented.shape[0]:
            factor = count / augmentation_data.X_train.shape[0]
            ax2.text(bar.get_x() + bar.get_width() / 2, count / 2, f"{factor:.1f}x", ha="center", va="center", fontweight="bold", color="white", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[0, 2]
    per_class_accuracy = []
    for idx in range(len(class_names)):
        mask = y_test == idx
        if mask.sum() > 0:
            per_class_accuracy.append(accuracy_score(y_test[mask], best_predictions[mask]))
        else:
            per_class_accuracy.append(0)
    bars = ax3.bar(class_names, per_class_accuracy, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))), alpha=0.8)
    ax3.set_title(f"Per-Class Accuracy\n({best_model})", fontweight="bold")
    ax3.set_ylabel("Accuracy")
    ax3.set_xlabel("Class")
    for bar, acc in zip(bars, per_class_accuracy):
        ax3.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = axes[1, 0]
    cm = confusion_matrix(y_test, best_predictions)
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    ax4.set_title(f"Confusion Matrix\n{best_model}", fontweight="bold")
    tick_marks = np.arange(len(class_names))
    ax4.set_xticks(tick_marks)
    ax4.set_xticklabels(class_names)
    ax4.set_yticks(tick_marks)
    ax4.set_yticklabels(class_names)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax4.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontweight="bold")

    ax5 = axes[1, 1]
    models_for_comparison = [name for name in results if name != "Neural Network"]
    cv_accs = [results[name]["cv_mean"] for name in models_for_comparison]
    test_accs = [results[name]["test_accuracy"] for name in models_for_comparison]
    width = 0.35
    x = np.arange(len(models_for_comparison))
    bars1 = ax5.bar(x - width / 2, cv_accs, width, label="CV Accuracy", alpha=0.8, color="#9C27B0")
    bars2 = ax5.bar(x + width / 2, test_accs, width, label="Test Accuracy", alpha=0.8, color="#FF5722")
    ax5.set_title("Training vs Test Performance", fontweight="bold")
    ax5.set_ylabel("Accuracy")
    ax5.set_xticks(x)
    ax5.set_xticklabels([name.split()[0] for name in models_for_comparison], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    ax6 = axes[1, 2]
    ax6.axis("off")
    factor = augmentation_data.X_train_augmented.shape[0] / augmentation_data.X_train.shape[0]
    best_accuracy = results[best_model]["test_accuracy"]
    best_f1 = results[best_model]["test_f1"]
    summary_text = (
        "Data Augmentation Strategy:\n\n"
        "Method: Sliding Window + Gaussian Noise\n"
        "• Window Size: 70% of features\n"
        "• Stride: 30% of features\n"
        "• Noise Level: 2% Gaussian noise\n\n"
        f"Results:\n• Training samples: {augmentation_data.X_train.shape[0]} → {augmentation_data.X_train_augmented.shape[0]}\n"
        f"• Augmentation factor: {factor:.1f}x\n"
        f"• Test set: {augmentation_data.X_test.shape[0]} (unchanged)\n\n"
        f"Best Model: {best_model}\n"
        f"• Test Accuracy: {best_accuracy:.4f}\n"
        f"• F1 Score: {best_f1:.4f}"
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))

    fig.tight_layout()
    path = config.figure_dir / "comprehensive_augmentation_analysis.png"
    save_figure(fig, path)
    print(f"✅ Comprehensive analysis figure saved to: {path}")
    return path


def save_summary_json(
    data_bundle: DataBundle,
    feature_artifacts: FeatureArtifacts,
    augmentation_data: AugmentationArtifacts,
    results_df: pd.DataFrame,
    best_model: str,
    config: PipelineConfig,
) -> Path:
    summary = {
        "dataset_size_original": int(len(data_bundle.df)),
        "training_size_original": int(augmentation_data.X_train.shape[0]),
        "training_size_augmented": int(augmentation_data.X_train_augmented.shape[0]),
        "test_size": int(augmentation_data.X_test.shape[0]),
        "augmentation_factor": float(augmentation_data.X_train_augmented.shape[0] / augmentation_data.X_train.shape[0]),
        "num_features_original": int(len(data_bundle.numeric_columns)),
        "num_features_selected": int(len(feature_artifacts.top_features)),
        "num_classes": int(len(data_bundle.class_names)),
        "best_model": best_model,
        "best_f1_score": float(results_df.iloc[0]["F1_Score"]),
        "best_accuracy": float(results_df.iloc[0]["Test_Accuracy"]),
        "augmentation_method": "Sliding Window + Gaussian Noise",
        "window_size": 0.7,
        "stride": 0.3,
        "noise_level": 0.02,
    }

    path = config.data_dir / "analysis_summary_augmented.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path
