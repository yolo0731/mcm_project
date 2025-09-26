from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from augmentation import AugmentationArtifacts
from problem2_config import PipelineConfig
from utils import save_figure

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError:  # pragma: no cover - optional dependency
    tf = None  # type: ignore


@dataclass
class ModelArtifacts:
    results: Dict[str, Dict[str, object]]
    trained_models: Dict[str, object]
    ensemble_model: VotingClassifier
    confusion_matrix_path: str
    nn_history_path: str | None


def tensorflow_available() -> bool:
    return tf is not None


def define_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=30,
            max_depth=3,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=0.5,
            gamma="scale",
            probability=True,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            C=0.5,
            penalty="l2",
            max_iter=1000,
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=42,
            eval_metric="mlogloss",
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            min_child_samples=3,
            random_state=42,
            verbose=-1,
        )

    return models


def train_models(
    data: AugmentationArtifacts,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> ModelArtifacts:
    print("=== Define Improved Models ===")
    total_samples = data.X_train.shape[0] + data.X_test.shape[0]
    print(f"Dataset size: {total_samples} samples")
    print(f"Number of features: {data.X_train.shape[1]}")

    models = define_models()
    print(f"Defined {len(models)} base models (optimized for small datasets)")
    print("Model list:", list(models.keys()))

    min_class_samples = pd.Series(np.hstack([data.y_train, data.y_test])).value_counts().min()
    cv_folds_initial = min(5, max(2, min_class_samples))
    print(
        f"Cross-validation strategy: {cv_folds_initial}-fold StratifiedKFold (based on minimum class samples: {min_class_samples})"
    )
    print("✅ Model definition completed")

    print("=== Model Training on Augmented Data and Testing on Original Test Set ===")
    print(f"Training on: {data.X_train_augmented_scaled.shape[0]} augmented samples")
    print(f"Testing on: {data.X_test_scaled.shape[0]} original test samples")

    results: Dict[str, Dict[str, object]] = {}
    trained_models: Dict[str, object] = {}

    augmented_counts = pd.Series(data.y_train_augmented).value_counts()
    min_class_samples_augmented = augmented_counts.min()
    cv_folds = max(2, min(5, int(min_class_samples_augmented // 3)))
    if cv_folds < 2:
        cv_folds = 2
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.train_test_split_seed)
    print(f"Cross-validation strategy: {cv_folds}-fold StratifiedKFold")

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        cv_scores = cross_val_score(
            model,
            data.X_train_augmented_scaled,
            data.y_train_augmented,
            cv=cv_strategy,
            scoring="accuracy",
            n_jobs=-1,
        )
        model.fit(data.X_train_augmented_scaled, data.y_train_augmented)
        trained_models[name] = model

        y_pred = model.predict(data.X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(data.X_test_scaled)
        else:
            y_proba = np.zeros((len(y_pred), len(class_names)))

        accuracy = accuracy_score(data.y_test, y_pred)
        precision = precision_score(data.y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(data.y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(data.y_test, y_pred, average="macro", zero_division=0)

        results[name] = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "predictions": y_pred,
            "probabilities": y_proba,
        }

        print(f"Cross-validation accuracy (on augmented data): {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"Test set accuracy (on original test data): {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")

    print("\n✅ All model training on augmented data completed")

    ensemble_artifacts = _train_ensemble(trained_models, results, data, class_names, cv_strategy)
    results.update(ensemble_artifacts["results_update"])
    confusion_matrix_path = _plot_confusion_matrices(results, data.y_test, class_names, config)

    nn_history_path = None
    if tf is not None:
        nn_history_path, nn_results = _train_neural_network(data, class_names, config, results)
        if nn_results is not None:
            results.update(nn_results)

    return ModelArtifacts(
        results=results,
        trained_models=trained_models,
        ensemble_model=ensemble_artifacts["ensemble_model"],
        confusion_matrix_path=str(confusion_matrix_path),
        nn_history_path=nn_history_path,
    )


def _train_ensemble(
    trained_models: Dict[str, object],
    results: Dict[str, Dict[str, np.ndarray]],
    data: AugmentationArtifacts,
    class_names: Tuple[str, ...],
    cv_strategy: StratifiedKFold,
) -> Dict[str, object]:
    print("=== Create Ensemble Model (Trained on Augmented Data) ===")
    model_performance = sorted(
        ((name, info["test_accuracy"]) for name, info in results.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    top_3_models = [name for name, _ in model_performance[:3]]
    print(f"Selected top 3 performing models: {top_3_models}")

    estimators = [(name, trained_models[name]) for name in top_3_models]
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(data.X_train_augmented_scaled, data.y_train_augmented)

    y_pred = ensemble.predict(data.X_test_scaled)
    y_proba = ensemble.predict_proba(data.X_test_scaled)

    accuracy = accuracy_score(data.y_test, y_pred)
    precision = precision_score(data.y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(data.y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(data.y_test, y_pred, average="macro", zero_division=0)

    cv_scores = cross_val_score(
        ensemble,
        data.X_train_augmented_scaled,
        data.y_train_augmented,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("\nEnsemble model performance (trained on augmented data, tested on original):")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
    print(f"Test set accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("✅ Ensemble model creation completed")

    return {
        "ensemble_model": ensemble,
        "results_update": {
            "Ensemble": {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "predictions": y_pred,
                "probabilities": y_proba,
            }
        },
    }


def _plot_confusion_matrices(
    results: Dict[str, Dict[str, np.ndarray]],
    y_test: np.ndarray,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (model_name, info) in enumerate(results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cm = confusion_matrix(y_test, info["predictions"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{model_name}\n(Trained on Augmented Data)", fontsize=12, fontweight="bold")
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Confusion Matrices - Models Trained on Augmented Data, Tested on Original Data",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig_path = config.figure_dir / "confusion_matrices_augmented_models.png"
    save_figure(fig, fig_path)
    print(f"✅ Confusion matrices saved to: {fig_path}")
    return str(fig_path)


def _train_neural_network(
    data: AugmentationArtifacts,
    class_names: Tuple[str, ...],
    config: PipelineConfig,
    results: Dict[str, Dict[str, object]],
):
    if tf is None:
        print("⚠️ Skipping deep learning model (TensorFlow not available)")
        return None, None

    print("=== Training Deep Learning Model on Augmented Data ===")
    tf.random.set_seed(config.train_test_split_seed)

    input_dim = data.X_train_augmented_scaled.shape[1]
    num_classes = len(class_names)

    model = Sequential(
        [
            Dense(512, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=0.0001),
    ]

    history = model.fit(
        data.X_train_augmented_scaled,
        data.y_train_augmented,
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    y_proba = model.predict(data.X_test_scaled, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    accuracy = accuracy_score(data.y_test, y_pred)
    precision = precision_score(data.y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(data.y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(data.y_test, y_pred, average="macro", zero_division=0)

    print("Neural network performance (trained on augmented data, tested on original):")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Test set accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history.history["loss"], label="Training Loss", color="#FF9800")
    axes[0, 0].plot(history.history["val_loss"], label="Validation Loss", color="#2196F3")
    axes[0, 0].set_title("Model Loss (Trained on Augmented Data)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history.history["accuracy"], label="Training Accuracy", color="#FF9800")
    axes[0, 1].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#2196F3")
    axes[0, 1].set_title("Model Accuracy (Trained on Augmented Data)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(
        ["Original Training", "Augmented Training", "Test Set"],
        [data.X_train.shape[0], data.X_train_augmented.shape[0], data.X_test.shape[0]],
        color=["#FF9800", "#4CAF50", "#2196F3"],
        alpha=0.8,
    )
    axes[1, 0].set_title("Dataset Size Comparison")
    axes[1, 0].set_ylabel("Number of Samples")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    comparison_order = ["Random Forest", "XGBoost", "LightGBM", "Neural Network"]
    model_labels: List[str] = []
    accuracies: List[float] = []
    for name in comparison_order:
        if name == "Neural Network":
            model_labels.append(name)
            accuracies.append(accuracy)
        elif name in results:
            model_labels.append(name)
            accuracies.append(float(results[name]["test_accuracy"]))

    axes[1, 1].bar(
        range(len(model_labels)),
        accuracies,
        color=["#FF5722", "#8BC34A", "#9C27B0", "#E91E63"][: len(model_labels)],
        alpha=0.8,
    )
    axes[1, 1].set_title("Model Accuracy Comparison (Test Set)")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_xticks(range(len(model_labels)))
    axes[1, 1].set_xticklabels(model_labels, rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for idx, acc in enumerate(accuracies):
        axes[1, 1].text(idx, acc + 0.01, f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    fig_path = config.figure_dir / "neural_network_augmented_training.png"
    save_figure(fig, fig_path)
    print("✅ Deep learning model training on augmented data completed")
    print(f"✅ Neural network training history figure saved to: {fig_path}")

    return str(fig_path), {
        "Neural Network": {
            "cv_mean": max(history.history["val_accuracy"]),
            "cv_std": 0.0,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "predictions": y_pred,
            "probabilities": y_proba,
        }
    }
