from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

from data_processing import DataBundle
from problem2_config import PipelineConfig
from utils import save_figure, safe_display


@dataclass
class FeatureArtifacts:
    feature_importance_df: pd.DataFrame
    top_features: List[str]
    X_selected: pd.DataFrame


def _normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if np.isclose(max_val, min_val):
        return pd.Series(np.zeros_like(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def perform_feature_analysis(bundle: DataBundle, config: PipelineConfig) -> FeatureArtifacts:
    X = bundle.X_numeric
    y = bundle.y_encoded

    print("=== Feature Importance Analysis ===")
    print(f"Original feature count: {len(bundle.numeric_columns)}")
    print(f"Sample count: {len(y)}")
    print(f"Feature/sample ratio: {len(bundle.numeric_columns) / len(y):.2f}")

    variance_selector = VarianceThreshold(threshold=config.feature_variance_threshold)
    X_var_filtered = variance_selector.fit_transform(X)
    high_var_features = X.columns[variance_selector.get_support()].tolist()
    X_filtered = pd.DataFrame(X_var_filtered, columns=high_var_features, index=X.index)
    print(f"After removing low variance features: {len(high_var_features)}")

    selector_univariate = SelectKBest(f_classif, k="all")
    selector_univariate.fit(X_filtered, y)
    univariate_scores = selector_univariate.scores_

    rf_importance = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_importance.fit(X_filtered, y)
    rf_scores = rf_importance.feature_importances_

    gb_importance = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42,
    )
    gb_importance.fit(X_filtered, y)
    gb_scores = gb_importance.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "feature": high_var_features,
            "univariate_score": univariate_scores,
            "rf_importance": rf_scores,
            "gb_importance": gb_scores,
        }
    )

    for col in ["univariate_score", "rf_importance", "gb_importance"]:
        feature_importance_df[col] = _normalize(feature_importance_df[col])

    feature_importance_df["combined_score"] = (
        0.3 * feature_importance_df["univariate_score"]
        + 0.4 * feature_importance_df["rf_importance"]
        + 0.3 * feature_importance_df["gb_importance"]
    )

    feature_importance_df = feature_importance_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    safe_display(feature_importance_df.head(10), title="Top-10 important features:")

    n_top_features = min(
        config.max_top_features,
        max(config.min_top_features, int(len(y) * 0.3)),
    )
    top_features = feature_importance_df.head(n_top_features)["feature"].tolist()
    X_selected = X_filtered[top_features].copy()

    print(
        f"\n✅ Selected {len(top_features)} most important features ({len(top_features)/len(y)*100:.1f}% of sample count)"
    )
    print(f"New feature/sample ratio: {len(top_features) / len(y):.2f}")
    print(f"Selected features: {top_features[:5]}...")

    top_feat_path = config.data_dir / "top_features.csv"
    top_feat_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": top_features}).to_csv(top_feat_path, index=False)
    print(f"✅ Feature list saved to: {top_feat_path}")

    class_distribution = bundle.y.value_counts()
    _plot_top_feature_table(feature_importance_df, config)
    _plot_feature_overview(
        feature_importance_df,
        X_selected,
        class_distribution,
        config,
    )

    return FeatureArtifacts(
        feature_importance_df=feature_importance_df,
        top_features=top_features,
        X_selected=X_selected,
    )


def _plot_top_feature_table(feature_importance_df: pd.DataFrame, config: PipelineConfig) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("tight")
    ax.axis("off")

    top_15_features = feature_importance_df.head(15)
    features_data = []
    for idx, row in enumerate(top_15_features.itertuples(index=False), start=1):
        features_data.append(
            [
                str(idx),
                row.feature,
                f"{row.combined_score:.4f}",
                f"{row.univariate_score:.4f}",
                f"{row.rf_importance:.4f}",
                f"{row.gb_importance:.4f}",
            ]
        )

    table = ax.table(
        cellText=features_data,
        colLabels=[
            "Rank",
            "Feature Name",
            "Combined Score",
            "Univariate",
            "Random Forest",
            "Gradient Boosting",
        ],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for i in range(6):
        table[(0, i)].set_facecolor("#9C27B0")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for row_idx in range(1, len(features_data) + 1):
        for col_idx in range(6):
            if row_idx <= 5:
                table[(row_idx, col_idx)].set_facecolor("#E1BEE7")
                table[(row_idx, col_idx)].set_text_props(weight="bold")
            elif row_idx % 2 == 0:
                table[(row_idx, col_idx)].set_facecolor("#F3E5F5")
            else:
                table[(row_idx, col_idx)].set_facecolor("#FFFFFF")

    ax.set_title("Top 15 Most Important Features", fontsize=16, fontweight="bold", pad=20)

    fig_path = config.figure_dir / "top_features_table.png"
    save_figure(fig, fig_path)
    print(f"✅ Top features table saved to: {fig_path}")


def _plot_feature_overview(
    feature_importance_df: pd.DataFrame,
    X_selected: pd.DataFrame,
    class_distribution: pd.Series,
    config: PipelineConfig,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    top_10 = feature_importance_df.head(10)
    ax1 = axes[0, 0]
    bars = ax1.barh(range(len(top_10)), top_10["combined_score"])
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10["feature"], fontsize=10)
    ax1.set_xlabel("Combined Importance Score")
    ax1.set_title("Top-10 Feature Importance")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    top_5 = feature_importance_df.head(5)
    x = np.arange(len(top_5))
    width = 0.25
    ax2.bar(x - width, top_5["univariate_score"], width, label="Univariate", alpha=0.8)
    ax2.bar(x, top_5["rf_importance"], width, label="Random Forest", alpha=0.8)
    ax2.bar(x + width, top_5["gb_importance"], width, label="Gradient Boosting", alpha=0.8)
    ax2.set_xlabel("Features")
    ax2.set_ylabel("Normalized Importance Score")
    ax2.set_title("Feature Importance by Different Methods")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.split("_")[-1] for f in top_5["feature"]], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    corr_cols = min(8, X_selected.shape[1])
    if corr_cols > 0:
        corr_matrix = X_selected.iloc[:, :corr_cols].corr()
        im = ax3.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(corr_matrix.columns)))
        ax3.set_yticks(range(len(corr_matrix.columns)))
        ax3.set_xticklabels([f.split("_")[-1] for f in corr_matrix.columns], rotation=45)
        ax3.set_yticklabels([f.split("_")[-1] for f in corr_matrix.columns])
        ax3.set_title("Feature Correlation Matrix")
        fig.colorbar(im, ax=ax3)
    else:
        ax3.text(0.5, 0.5, "No features selected", ha="center", va="center")
        ax3.set_axis_off()

    ax4 = axes[1, 1]
    class_counts = class_distribution
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    ax4.pie(
        class_counts.values,
        labels=class_counts.index,
        autopct="%1.1f%%",
        colors=colors,
    )
    ax4.set_title("Class Distribution")

    fig.tight_layout()
    fig_path = config.figure_dir / "feature_analysis.png"
    save_figure(fig, fig_path)
    print("✅ Feature analysis visualization completed")
    print(f"✅ Figure saved to: {fig_path}")
