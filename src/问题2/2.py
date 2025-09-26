from __future__ import annotations

import sys
import warnings
from pathlib import Path

import seaborn as sns

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from augmentation import split_and_augment
from data_processing import infer_target_column, load_feature_data, preprocess_features
from feature_analysis import perform_feature_analysis
from modeling import tensorflow_available, train_models
from problem2_config import PipelineConfig
from reporting import (
    build_results_dataframe,
    export_best_model_report,
    plot_comprehensive_analysis,
    plot_detailed_analysis,
    plot_model_performance_table,
    save_results_tables,
    save_summary_json,
)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    print("# 问题2：源域故障诊断 - 改进算法版")
    config = PipelineConfig()
    print("✅ Import completed")
    if not tensorflow_available():
        print("⚠️ TensorFlow not available, will skip deep learning models")

    df = load_feature_data(config)
    target_col = infer_target_column(df, config)
    data_bundle = preprocess_features(df, target_col, config)

    feature_artifacts = perform_feature_analysis(data_bundle, config)
    class_names = tuple(data_bundle.class_names)

    augmentation_data = split_and_augment(
        feature_artifacts.X_selected,
        data_bundle.y_encoded,
        class_names,
        config,
    )

    model_artifacts = train_models(augmentation_data, class_names, config)

    results_df = build_results_dataframe(model_artifacts.results)
    save_results_tables(results_df, config)

    best_model = results_df.iloc[0]["Model"]
    best_f1 = results_df.iloc[0]["F1_Score"]
    print(f"\n🏆 Best model: {best_model} (F1: {best_f1:.4f})")

    plot_model_performance_table(results_df, config)
    plot_detailed_analysis(
        results_df,
        model_artifacts.results,
        best_model,
        data_bundle,
        feature_artifacts,
        augmentation_data,
        model_artifacts,
        config,
    )
    plot_comprehensive_analysis(
        results_df,
        best_model,
        model_artifacts.results,
        feature_artifacts,
        augmentation_data,
        class_names,
        config,
    )

    report_path = export_best_model_report(
        best_model,
        model_artifacts.results,
        augmentation_data,
        class_names,
        config,
    )
    print(f"✅ Detailed classification report saved to: {report_path}")

    summary_path = save_summary_json(
        data_bundle,
        feature_artifacts,
        augmentation_data,
        results_df,
        best_model,
        config,
    )
    print(f"✅ Analysis summary saved to: {summary_path}")

    print("\n✅ Problem 2 analysis with data augmentation completed!")
    print("\n📁 Generated files with data augmentation:")
    print(f"- Data files ({config.data_dir}/):")
    print("  - model_results.csv: Model performance metrics")
    print("  - model_results_augmented.csv: Model performance metrics (augmented) duplicate")
    print("  - detailed_classification_report_augmented.csv: Detailed classification report")
    print("  - analysis_summary_augmented.json: Complete analysis summary")
    print("\n- Visualization files (figs/问题2/):")
    print("  1. top_features_table.png")
    print("  2. feature_analysis.png")
    print("  3. data_augmentation_analysis.png")
    print("  4. data_augmentation_table.png")
    print("  5. confusion_matrices_augmented_models.png")
    if model_artifacts.nn_history_path:
        print("  6. neural_network_augmented_training.png")
    print("  7. model_performance_table.png")
    print("  8. detailed_analysis.png")
    print("  9. comprehensive_augmentation_analysis.png")

    print("\n🎯 Key Results:")
    factor = augmentation_data.X_train_augmented.shape[0] / augmentation_data.X_train.shape[0]
    print(
        f"  • Training set size: {augmentation_data.X_train.shape[0]} → {augmentation_data.X_train_augmented.shape[0]} ({factor:.1f}x augmentation)"
    )
    print(f"  • Test set size: {augmentation_data.X_test.shape[0]} (unchanged for fair evaluation)")
    print(f"  • Best model: {best_model}")
    print(f"  • Best accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")
    print(f"  • Best F1 score: {best_f1:.4f}")
    print("\n✨ All models trained on augmented data, tested on original test set for unbiased evaluation!")


if __name__ == "__main__":
    main()
