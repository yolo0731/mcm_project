import sys
import data_processing
import visualization
import advanced_analysis
from pathlib import Path
import re


def _load_waveform_samples(project_root: Path, features_df):
    """Load representative DE channel signals for each fault type so plots match the notebook."""
    try:
        import scipy.io
    except ImportError as exc:
        print(f"Skipping waveform visualization (scipy missing): {exc}")
        return {}

    loaded = {}
    for fault_type in ["B", "IR", "OR", "N"]:
        candidates = features_df[features_df.get("label_cls") == fault_type]
        if candidates.empty:
            continue

        raw_path = Path(str(candidates.iloc[0]["file"]))
        if not raw_path.is_absolute():
            raw_path = (project_root / raw_path).resolve()

        if not raw_path.exists():
            print(f"  ✗ 无法找到 {fault_type} 对应的原始文件: {raw_path}")
            continue

        try:
            mat_data = scipy.io.loadmat(str(raw_path))
        except Exception as exc:
            print(f"  ✗ 无法加载 {fault_type} 波形: {exc}")
            continue

        de_keys = [k for k in mat_data.keys() if 'DE' in k.upper() and not k.startswith("__")]
        if not de_keys:
            print(f"  ✗ {fault_type} 文件中未找到DE测点信号")
            continue

        signal = mat_data[de_keys[0]].flatten()
        if signal.size == 0:
            print(f"  ✗ {fault_type} 信号为空")
            continue

        loaded[fault_type] = signal
        print(f"  ✓ 加载 {fault_type}: {de_keys[0]}, 长度={signal.size}")

    if not loaded:
        print("未成功加载任何波形信号，跳过波形对比图")
    else:
        print(f"共加载 {len(loaded)} 类故障用于波形对比")

    return loaded

def main():
    # Define configuration
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    try:
        from config import (
            FORCE_REBUILD_FEATURES,
            TARGET_FS,
            BP_LOW,
            BP_HIGH,
            FILTER_ORDER,
            BEARING_GEOM,
        )
    except ImportError:
        FORCE_REBUILD_FEATURES = False
        TARGET_FS = 32000
        BP_LOW, BP_HIGH = 500.0, 16000.0
        FILTER_ORDER = 4
        BEARING_GEOM = {
            "DE": {"Nd": 9, "d": 0.3126, "D": 1.537},
            "FE": {"Nd": 9, "d": 0.2656, "D": 1.122},
        }

    DEFAULT_FS_GUESS = 12000

    config = {
        "PROJECT_ROOT": project_root,
        "ROOT": project_root / "data/raw/源域数据集",
        "SELECTION_FILE": project_root / "data/processed/MAT数据详细清单.xlsx",
        "SAVE_PATH": project_root / "data/processed/94feature.csv",
        "TARGET_FS": TARGET_FS,
        "DEFAULT_FS_GUESS": DEFAULT_FS_GUESS,
        "BP_LOW": BP_LOW,
        "BP_HIGH": BP_HIGH,
        "FILTER_ORDER": FILTER_ORDER,
        "GEOM": BEARING_GEOM,
        "LABEL_RE": re.compile(
            r"(?P<cls>OR|IR|B|N)"
            r"(?P<size>\\d{3})?"
            r"(?:@(?P<pos>(3|6|12)))?"
            r"(?:_(?P<load>\\d))?",
            re.IGNORECASE
        ),
    }

    # Step 1: Process data and extract features
    force_rebuild = 'FORCE_REBUILD_FEATURES' in locals() and FORCE_REBUILD_FEATURES
    features_df = data_processing.process_data(config, force_rebuild=force_rebuild)

    if features_df.empty:
        print("No features were extracted. Exiting.")
        return

    # Step 2: Run visualizations
    figs_dir = project_root / "figs/问题1"
    figs_dir.mkdir(parents=True, exist_ok=True)

    visualization.plot_sampling_rate_distribution(features_df, figs_dir / "sampling_length_hist.png")
    visualization.plot_sample_distribution(features_df, figs_dir / "fault_type_count_bar_pie.png")
    waveform_signals = _load_waveform_samples(project_root, features_df)
    if waveform_signals:
        visualization.plot_signal_waveforms(waveform_signals, project_root / "data/processed/问题1/signal_waveforms_comparison.png")
    visualization.plot_rms_kurtosis_scatter(features_df, figs_dir / "rms_kurtosis_scatter.png")
    visualization.plot_feature_correlation_heatmap(features_df, figs_dir / "feature_correlation_heatmap.png")
    visualization.plot_load_stratified_distribution(features_df, figs_dir / "load_stratified_analysis.png")
    visualization.plot_fault_type_distribution(features_df, figs_dir / "fault_type_distribution.png")
    visualization.plot_rpm_distribution(features_df, figs_dir / "rpm_distribution.png")
    visualization.plot_vibration_mean_comparison(features_df, figs_dir / "vibration_mean_comparison.png")
    visualization.plot_vibration_std_comparison(features_df, figs_dir / "vibration_std_comparison.png")
    visualization.plot_vibration_amplitude_range(features_df, figs_dir / "vibration_amplitude_range.png")
    
    # Step 3: Perform advanced analysis
    advanced_analysis.perform_advanced_analysis(features_df, project_root=project_root, config=config)

if __name__ == "__main__":
    main()
