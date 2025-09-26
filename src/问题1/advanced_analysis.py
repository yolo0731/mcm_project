import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from pathlib import Path
import scipy.io
from scipy import signal, stats
from collections import Counter
import feature_extraction as fe

DEFAULT_GEOM = {
    "DE": {"Nd": 9, "d": 0.3126, "D": 1.537},
    "FE": {"Nd": 9, "d": 0.2656, "D": 1.122},
}


def _window_stats(values, prefix):
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_peak": 0.0,
            f"{prefix}_kurtosis": 0.0,
            f"{prefix}_skew": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_peak": float(np.max(values)),
        f"{prefix}_kurtosis": float(stats.kurtosis(values)),
        f"{prefix}_skew": float(stats.skew(values)),
    }


def _bearing_aligned_features(window, fs, fr_hz, geom_cfg, prefix="DE_"):
    features = {}
    try:
        _, env_mag, freq_vec = fe.envelope_spectrum(window, fs)
        if np.isfinite(fr_hz) and fr_hz > 0:
            geom = fe.bearing_freqs(fr_hz, geom_cfg["Nd"], geom_cfg["d"], geom_cfg["D"])
            aligned = fe.freq_aligned_indicators(env_mag, freq_vec, geom["fr"], geom, prefix=prefix)
            features.update(aligned)
    except Exception as exc:
        print(f"    [频率对齐特征失败] {exc}")
    return features

def perform_advanced_analysis(original_df, project_root=None, config=None):
    project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    fig_dir = project_root / 'figs/问题1'
    data_dir = project_root / 'data/processed/问题1'
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = config or {}
    target_fs = cfg.get('TARGET_FS', 32000)
    bp_low = cfg.get('BP_LOW', 500.0)
    bp_high = cfg.get('BP_HIGH', 16000.0)
    filter_order = cfg.get('FILTER_ORDER', 4)
    geom_cfg = cfg.get('GEOM', DEFAULT_GEOM)

    print("="*60)
    print("第十六步：改进的特征工程和2D可视化分析（数据平衡版）")
    print("="*60)

    # 1. 生成平衡的数据集
    print("1. 生成平衡的数据集...")
    print(f"原始数据分布:")
    print(original_df['label_cls'].value_counts())

    WINDOW_SIZE = 4096
    WINDOW_STEP = 1024
    TARGET_SAMPLES_PER_CLASS = 121

    all_features = []
    all_labels = []
    all_metadata = []

    for fault_type in ['OR', 'IR', 'B', 'N']:
        print(f"\n处理故障类型: {fault_type}")
        fault_files = original_df[original_df['label_cls'] == fault_type]

        if len(fault_files) == 0:
            print(f"  跳过{fault_type}: 无数据")
            continue

        candidate_windows = []
        candidate_meta = []
        window_scores = []

        for idx, row in fault_files.iterrows():
            try:
                mat_file = Path(row['file'])
                if not mat_file.is_absolute():
                    mat_file = (project_root / mat_file).resolve()
                if not mat_file.exists():
                    continue
                mat_data = scipy.io.loadmat(str(mat_file))
                de_keys = [k for k in mat_data.keys() if 'DE' in k.upper() and not k.startswith('__')]
                if not de_keys:
                    continue
                signal_data = mat_data[de_keys[0]].flatten()
                signal_data = signal.detrend(signal_data, type='linear')
                fs_in = row.get('fs_inferred', target_fs)
                fs_in = fs_in if np.isfinite(fs_in) and fs_in > 0 else target_fs

                processed = fe.preprocess_signal(signal_data, fs_in, target_fs, bp_low, bp_high, filter_order)
                env_signal = np.abs(signal.hilbert(processed))

                windows = fe.sliding_window(processed, WINDOW_SIZE, WINDOW_STEP)
                env_windows = fe.sliding_window(env_signal, WINDOW_SIZE, WINDOW_STEP)
                max_windows = min(40, len(windows))
                if len(windows) > max_windows:
                    indices = np.linspace(0, len(windows)-1, max_windows, dtype=int)
                    windows = windows[indices]
                    env_windows = env_windows[indices]

                for win_idx, (window, env_window) in enumerate(zip(windows, env_windows)):
                    base_feats = fe.extract_enhanced_features(window, fs=target_fs)
                    base_feats = {f"raw_{k}": v for k, v in base_feats.items()}

                    env_stats = _window_stats(env_window, "env")
                    ratios = {
                        'env_rms_ratio': env_stats['env_rms'] / (base_feats.get('raw_rms', 1e-6) + 1e-6),
                        'env_peak_ratio': env_stats['env_peak'] / (base_feats.get('raw_peak', 1e-6) + 1e-6),
                    }

                    fr_hz = row.get('fr_hz', np.nan)
                    aligned = _bearing_aligned_features(window, target_fs, fr_hz,
                                                       geom_cfg.get('DE', DEFAULT_GEOM['DE']))

                    combined = {**base_feats, **env_stats, **ratios, **aligned}
                    candidate_windows.append(combined)
                    candidate_meta.append({
                        'file': row['file'],
                        'window_idx': win_idx,
                        'fault_type': fault_type
                    })
                    window_scores.append(env_stats['env_kurtosis'])
            except Exception as e:
                print(f"  处理文件时出错: {e}")
                continue

        if not candidate_windows:
            print(f"  {fault_type}: 未生成有效窗口")
            continue

        print(f"  {fault_type}: 生成了 {len(candidate_windows)} 个窗口特征")

        order = np.argsort(window_scores)[::-1]
        candidate_windows = [candidate_windows[i] for i in order]
        candidate_meta = [candidate_meta[i] for i in order]

        if len(candidate_windows) > TARGET_SAMPLES_PER_CLASS:
            candidate_windows = candidate_windows[:TARGET_SAMPLES_PER_CLASS]
            candidate_meta = candidate_meta[:TARGET_SAMPLES_PER_CLASS]
        elif len(candidate_windows) < TARGET_SAMPLES_PER_CLASS:
            reps = TARGET_SAMPLES_PER_CLASS - len(candidate_windows)
            for i in range(reps):
                clone = candidate_windows[i % len(candidate_windows)].copy()
                for key, val in clone.items():
                    if isinstance(val, (int, float)):
                        clone[key] = float(val) + np.random.normal(0, 1e-3)
                candidate_windows.append(clone)
                candidate_meta.append(candidate_meta[i % len(candidate_meta)])

        all_features.extend(candidate_windows)
        all_labels.extend([fault_type] * len(candidate_windows))
        all_metadata.extend(candidate_meta)
        print(f"  {fault_type}: 平衡后 {len(candidate_windows)} 个样本")

    print(f"\n总共生成 {len(all_features)} 个平衡样本")
    balanced_counts = Counter(all_labels)
    print("平衡后的分布:")
    for label, count in balanced_counts.items():
        print(f"  {label}: {count}")

    if len(all_features) > 0:
        print("\n2. 特征工程...")
        features_df = pd.DataFrame(all_features)
        labels_array = np.array(all_labels)
        print(f"特征矩阵形状: {features_df.shape}")
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        feature_variance = features_df.var()
        valid_features = feature_variance > 1e-6
        features_df = features_df.loc[:, valid_features]
        print(f"移除常数特征后: {features_df.shape[1]} 个特征")
        k_best = min(80, features_df.shape[1])
        f_selector = SelectKBest(score_func=f_classif, k=k_best)
        X_f = f_selector.fit_transform(features_df, labels_array)

        mi_k = min(60, X_f.shape[1])
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=mi_k)
        X_selected = mi_selector.fit_transform(X_f, labels_array)

        selected_features = features_df.columns[f_selector.get_support()]
        selected_feature_names = selected_features[mi_selector.get_support()]
        print(f"特征选择后: {X_selected.shape[1]} 个特征")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        print("\n3. 降维分析...")
        unique_labels = np.unique(labels_array)
        n_classes = len(unique_labels)
        pca = PCA(n_components=min(20, X_scaled.shape[1]-1), random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        print(f"PCA降维: {X_scaled.shape[1]} -> {X_pca.shape[1]} 维")
        print(f"PCA累计方差解释比: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
        lda = LDA(n_components=min(n_classes-1, X_scaled.shape[1]))
        X_lda = lda.fit_transform(X_scaled, labels_array)
        print(f"LDA降维: {X_scaled.shape[1]} -> {X_lda.shape[1]} 维")

        try:
            qda = QDA(reg_param=1e-3)
            qda.fit(X_scaled, labels_array)
            X_qda = qda.predict_proba(X_scaled)
        except Exception as exc:
            print(f"QDA 训练失败: {exc}")
            X_qda = None

        print("\n4. 2D t-SNE可视化...")
        datasets = {'LDA': X_lda}
        if X_qda is not None:
            datasets['QDA-Proba'] = X_qda
        n_panels = len(datasets)
        fig_rows = 2
        fig_cols = int(np.ceil((n_panels + 1) / fig_rows))
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(6 * fig_cols, 5 * fig_rows))
        axes = axes.flatten()
        colors = {'OR': '#E74C3C', 'IR': '#3498DB', 'B': '#2ECC71', 'N': '#F39C12'}
        best_score = -1
        best_result = None
        best_method = None
        best_params = None

        tsne_grid = [
            {'perplexity': 8, 'learning_rate': 180, 'early_exaggeration': 16, 'metric': 'cosine', 'init': 'pca', 'random_state': 42},
            {'perplexity': 12, 'learning_rate': 220, 'early_exaggeration': 20, 'metric': 'euclidean', 'init': 'pca', 'random_state': 99},
            {'perplexity': 6, 'learning_rate': 150, 'early_exaggeration': 28, 'metric': 'cosine', 'init': 'random', 'random_state': 21},
            {'perplexity': 10, 'learning_rate': 260, 'early_exaggeration': 24, 'metric': 'euclidean', 'init': 'random', 'random_state': 55},
        ]

        for i, (method, X_data) in enumerate(datasets.items()):
            print(f"  执行 {method} + 2D t-SNE...")
            n_samples = X_data.shape[0]
            local_best_score = -1
            local_best_result = None
            local_best_params = None

            for params in tsne_grid:
                perplexity = min(params['perplexity'], max(3, n_samples // 6))
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    learning_rate=params['learning_rate'],
                    early_exaggeration=params['early_exaggeration'],
                    metric=params.get('metric', 'euclidean'),
                    max_iter=2000,
                    random_state=params.get('random_state', 42),
                    init=params.get('init', 'pca')
                )
                X_tsne = tsne.fit_transform(X_data)
                try:
                    score = silhouette_score(X_tsne, labels_array)
                except Exception:
                    score = -1
                print(f"    参数 {params} -> 轮廓系数: {score:.3f}")
                if score > local_best_score:
                    local_best_score = score
                    local_best_result = X_tsne.copy()
                    local_best_params = params

            if local_best_result is None:
                local_best_result = np.zeros((n_samples, 2))
                local_best_score = -1
                local_best_params = {'perplexity': None}

            if local_best_score > best_score:
                best_score = local_best_score
                best_result = local_best_result.copy()
                best_method = method
                best_params = local_best_params

            ax = axes[i]
            for label in unique_labels:
                mask = labels_array == label
                ax.scatter(local_best_result[mask, 0], local_best_result[mask, 1],
                          c=colors.get(label, 'gray'),
                          label=f'{label} ({np.sum(mask)})',
                          alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
            ax.set_title(f'{method} + 2D t-SNE\nBest Silhouette: {local_best_score:.3f}')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        if best_result is not None and n_panels < len(axes):
            ax = axes[n_panels]
            for label in unique_labels:
                mask = labels_array == label
                ax.scatter(best_result[mask, 0], best_result[mask, 1],
                          c=colors.get(label, 'gray'),
                          label=f'{label} ({np.sum(mask)})',
                          alpha=0.8, s=60, edgecolors='white', linewidth=1)
            param_text = f"perp={best_params.get('perplexity')} lr={best_params.get('learning_rate')} metric={best_params.get('metric')}"
            ax.set_title(f'Best Result: {best_method}\nSilhouette Score: {best_score:.3f}\n({param_text})',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        for j in range(n_panels + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle('2D t-SNE: Balanced Dataset Visualization', fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_dir / 'tsne_comparison_balanced.png', dpi=300, bbox_inches='tight')
        plt.close()

        if best_result is not None:
            print("\n5. Best result visualization...")
            print(f"Best embedding source: {best_method}, silhouette: {best_score:.3f}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            for label in unique_labels:
                mask = labels_array == label
                ax1.scatter(best_result[mask, 0], best_result[mask, 1],
                           c=colors.get(label, 'gray'),
                           label=f'{label} ({np.sum(mask)} samples)',
                           alpha=0.8, s=50, edgecolors='white', linewidth=0.8)
            ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
            ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
            ax1.set_title(f'2D t-SNE Visualization (Method: {best_method})\n' +
                         f'Balanced Dataset - Silhouette Score: {best_score:.3f}',
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(title='Fault Type', title_fontsize=12, fontsize=10)
            ax2.scatter(best_result[:, 0], best_result[:, 1],
                       c=[colors.get(label, 'gray') for label in labels_array],
                       alpha=0.6, s=30)
            ax2.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
            ax2.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
            ax2.set_title('Density View', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / 'tsne_best_result_balanced.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("\n6. 保存平衡数据集和结果...")
        balanced_df = pd.DataFrame(X_selected, columns=selected_feature_names)
        balanced_df['label'] = labels_array
        balanced_df.to_csv(data_dir / 'balanced_features.csv', index=False)
        print(f"✓ 平衡特征数据集已保存: {balanced_df.shape}")

        if best_result is not None:
            tsne_df = pd.DataFrame({
                'tsne_dim1': best_result[:, 0],
                'tsne_dim2': best_result[:, 1],
                'label': labels_array,
                'method': best_method,
                'silhouette_score': best_score
            })
            tsne_df.to_csv(data_dir / 'tsne_results_balanced.csv', index=False)
            print(f"✓ t-SNE结果已保存")

        print("\nStep sixteen complete!")
        print(f"✓ Balanced dataset created: {TARGET_SAMPLES_PER_CLASS} samples per class")
        print(f"✓ Best visualization: {best_method} (silhouette: {best_score:.3f})")
    else:
        print("❌ 未能生成足够的特征数据")
