import os
import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
import feature_extraction as fe


def _ensure_absolute_file_paths(features_df, config):
    """Ensure the `file` column points to existing absolute paths."""
    if 'file' not in features_df.columns:
        return features_df

    df = features_df.copy()
    root_path = Path(config['ROOT']).resolve()
    filename_col = 'filename' if 'filename' in df.columns else None
    cache = {}

    def _locate(candidate_name):
        candidate_name = str(candidate_name).strip()
        if not candidate_name:
            return None
        if candidate_name in cache:
            return cache[candidate_name]
        matches = list(root_path.rglob(candidate_name))
        cache[candidate_name] = matches[0].resolve() if matches else None
        return cache[candidate_name]

    resolved_paths = []
    for idx, value in df['file'].items():
        stored_path = Path(str(value))
        if stored_path.is_absolute() and stored_path.exists():
            resolved_paths.append(str(stored_path.resolve()))
            continue

        filename_value = None
        if filename_col:
            filename_value = df.at[idx, filename_col]
        if filename_value:
            filename_value = Path(str(filename_value)).name
        else:
            filename_value = stored_path.name

        located = _locate(filename_value)
        if located and located.exists():
            resolved_paths.append(str(located))
        else:
            # Fallback: keep original path string
            resolved_paths.append(str(stored_path))

    df['file'] = resolved_paths
    return df

def process_data(config, force_rebuild=False):
    """Loads data, processes signals, and extracts features."""
    save_path = Path(config['SAVE_PATH'])
    should_rebuild = force_rebuild or not save_path.exists()
    
    if not should_rebuild:
        print(f"✓ 发现已存在的特征文件: {config['SAVE_PATH']}")
        print("正在加载现有特征文件...")
        features_df = pd.read_csv(save_path)
        features_df = _ensure_absolute_file_paths(features_df, config)
        print(f"已加载特征数据集，形状: {features_df.shape}")
        if 'label_cls' in features_df.columns:
            print(f"Fault Type Distribution:")
            print(features_df['label_cls'].value_counts())
        print("✓ 使用现有特征文件，跳过特征提取过程")
        return features_df

    if force_rebuild:
        print("⚡ 强制重建模式：重新提取所有特征")
    else:
        print(f"未找到特征文件，开始处理...")

    try:
        selection_df = pd.read_excel(config['SELECTION_FILE'])
        print(f"从 {config['SELECTION_FILE']} 成功读取筛选文件列表")
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return pd.DataFrame()

    possible_filename_cols = ['文件名', 'filename', 'file_name', '文件', 'File']
    filename_col = None
    for col in possible_filename_cols:
        if col in selection_df.columns:
            filename_col = col
            break
    
    if filename_col is None:
        print(f"错误：未找到文件名列。可用列: {list(selection_df.columns)}")
        return pd.DataFrame()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    root_path = Path(config['ROOT'])
    
    total_files = len(selection_df)
    processed_count = 0
    
    for idx, row in selection_df.iterrows():
        processed_count += 1
        filename = str(row[filename_col]).strip()
        if not filename.lower().endswith('.mat'):
            filename += '.mat'
        
        print(f"[{processed_count}/{total_files}] 处理文件: {filename}")
        
        mat_files = list(root_path.rglob(filename))
        if not mat_files:
            print(f"  [未找到] {filename}")
            continue
        mat_file = mat_files[0]
        
        try:
            mat_data = scipy.io.loadmat(str(mat_file))
            signal_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            
            if not signal_keys:
                print(f"  [无信号] {filename}")
                continue
            
            fs_in = fe.infer_fs_from_path(str(mat_file), config['DEFAULT_FS_GUESS'])
            
            rpm_val = None
            rpm_cols = ['rpm', 'RPM', '转速', 'speed']
            for rpm_col in rpm_cols:
                if rpm_col in row and pd.notnull(row[rpm_col]):
                    rpm_val = float(row[rpm_col])
                    break
            
            if rpm_val is None:
                for key in mat_data.keys():
                    if 'rpm' in key.lower():
                        rpm_data = mat_data[key]
                        if hasattr(rpm_data, 'flatten'):
                            rpm_array = rpm_data.flatten()
                            rpm_val = float(rpm_array[0]) if len(rpm_array) > 0 else None
                        break
            
            fr_hz = rpm_val / 60.0 if rpm_val is not None else np.nan
            meta = fe.parse_label_from_name(filename, config['LABEL_RE'])
            
            row_data = {
                "file": str(mat_file.resolve()),
                "filename": filename,
                "fs_inferred": fs_in,
                "fs_target": config['TARGET_FS'],
                "rpm_mean": rpm_val,
                "fr_hz": fr_hz,
                "label_cls": meta["cls"],
                "label_size_in": meta["size_in"],
                "label_load_hp": meta["load_hp"],
                "label_or_pos": meta["or_pos"],
            }
            
            processed_signals = 0
            for sig_key in signal_keys:
                signal_data = mat_data[sig_key].flatten()
                
                if len(signal_data) < max(64, int(0.2*fs_in)):
                    continue
                
                sig_key_upper = sig_key.upper()
                if "DE" in sig_key_upper:
                    channel = "DE"
                elif "FE" in sig_key_upper:
                    channel = "FE"
                elif "BA" in sig_key_upper:
                    channel = "BA"
                else:
                    channel = "UNKNOWN"
                
                try:
                    x_processed = fe.preprocess_signal(signal_data, fs_in, config['TARGET_FS'], config['BP_LOW'], config['BP_HIGH'], config['FILTER_ORDER'])
                    feats_time = fe.time_features(x_processed)
                    feats_freq = fe.spectral_features(x_processed, config['TARGET_FS'])
                    feats_env = fe.envelope_features(x_processed, config['TARGET_FS'])
                    
                    _, env_mag, fvec = fe.envelope_spectrum(x_processed, config['TARGET_FS'])
                    
                    prefix = f"{channel}_"
                    row_data.update({f"{prefix}{k}": v for k,v in feats_time.items()})
                    row_data.update({f"{prefix}{k}": v for k,v in feats_freq.items()})
                    row_data.update({f"{prefix}{k}": v for k,v in feats_env.items()})
                    
                    if channel in ("DE", "FE") and np.isfinite(fr_hz) and fr_hz > 0:
                        geom = fe.bearing_freqs(fr_hz, config['GEOM'][channel]["Nd"], config['GEOM'][channel]["d"], config['GEOM'][channel]["D"])
                        aligned = fe.freq_aligned_indicators(env_mag, fvec, geom["fr"], geom, prefix=prefix)
                        row_data.update(aligned)
                    
                    processed_signals += 1
                except Exception as e:
                    print(f"    [特征提取错误] {sig_key}: {e}")
                    continue
            
            if processed_signals > 0:
                rows.append(row_data)
        except Exception as e:
            print(f"  [文件加载错误] {filename}: {e}")
            continue
            
    if len(rows) > 0:
        features_df = pd.DataFrame(rows)
        features_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✓ 特征已保存到: {save_path}")
        return features_df
    else:
        print("❌ 未提取到任何特征！")
        return pd.DataFrame()
