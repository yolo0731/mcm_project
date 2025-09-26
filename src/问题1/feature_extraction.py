import os, re, math, warnings
import numpy as np
import pandas as pd
from scipy import signal, stats
import scipy.io
from typing import Dict, List, Optional
from collections import Counter

def parse_label_from_name(filename: str, LABEL_RE) -> dict:
    """从文件名解析故障标签信息"""
    name = filename.upper()
    m = LABEL_RE.search(name)
    out = {"cls": None, "size_in": None, "load_hp": None, "or_pos": None}
    if m:
        cls = m.group("cls").upper()
        out["cls"] = cls
        size = m.group("size")
        out["size_in"] = float(size)/1000.0 if (size and cls in {"OR","IR","B"}) else None
        ld = m.group("load")
        out["load_hp"] = int(ld) if ld is not None else None
        pos = m.group("pos")
        out["or_pos"] = int(pos) if (pos and cls=="OR") else None
    return out

def infer_fs_from_path(path: str, default_fs) -> int:
    """从文件路径推断原始采样率"""
    s = str(path).lower()
    if re.search(r"(48k|48000)", s): return 48000
    if re.search(r"(12k|12000)", s): return 12000
    return default_fs

def butter_bandpass(low, high, fs, order=4):
    """设计巴特沃斯带通滤波器"""
    nyq = 0.5*fs
    low_n, high_n = max(1e-9, low/nyq), min(0.999999, high/nyq)
    if high_n <= low_n: high_n = min(0.999999, low_n*1.5)
    b, a = signal.butter(order, [low_n, high_n], btype='band')
    return b, a

def preprocess_signal(x: np.ndarray, fs_in: int, fs_out: int, BP_LOW, BP_HIGH, FILTER_ORDER) -> np.ndarray:
    """信号预处理：去趋势、滤波、重采样"""
    # 1. 线性去趋势
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    # 2. 带通滤波
    b, a = butter_bandpass(BP_LOW, min(BP_HIGH, 0.49*fs_in), fs=fs_in, order=FILTER_ORDER)
    x = signal.filtfilt(b, a, x, method="gust")
    # 3. 重采样到目标频率
    g = math.gcd(fs_in, fs_out)
    up, down = fs_out//g, fs_in//g
    return signal.resample_poly(x, up=up, down=down, padtype="line")

def time_features(x: np.ndarray) -> Dict[str, float]:
    """提取时域统计特征"""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return dict(mean=np.nan,std=np.nan,rms=np.nan,kurtosis=np.nan,skewness=np.nan,
                    peak=np.nan,crest_factor=np.nan,impulse_factor=np.nan,
                    shape_factor=np.nan,clearance_factor=np.nan)
    
    # 基本统计量
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if N > 1 else 0.0
    rms = float(np.sqrt(np.mean(x**2)))
    kurt = float(stats.kurtosis(x, fisher=False, bias=False)) if N > 3 else np.nan
    skew = float(stats.skew(x, bias=False)) if N > 2 else np.nan
    peak = float(np.max(np.abs(x)))
    
    # 无量纲指标
    mean_abs = float(np.mean(np.abs(x)))
    sqrt_mean_abs = float(np.sqrt(np.mean(np.sqrt(np.abs(x))**2)))
    
    cf = peak/(rms+1e-12)          # 峰值因子
    imp = peak/(mean_abs+1e-12)    # 脉冲因子
    shp = rms/(mean_abs+1e-12)     # 形状因子
    clr = peak/(sqrt_mean_abs+1e-12)  # 间隙因子
    
    return dict(mean=mean,std=std,rms=rms,kurtosis=kurt,skewness=skew,peak=peak,
                crest_factor=cf,impulse_factor=imp,shape_factor=shp,clearance_factor=clr)

def spectral_features(x: np.ndarray, fs: int) -> Dict[str, float]:
    """提取频域谱特征"""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    mag = np.abs(X)+1e-12
    ps = mag**2
    psn = ps/ps.sum()  # 归一化功率谱
    
    # 频域特征指标
    idx = int(np.argmax(mag))
    dom = float(freqs[idx])                               # 主频
    cent = float((freqs*psn).sum())                       # 谱质心
    sent = float(-(psn*np.log(psn)).sum())                # 谱熵
    bw = float(np.sqrt(((freqs-cent)**2*psn).sum()))      # 谱带宽
    
    return dict(dom_freq=dom,spec_centroid=cent,spec_entropy=sent,spec_bandwidth=bw)

def envelope_features(x: np.ndarray, fs: int) -> Dict[str, float]:
    """提取包络域特征"""
    env = np.abs(signal.hilbert(x))  # Hilbert包络
    env_rms = float(np.sqrt(np.mean(env**2)))
    env_kurt = float(stats.kurtosis(env, fisher=False, bias=False)) if len(env) > 3 else np.nan
    return dict(env_rms=env_rms, env_kurtosis=env_kurt)

def envelope_spectrum(x: np.ndarray, fs: int):
    """计算包络谱"""
    analytic = signal.hilbert(x)
    env = np.abs(analytic)
    e = env - np.mean(env)  # 去直流
    X = np.fft.rfft(e)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(e), d=1/fs)
    return env, mag, freqs

def bearing_freqs(fr_hz: float, Nd: int, d: float, D: float) -> dict:
    """计算轴承特征频率"""
    rho = d / D
    ftf = 0.5 * (1 - rho) * fr_hz
    bpfo = 0.5 * Nd * (1 - rho) * fr_hz
    bpfi = 0.5 * Nd * (1 + rho) * fr_hz
    bsf = (1 - rho**2) / (2*rho) * fr_hz
    return {"fr": fr_hz, "FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "rho": rho}

def band_metrics(freqs, mag, f0, delta=2.0):
    """计算特征频率附近的频带能量指标"""
    idx = np.where((freqs >= f0 - delta) & (freqs <= f0 + delta))[0]
    if idx.size == 0: return 0.0, 0.0
    peak = float(mag[idx].max())
    df = float(freqs[1]-freqs[0]) if len(freqs) > 1 else 1.0
    energy = float((mag[idx]**2).sum() * df)
    return peak, energy

def harmonic_energy(freqs, mag, f0, M=5, delta=2.0):
    """计算前M次倍频的谐波能量"""
    e = 0.0
    for m in range(1, M+1):
        _, ei = band_metrics(freqs, mag, m*f0, delta)
        e += ei
    return e

def sideband_energy(freqs, mag, f0, fr, M=5, Q=3, delta=2.0):
    """计算谐波周围的转频调制边带能量"""
    e = 0.0
    for m in range(1, M+1):
        base = m*f0
        for q in range(1, Q+1):
            for sign in (-1, +1):
                _, ei = band_metrics(freqs, mag, base + sign*q*fr, delta)
                e += ei
    return e

def freq_aligned_indicators(env_mag, freqs, fr, targets: dict,
                            delta=2.0, M=5, Q=3, prefix=""):
    """计算用于轴承故障诊断的频率对齐指标"""
    total_energy = float((env_mag**2).sum() * (freqs[1]-freqs[0] if len(freqs)>1 else 1.0))
    out = {}
    
    for key in ["FTF","BPFO","BPFI","BSF"]:
        f0 = targets[key]
        # 基础指标
        pk, be = band_metrics(freqs, env_mag, f0, delta)
        he = harmonic_energy(freqs, env_mag, f0, M, delta)
        sb = sideband_energy(freqs, env_mag, f0, fr, M, Q, delta)
        
        # 频率对齐指标集合
        out[f"{prefix}{key}_peak"] = pk                              # 峰值
        out[f"{prefix}{key}_bandE"] = be                             # 频带能量
        out[f"{prefix}{key}_Eratio"] = be / (total_energy + 1e-12)   # 能量占比
        out[f"{prefix}{key}_harmE_M{M}"] = he                        # 谐波能量
        out[f"{prefix}{key}_harmRatio_M{M}"] = he / (total_energy + 1e-12)  # 谐波能量占比
        out[f"{prefix}{key}_SB_Q{Q}"] = sb                           # 边带能量
        out[f"{prefix}{key}_SBI_Q{Q}"] = sb / (he + 1e-12)           # 边带调制指数
    
    return out

def sliding_window(signal_data, window_size, step_size):
    """滑窗分割信号"""
    windows = []
    n_samples = len(signal_data)
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal_data[start:end])
    return np.array(windows)

def extract_enhanced_features(window, fs=32000):
    """提取增强的多域特征"""
    features = {}
    # 基本统计特征
    features['mean'] = np.mean(window)
    features['std'] = np.std(window)
    features['var'] = np.var(window)
    features['rms'] = np.sqrt(np.mean(window**2))
    features['peak'] = np.max(np.abs(window))
    features['peak_to_peak'] = np.max(window) - np.min(window)
    # 形状特征
    features['skewness'] = stats.skew(window)
    features['kurtosis'] = stats.kurtosis(window)
    # 脉冲特征
    mean_abs = np.mean(np.abs(window))
    if mean_abs > 0 and features['rms'] > 0:
        features['crest_factor'] = features['peak'] / features['rms']
        features['impulse_factor'] = features['peak'] / mean_abs
        features['shape_factor'] = features['rms'] / mean_abs
        sqrt_mean = np.mean(np.sqrt(np.abs(window)))
        features['clearance_factor'] = features['peak'] / (sqrt_mean**2) if sqrt_mean > 0 else 0
    else:
        features['crest_factor'] = 0
        features['impulse_factor'] = 0
        features['shape_factor'] = 0
        features['clearance_factor'] = 0
    # 频域特征
    fft = np.fft.fft(window)
    fft_mag = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(window), 1/fs)[:len(fft)//2]
    if len(fft_mag) > 0 and np.sum(fft_mag) > 0:
        psd = fft_mag**2 / len(window)
        total_power = np.sum(psd)
        normalized_psd = psd / total_power if total_power > 0 else np.ones_like(psd) / len(psd)
        features['dominant_freq'] = freqs[np.argmax(fft_mag)]
        features['spectral_centroid'] = np.sum(freqs * normalized_psd)
        features['spectral_bandwidth'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * normalized_psd))
        # 谱熵
        normalized_psd_nonzero = normalized_psd[normalized_psd > 1e-10]
        features['spectral_entropy'] = -np.sum(normalized_psd_nonzero * np.log(normalized_psd_nonzero + 1e-10)) if len(normalized_psd_nonzero) > 0 else 0
    else:
        features['dominant_freq'] = 0
        features['spectral_centroid'] = 0
        features['spectral_bandwidth'] = 0
        features['spectral_entropy'] = 0
    # 包络特征
    try:
        analytic_signal = signal.hilbert(window)
        envelope = np.abs(analytic_signal)
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_rms'] = np.sqrt(np.mean(envelope**2))
        features['envelope_kurtosis'] = stats.kurtosis(envelope)
        features['envelope_peak'] = np.max(envelope)
        features['envelope_crest'] = features['envelope_peak'] / features['envelope_rms'] if features['envelope_rms'] > 0 else 0
    except:
        features['envelope_mean'] = 0
        features['envelope_std'] = 0
        features['envelope_rms'] = 0
        features['envelope_kurtosis'] = 0
        features['envelope_peak'] = 0
        features['envelope_crest'] = 0
    # 检查并处理异常值
    for key, value in features.items():
        if not np.isfinite(value):
            features[key] = 0.0
    return features