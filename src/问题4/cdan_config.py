"""
基于CDAN模型的轴承故障迁移诊断可解释性分析 - 配置文件
"""
import torch
import os

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径配置
DATA_DIR = "data"
SOURCE_DATA_PATH = "data/processed/94feature.csv"
TARGET_DATA_PATH = "data/processed/16feature.csv"

# 模型路径配置 - 需要基于问题3的CDAN模型
MODEL_DIR = "models/问题4"
os.makedirs(MODEL_DIR, exist_ok=True)

# CDAN模型架构参数
CDAN_CONFIG = {
    'feature_dim': 128,  # 特征提取器输出维度
    'num_classes': 4,    # 故障类别数
    'conditional_dim': 512,  # 条件映射维度 d×K
    'domain_hidden_dim': 256,  # 域判别器隐藏层维度
    'dropout_rate': 0.5
}

# 轴承物理参数配置
BEARING_PARAMS = {
    'Z': 9,         # 滚动体数量
    'fr': 29.93,    # 转频 (Hz)
    'd': 0.3124,    # 滚动体直径 (inch)
    'D': 1.537,     # 节圆直径 (inch)
    'alpha': 0.0    # 接触角 (rad)
}

# 理论故障频率
FAULT_FREQUENCIES = {
    'BPFI': 107.54,   # 内圈故障频率
    'BPFO': 162.46,   # 外圈故障频率
    'BSF': 70.70,     # 滚动体故障频率
    'FTF': 11.95      # 保持架故障频率
}

# 输出路径配置
OUTPUT_DIR = "data/processed/问题4"
FIGS_DIR = "figs/问题4"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# 可解释性分析配置
INTERPRETABILITY_CONFIG = {
    'shap_samples': 100,
    'tsne_perplexity': 30,
    'visualization_epochs': [5, 10, 15, 20, 25, 30],
    'energy_threshold': 0.3,  # 物理验证能量比阈值
    'tolerance': 0.1,  # 频率容差
}

# 评估权重配置
EVALUATION_WEIGHTS = {
    'fidelity': 0.3,           # 保真度
    'stability': 0.25,         # 稳定性
    'comprehensiveness': 0.25, # 完整性
    'physical_reasonableness': 0.2  # 物理合理性
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'font_size': 12,
    'title_size': 14
}

print(f"✅ CDAN可解释性分析配置完成")
print(f"   设备: {DEVICE}")
print(f"   输出目录: {OUTPUT_DIR}")
print(f"   图片目录: {FIGS_DIR}")