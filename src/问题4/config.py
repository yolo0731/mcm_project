"""
问题4配置文件 - 智能故障诊断系统综合评价与优化
"""
import torch
from pathlib import Path

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
SOURCE_DATA_PATH = "data/processed/94feature.csv"
TARGET_DATA_PATH = "data/processed/16feature.csv"

# 预训练模型路径 (从问题3)
SOURCE_MODEL_PATH = "data/processed/问题3/source_pretrained_model.pth"
DANN_MODEL_PATH = "data/processed/问题3/dann_model.pth"

# 输出路径
OUTPUT_DIR = "data/processed/问题4"
FIGS_DIR = "figs/问题4"
MODELS_DIR = "models/问题4"

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGS_DIR).mkdir(parents=True, exist_ok=True)
# Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)  # Removed - don't auto-create models directory

# 模型评价指标
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'confusion_matrix', 'roc_auc', 'cross_entropy_loss'
]

# 故障类型映射
FAULT_TYPES = {
    'B': 'Ball Fault',
    'IR': 'Inner Ring Fault',
    'N': 'Normal',
    'OR': 'Outer Ring Fault'
}

CLASS_NAMES = ['B', 'IR', 'N', 'OR']
NUM_CLASSES = 4

# 综合评价权重
EVALUATION_WEIGHTS = {
    'accuracy': 0.25,
    'robustness': 0.20,
    'generalization': 0.20,
    'domain_adaptation': 0.15,
    'computational_efficiency': 0.10,
    'interpretability': 0.10
}

# 优化参数
OPTIMIZATION_PARAMS = {
    'ensemble_methods': ['voting', 'bagging', 'boosting'],
    'feature_selection_methods': ['pca', 'lda', 'mutual_info'],
    'model_compression': ['pruning', 'quantization', 'distillation'],
    'augmentation_methods': ['noise', 'scaling', 'rotation']
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'font_size': 12,
    'colors': ['#2E8B57', '#DC143C', '#4169E1', '#FF8C00'],
    'style': 'seaborn-v0_8'
}