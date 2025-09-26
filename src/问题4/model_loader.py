"""
模型加载器 - 从问题3加载预训练模型（匹配问题3的实际架构）
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import sys
sys.path.append('../问题3')
from pathlib import Path
import config

class GradientReversalFn(Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    """梯度反转模块包装器"""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

class FeatureExtractor(nn.Module):
    """多尺度1D CNN特征提取器（匹配问题3架构）"""

    def __init__(self, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=32, stride=8, padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=2),
        )
        branch_channels = 64
        kernel_sizes = (7, 15, 31)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(32, branch_channels, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            for k in kernel_sizes
        ])
        att_hidden = max(32, branch_channels // 2)
        self.attention = nn.Sequential(
            nn.Linear(branch_channels, att_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(att_hidden, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(branch_channels, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        stem_feat = self.stem(x)
        branch_vectors = []
        att_scores = []
        for branch in self.branches:
            feat = branch(stem_feat).squeeze(-1)
            branch_vectors.append(feat)
            att_scores.append(self.attention(feat))
        stacked = torch.stack(branch_vectors, dim=1)
        scores = torch.stack(att_scores, dim=1).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        fused = (stacked * weights).sum(dim=1)
        return self.projection(fused)

class LabelClassifier(nn.Module):
    """残差MLP分类器（匹配问题3架构）"""

    def __init__(self, in_features=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.BatchNorm1d(4),
        )

    def forward(self, x):
        hidden = self.block1(x)
        residual = hidden
        hidden = self.block2(hidden) + residual

        # 处理单样本的BatchNorm1d问题
        if hidden.size(0) == 1:
            return nn.functional.linear(hidden, self.out[0].weight, self.out[0].bias)
        else:
            return self.out(hidden)

class DomainClassifier(nn.Module):
    """域分类器（匹配问题3架构）"""

    def __init__(self, in_features=128, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, alpha=1.0):
        self.grl.alpha = float(alpha)
        reversed_feat = self.grl(x)
        return self.net(reversed_feat)

class SourceModel(nn.Module):
    """源域模型（匹配问题3架构）"""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_dim=feature_dim)
        self.label_classifier = LabelClassifier(in_features=feature_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.label_classifier(features)

class DANNModel(nn.Module):
    """DANN模型（匹配问题3架构）"""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_dim=feature_dim)
        self.label_classifier = LabelClassifier(in_features=feature_dim)
        self.domain_classifier = DomainClassifier(in_features=feature_dim)

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        label_logits = self.label_classifier(features)
        domain_logits = self.domain_classifier(features, alpha=alpha)
        return label_logits, domain_logits, features

    def load_source_weights(self, state_dict):
        """从源域模型加载权重"""
        own_state = self.state_dict()
        for key, value in state_dict.items():
            if key.startswith("feature_extractor") or key.startswith("label_classifier"):
                if key in own_state:
                    own_state[key].copy_(value)

def load_pretrained_models():
    """加载预训练模型"""
    print("🔧 Loading Pre-trained Models from Problem 3")
    print("=" * 60)

    device = config.DEVICE

    # 加载源域模型
    source_model = SourceModel().to(device)
    source_path = config.SOURCE_MODEL_PATH

    if Path(source_path).exists():
        try:
            checkpoint = torch.load(source_path, map_location=device)
            if 'model_state' in checkpoint:
                source_model.load_state_dict(checkpoint['model_state'])
                print(f"✅ Source model loaded from: {source_path}")
                print(f"   Training config: {checkpoint.get('config', 'N/A')}")
            else:
                source_model.load_state_dict(checkpoint)
                print(f"✅ Source model loaded from: {source_path}")
        except Exception as e:
            print(f"❌ Failed to load source model: {e}")
            source_model = None
    else:
        print(f"❌ Source model not found at: {source_path}")
        source_model = None

    # 加载DANN模型
    dann_model = DANNModel().to(device)
    dann_path = config.DANN_MODEL_PATH

    if Path(dann_path).exists():
        try:
            checkpoint = torch.load(dann_path, map_location=device)
            if 'model_state' in checkpoint:
                dann_model.load_state_dict(checkpoint['model_state'])
                print(f"✅ DANN model loaded from: {dann_path}")
                print(f"   Training config: {checkpoint.get('config', 'N/A')}")
            else:
                dann_model.load_state_dict(checkpoint)
                print(f"✅ DANN model loaded from: {dann_path}")
        except Exception as e:
            print(f"❌ Failed to load DANN model: {e}")
            dann_model = None
    else:
        print(f"❌ DANN model not found at: {dann_path}")
        dann_model = None

    return source_model, dann_model

if __name__ == "__main__":
    # 测试模型加载
    source_model, dann_model = load_pretrained_models()

    if source_model is not None:
        print(f"Source model parameters: {sum(p.numel() for p in source_model.parameters())}")

    if dann_model is not None:
        print(f"DANN model parameters: {sum(p.numel() for p in dann_model.parameters())}")