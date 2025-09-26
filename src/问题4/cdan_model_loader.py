"""
CDAN模型加载器 - 基于条件域对抗网络的轴承故障诊断
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cdan_config as config

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class FeatureExtractor(nn.Module):
    """特征提取器 G(·)"""

    def __init__(self, input_dim=120, output_dim=128):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),

            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.net(x)

class LabelClassifier(nn.Module):
    """标签分类器 C(·)"""

    def __init__(self, feature_dim=128, num_classes=4):
        super(LabelClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, features):
        return self.net(features)

class ConditionalMapping(nn.Module):
    """条件映射层 T⊗(f,h) = f ⊗ h^T"""

    def __init__(self, feature_dim=128, num_classes=4):
        super(ConditionalMapping, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, features, predictions):
        """
        Args:
            features: [batch_size, feature_dim]
            predictions: [batch_size, num_classes]

        Returns:
            conditional_features: [batch_size, feature_dim * num_classes]
        """
        batch_size = features.size(0)

        # 软化预测分布
        h = F.softmax(predictions, dim=1)  # [batch_size, num_classes]

        # 条件映射: f ⊗ h^T
        # features: [batch_size, feature_dim, 1]
        # h: [batch_size, 1, num_classes]
        f_expanded = features.unsqueeze(2)  # [batch_size, feature_dim, 1]
        h_expanded = h.unsqueeze(1)         # [batch_size, 1, num_classes]

        # 外积操作
        conditional_tensor = torch.bmm(f_expanded, h_expanded)  # [batch_size, feature_dim, num_classes]

        # 展平为向量
        conditional_features = conditional_tensor.view(batch_size, -1)  # [batch_size, feature_dim * num_classes]

        return conditional_features, conditional_tensor

class DomainDiscriminator(nn.Module):
    """域判别器 D(·)"""

    def __init__(self, input_dim=512, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, conditional_features, alpha=1.0):
        """
        Args:
            conditional_features: 条件特征 [batch_size, feature_dim * num_classes]
            alpha: 梯度反转强度
        """
        reversed_features = GradientReversalFunction.apply(conditional_features, alpha)
        return self.net(reversed_features)

class CDANModel(nn.Module):
    """完整的CDAN模型"""

    def __init__(self,
                 input_dim=120,
                 feature_dim=128,
                 num_classes=4,
                 domain_hidden_dim=256):
        super(CDANModel, self).__init__()

        self.feature_extractor = FeatureExtractor(input_dim, feature_dim)
        self.label_classifier = LabelClassifier(feature_dim, num_classes)
        self.conditional_mapping = ConditionalMapping(feature_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(
            feature_dim * num_classes,
            domain_hidden_dim
        )

        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def forward(self, x, alpha=None, return_features=False):
        """
        Args:
            x: 输入特征
            alpha: 梯度反转强度
            return_features: 是否返回中间特征
        """
        # 特征提取
        features = self.feature_extractor(x)

        # 标签预测
        class_logits = self.label_classifier(features)

        if alpha is not None:
            # 条件映射
            conditional_features, conditional_tensor = self.conditional_mapping(features, class_logits)

            # 域判别
            domain_logits = self.domain_discriminator(conditional_features, alpha)

            if return_features:
                return {
                    'class_logits': class_logits,
                    'domain_logits': domain_logits,
                    'features': features,
                    'conditional_features': conditional_features,
                    'conditional_tensor': conditional_tensor
                }
            else:
                return class_logits, domain_logits
        else:
            if return_features:
                return {
                    'class_logits': class_logits,
                    'features': features
                }
            else:
                return class_logits

def load_cdan_model(model_path=None):
    """加载预训练的CDAN模型"""
    print("🔧 Loading CDAN Model for Interpretability Analysis")
    print("=" * 60)

    device = config.DEVICE

    # 创建模型实例
    model = CDANModel(
        input_dim=120,  # 基于现有特征数量
        feature_dim=config.CDAN_CONFIG['feature_dim'],
        num_classes=config.CDAN_CONFIG['num_classes'],
        domain_hidden_dim=config.CDAN_CONFIG['domain_hidden_dim']
    ).to(device)

    if model_path and Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ CDAN model loaded from: {model_path}")
                if 'config' in checkpoint:
                    print(f"   Training config: {checkpoint['config']}")
            else:
                # 尝试直接加载状态字典
                model.load_state_dict(checkpoint)
                print(f"✅ CDAN model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load CDAN model: {e}")
            print("   Using randomly initialized model for demonstration")
    else:
        print("⚠️  No pretrained CDAN model found, using randomly initialized model")
        print("   Note: For real analysis, you need a trained CDAN model from Problem 3")

    return model

def analyze_conditional_mapping(model, features, predictions):
    """分析条件映射T⊗(f,h)的物理意义"""
    model.eval()
    with torch.no_grad():
        conditional_features, conditional_tensor = model.conditional_mapping(features, predictions)

        # 计算类别选择性指标
        batch_size, feature_dim, num_classes = conditional_tensor.shape

        # S_k = |T⊗(:,k)|_2 / Σ_j|T⊗(:,j)|_2
        class_selectivity = {}
        for k in range(num_classes):
            class_norm = torch.norm(conditional_tensor[:, :, k], dim=1)  # [batch_size]
            total_norm = torch.norm(conditional_tensor.view(batch_size, -1), dim=1)  # [batch_size]
            selectivity = (class_norm / (total_norm + 1e-8)).mean().item()
            class_selectivity[f'Class_{k}'] = selectivity

        # 计算频率-类别关联度矩阵
        # R_ij = T⊗(i,j) / max(T⊗)
        max_val = torch.max(torch.abs(conditional_tensor))
        correlation_matrix = (conditional_tensor / (max_val + 1e-8)).mean(dim=0)  # [feature_dim, num_classes]

        return {
            'conditional_features': conditional_features,
            'conditional_tensor': conditional_tensor,
            'class_selectivity': class_selectivity,
            'correlation_matrix': correlation_matrix
        }

if __name__ == "__main__":
    # 测试模型加载和条件映射分析
    model = load_cdan_model()
    print(f"✅ CDAN model architecture:")
    print(f"   Feature extractor: {sum(p.numel() for p in model.feature_extractor.parameters())} parameters")
    print(f"   Label classifier: {sum(p.numel() for p in model.label_classifier.parameters())} parameters")
    print(f"   Conditional mapping: {sum(p.numel() for p in model.conditional_mapping.parameters())} parameters")
    print(f"   Domain discriminator: {sum(p.numel() for p in model.domain_discriminator.parameters())} parameters")

    # 测试条件映射
    test_input = torch.randn(10, 120).to(config.DEVICE)
    test_features = model.feature_extractor(test_input)
    test_predictions = model.label_classifier(test_features)

    analysis_result = analyze_conditional_mapping(model, test_features, test_predictions)
    print(f"✅ Conditional mapping analysis completed")
    print(f"   Class selectivity: {analysis_result['class_selectivity']}")
    print(f"   Correlation matrix shape: {analysis_result['correlation_matrix'].shape}")