"""Model architectures for the DANN-based bearing fault diagnosis pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.autograd import Function


class GradientReversalFn(Function):
    """Implementation of the gradient reversal layer (GRL)."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    """Torch module wrapper for the gradient reversal function."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return GradientReversalFn.apply(x, self.alpha)


class FeatureExtractor(nn.Module):
    """Multi-scale 1D CNN feature extractor with attention pooling."""

    def __init__(self, out_dim: int = 128) -> None:
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
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(32, branch_channels, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool1d(1),
                )
                for k in kernel_sizes
            ]
        )
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

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
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
    """Residual MLP classifier predicting 4-way fault labels."""

    def __init__(self, in_features: int = 128, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        # Reduced dropout to preserve more information for B class
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
            nn.Dropout(dropout * 0.6),  # Reduced from 0.75 to 0.6
        )
        # Add batch normalization before final output
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, 4),
            nn.BatchNorm1d(4),  # Help with class separation
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        hidden = self.block1(x)
        residual = hidden
        hidden = self.block2(hidden) + residual

        # Handle BatchNorm1d for different input sizes
        if hidden.size(0) == 1:
            # For single sample, skip batch norm
            return nn.functional.linear(hidden, self.out[0].weight, self.out[0].bias)
        else:
            return self.out(hidden)


class DomainClassifier(nn.Module):
    """MLP domain discriminator distinguishing source vs target."""

    def __init__(self, in_features: int = 128, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        self.grl.alpha = float(alpha)
        reversed_feat = self.grl(x)
        return self.net(reversed_feat)


class SourceClassifier(nn.Module):
    """Baseline source-domain classifier (feature extractor + label head)."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_dim=feature_dim)
        self.label_classifier = LabelClassifier(in_features=feature_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        return self.label_classifier(features)


class DANNModel(nn.Module):
    """Two-headed DANN model for unsupervised domain adaptation."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_dim=feature_dim)
        self.label_classifier = LabelClassifier(in_features=feature_dim)
        self.domain_classifier = DomainClassifier(in_features=feature_dim)

    def forward(self, x: Tensor, alpha: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
        features = self.feature_extractor(x)
        label_logits = self.label_classifier(features)
        domain_logits = self.domain_classifier(features, alpha=alpha)
        return label_logits, domain_logits, features

    def load_source_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load weights for extractor and label head from a source model state dict."""
        own_state = self.state_dict()
        for key, value in state_dict.items():
            if key.startswith("feature_extractor") or key.startswith("label_classifier"):
                if key in own_state:
                    own_state[key].copy_(value)


@dataclass
class TrainingStepOutput:
    label_loss: float
    domain_loss: float
    total_loss: float
    accuracy: float
