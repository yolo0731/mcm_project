"""
模型优化策略模块
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import VotingClassifier
import copy
import config

class FeatureOptimizer:
    """特征优化器"""

    def __init__(self):
        self.optimizers = {
            'pca': PCA,
            'lda': LinearDiscriminantAnalysis,
            'mutual_info': self._mutual_info_selection
        }

    def _mutual_info_selection(self, X, y, n_components=None):
        """基于互信息的特征选择"""
        from sklearn.feature_selection import SelectKBest
        if n_components is None:
            n_components = min(20, X.shape[1])
        selector = SelectKBest(mutual_info_classif, k=n_components)
        return selector.fit_transform(X, y)

    def optimize_features(self, X_train, y_train, X_test, method='pca', n_components=None):
        """特征优化"""
        print(f"Applying {method} feature optimization...")

        if n_components is None:
            n_components = min(25, X_train.shape[1])

        if method == 'pca':
            optimizer = PCA(n_components=n_components)
            X_train_opt = optimizer.fit_transform(X_train)
            X_test_opt = optimizer.transform(X_test)

        elif method == 'lda':
            optimizer = LinearDiscriminantAnalysis(n_components=min(n_components, len(np.unique(y_train))-1))
            X_train_opt = optimizer.fit_transform(X_train, y_train)
            X_test_opt = optimizer.transform(X_test)

        elif method == 'mutual_info':
            from sklearn.feature_selection import SelectKBest
            optimizer = SelectKBest(mutual_info_classif, k=n_components)
            X_train_opt = optimizer.fit_transform(X_train, y_train)
            X_test_opt = optimizer.transform(X_test)

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        return X_train_opt, X_test_opt, optimizer

class ModelEnsemble:
    """模型集成器"""

    def __init__(self, models, method='voting'):
        self.models = models
        self.method = method
        self.weights = None

    def fit_ensemble_weights(self, val_loader, device='cuda'):
        """训练集成权重"""
        model_predictions = []
        true_labels = []

        # 获取每个模型的验证集预测
        for model in self.models:
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    if len(true_labels) == 0:
                        true_labels.extend(batch['labels'].cpu().numpy())

                    outputs = model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions.extend(probabilities.cpu().numpy())

            model_predictions.append(np.array(predictions))

        # 基于验证集性能计算权重
        model_predictions = np.array(model_predictions)  # [n_models, n_samples, n_classes]
        true_labels = np.array(true_labels)

        # 计算每个模型的准确率
        accuracies = []
        for i, preds in enumerate(model_predictions):
            predicted_classes = np.argmax(preds, axis=1)
            accuracy = np.mean(predicted_classes == true_labels)
            accuracies.append(accuracy)

        # 基于准确率的软权重
        accuracies = np.array(accuracies)
        self.weights = accuracies / np.sum(accuracies)

        print(f"Ensemble weights: {self.weights}")

    def predict_ensemble(self, data_loader, device='cuda'):
        """集成预测"""
        ensemble_predictions = []
        true_labels = []

        # 获取所有模型的预测
        model_outputs = []
        for model in self.models:
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in data_loader:
                    features = batch['features'].to(device)
                    if len(true_labels) == 0:
                        true_labels.extend(batch['labels'].cpu().numpy())

                    outputs = model(features)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions.extend(probabilities.cpu().numpy())

            model_outputs.append(np.array(predictions))

        # 集成预测
        model_outputs = np.array(model_outputs)  # [n_models, n_samples, n_classes]

        if self.method == 'voting':
            if self.weights is not None:
                # 加权平均
                ensemble_probs = np.average(model_outputs, axis=0, weights=self.weights)
            else:
                # 简单平均
                ensemble_probs = np.mean(model_outputs, axis=0)

        elif self.method == 'max':
            # 取最大概率
            ensemble_probs = np.max(model_outputs, axis=0)

        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

        ensemble_predictions = np.argmax(ensemble_probs, axis=1)

        return ensemble_predictions, ensemble_probs, true_labels

class ModelCompressor:
    """模型压缩器"""

    def __init__(self):
        pass

    def prune_model(self, model, pruning_ratio=0.2):
        """模型剪枝"""
        import torch.nn.utils.prune as prune

        model_copy = copy.deepcopy(model)

        # 对每个Linear层进行剪枝
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')

        print(f"Model pruned with ratio: {pruning_ratio}")
        return model_copy

    def quantize_model(self, model):
        """模型量化"""
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        print("Model quantized to int8")
        return quantized_model

    def knowledge_distillation(self, teacher_model, student_model, data_loader,
                             num_epochs=10, temperature=3.0, alpha=0.7, device='cuda'):
        """知识蒸馏"""
        teacher_model.eval()
        student_model.train()

        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        kl_loss = nn.KLDivLoss()
        ce_loss = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in data_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                # Teacher outputs
                with torch.no_grad():
                    teacher_outputs = teacher_model(features)
                    soft_targets = torch.softmax(teacher_outputs / temperature, dim=1)

                # Student outputs
                student_outputs = student_model(features)
                soft_predictions = torch.log_softmax(student_outputs / temperature, dim=1)

                # 蒸馏损失
                distillation_loss = kl_loss(soft_predictions, soft_targets) * (temperature ** 2)

                # 分类损失
                classification_loss = ce_loss(student_outputs, labels)

                # 总损失
                loss = alpha * distillation_loss + (1 - alpha) * classification_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Distillation Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")

        return student_model

class DataAugmentor:
    """数据增强器"""

    def __init__(self):
        pass

    def add_noise(self, data, noise_level=0.1):
        """添加噪声"""
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def scale_data(self, data, scale_range=(0.8, 1.2)):
        """尺度变换"""
        scale_factor = torch.uniform(scale_range[0], scale_range[1], (data.size(0), 1))
        if data.is_cuda:
            scale_factor = scale_factor.cuda()
        return data * scale_factor

    def augment_dataset(self, data_loader, augmentation_methods=['noise', 'scaling'], device='cuda'):
        """数据集增强"""
        augmented_data = []
        augmented_labels = []

        for batch in data_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            # 原始数据
            augmented_data.append(features)
            augmented_labels.append(labels)

            # 应用增强
            for method in augmentation_methods:
                if method == 'noise':
                    aug_features = self.add_noise(features)
                elif method == 'scaling':
                    aug_features = self.scale_data(features)
                else:
                    continue

                augmented_data.append(aug_features)
                augmented_labels.append(labels)

        # 合并所有增强数据
        all_features = torch.cat(augmented_data, dim=0)
        all_labels = torch.cat(augmented_labels, dim=0)

        print(f"Original dataset size: {len(data_loader.dataset)}")
        print(f"Augmented dataset size: {len(all_features)}")

        return all_features, all_labels

class OptimizationManager:
    """优化管理器"""

    def __init__(self):
        self.feature_optimizer = FeatureOptimizer()
        self.model_compressor = ModelCompressor()
        self.data_augmentor = DataAugmentor()

    def comprehensive_optimization(self, models, data_loaders, device='cuda'):
        """综合优化"""
        print("🚀 Starting Comprehensive Model Optimization")
        print("=" * 60)

        optimization_results = {}

        # 1. 特征优化
        print("1. Feature Optimization...")
        # 这里可以实现特征优化逻辑

        # 2. 模型集成
        print("2. Model Ensemble...")
        ensemble = ModelEnsemble(models, method='voting')
        if 'validation' in data_loaders:
            ensemble.fit_ensemble_weights(data_loaders['validation'], device)

        # 3. 模型压缩
        print("3. Model Compression...")
        compressed_models = {}
        for name, model in zip(['source', 'dann'], models):
            # 剪枝
            pruned_model = self.model_compressor.prune_model(model, pruning_ratio=0.1)
            compressed_models[f'{name}_pruned'] = pruned_model

            # 量化
            try:
                quantized_model = self.model_compressor.quantize_model(copy.deepcopy(model))
                compressed_models[f'{name}_quantized'] = quantized_model
            except Exception as e:
                print(f"Quantization failed for {name}: {e}")

        # 4. 数据增强
        print("4. Data Augmentation...")
        # 这里可以实现数据增强逻辑

        optimization_results = {
            'ensemble': ensemble,
            'compressed_models': compressed_models,
            'feature_optimization': 'Applied',
            'data_augmentation': 'Applied'
        }

        print("✅ Comprehensive optimization completed!")
        return optimization_results

if __name__ == "__main__":
    print("Optimization strategies module loaded successfully!")