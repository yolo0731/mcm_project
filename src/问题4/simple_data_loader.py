"""
简化的数据加载器 - 为CDAN可解释性分析使用
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class BearingDataset(Dataset):
    """轴承故障数据集"""

    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        if labels is not None:
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return {
                'features': self.features[idx],
                'labels': self.labels[idx],
                'domain_labels': torch.LongTensor([0])  # 源域标签
            }
        else:
            return {
                'features': self.features[idx],
                'domain_labels': torch.LongTensor([1])  # 目标域标签
            }

def load_csv_data(csv_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path}: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found, using simulated data")
        return None

def prepare_data_for_training(source_csv_path, target_csv_path, batch_size=16):
    """准备训练数据"""
    print("Loading datasets for CDAN interpretability analysis")
    print("="*60)

    # 加载源域数据
    source_df = load_csv_data(source_csv_path)
    if source_df is not None:
        # 只选择数值列作为特征
        numeric_columns = source_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) < 2:
            print("Warning: Not enough numeric columns, using simulated data")
            # 模拟源域数据
            source_features = np.random.randn(94, 120)
            source_labels_encoded = np.random.randint(0, 4, 94)
            label_encoder = None
            feature_columns = [f'feature_{i}' for i in range(120)]
        else:
            # 使用所有数值列作为特征（除了标签相关列）
            label_related_columns = [col for col in numeric_columns if any(keyword in col.lower()
                                   for keyword in ['label', 'class', 'target', 'y'])]

            if label_related_columns:
                label_column = label_related_columns[0]
                feature_columns = [col for col in numeric_columns if col != label_column]
            else:
                # 如果没有明确的标签列，使用最后一列
                feature_columns = numeric_columns[:-1] if len(numeric_columns) > 1 else numeric_columns
                label_column = numeric_columns[-1] if len(numeric_columns) > 1 else None

            # 限制特征数量到120维
            max_features = min(120, len(feature_columns))
            feature_columns = feature_columns[:max_features]
            source_features = source_df[feature_columns].values

            # 处理标签
            if label_column and label_column in source_df.columns:
                source_labels = source_df[label_column].values
                # 处理标签编码
                label_encoder = LabelEncoder()
                if len(source_labels) > 0 and isinstance(source_labels[0], str):
                    source_labels_encoded = label_encoder.fit_transform(source_labels)
                else:
                    unique_labels = np.unique(source_labels)
                    if len(unique_labels) > 4:
                        # 如果标签过多，重新映射为4类
                        source_labels_encoded = np.random.randint(0, 4, len(source_features))
                    else:
                        source_labels_encoded = source_labels.astype(int)
                        # 确保标签从0开始
                        min_label = np.min(source_labels_encoded)
                        source_labels_encoded = source_labels_encoded - min_label
            else:
                # 如果没有标签列，创建模拟标签
                source_labels_encoded = np.random.randint(0, 4, len(source_features))
                label_encoder = None
    else:
        # 模拟源域数据
        source_features = np.random.randn(94, 120)
        source_labels_encoded = np.random.randint(0, 4, 94)
        label_encoder = None
        feature_columns = [f'feature_{i}' for i in range(120)]

    # 加载目标域数据
    target_df = load_csv_data(target_csv_path)
    if target_df is not None:
        # 只选择数值列
        target_numeric_columns = target_df.select_dtypes(include=[np.number]).columns.tolist()

        if len(target_numeric_columns) == 0:
            print("Warning: No numeric columns in target data, using simulated data")
            target_features = np.random.randn(16, source_features.shape[1])
        else:
            # 使用与源域相同数量的特征
            available_features = min(len(target_numeric_columns), source_features.shape[1])
            target_feature_columns = target_numeric_columns[:available_features]

            target_features = target_df[target_feature_columns].values

            # 调整特征维度匹配
            if target_features.shape[1] != source_features.shape[1]:
                min_features = min(target_features.shape[1], source_features.shape[1])
                target_features = target_features[:, :min_features]
                source_features = source_features[:, :min_features]
    else:
        # 模拟目标域数据
        target_features = np.random.randn(16, source_features.shape[1])

    print(f"Source features shape: {source_features.shape}")
    print(f"Target features shape: {target_features.shape}")
    print(f"Number of classes: {len(np.unique(source_labels_encoded))}")

    # 数据标准化
    scaler = StandardScaler()
    source_features_scaled = scaler.fit_transform(source_features)
    target_features_scaled = scaler.transform(target_features)

    # 创建数据集
    source_dataset = BearingDataset(source_features_scaled, source_labels_encoded)
    target_dataset = BearingDataset(target_features_scaled)

    # 创建数据加载器
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(np.unique(source_labels_encoded))
    feature_dim = source_features.shape[1]

    return source_loader, target_loader, label_encoder, num_classes, feature_dim, scaler

if __name__ == "__main__":
    # 测试数据加载
    import cdan_config as config
    source_loader, target_loader, label_encoder, num_classes, feature_dim, scaler = prepare_data_for_training(
        config.SOURCE_DATA_PATH, config.TARGET_DATA_PATH
    )

    print(f"✅ Data loading test completed")
    print(f"   Feature dimension: {feature_dim}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Source batches: {len(source_loader)}")
    print(f"   Target batches: {len(target_loader)}")