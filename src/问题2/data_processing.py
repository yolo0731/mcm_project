from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from problem2_config import PipelineConfig


@dataclass
class DataBundle:
    df: pd.DataFrame
    target_col: str
    feature_columns: List[str]
    X: pd.DataFrame
    y: pd.Series
    X_numeric: pd.DataFrame
    numeric_columns: List[str]
    y_encoded: np.ndarray
    label_encoder: LabelEncoder
    class_names: List[str]


def load_feature_data(config: PipelineConfig) -> pd.DataFrame:
    data_path = config.resolve_data_path()
    print(f"✅ Loading data file: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    return df


def infer_target_column(df: pd.DataFrame, config: PipelineConfig) -> str:
    for col in config.possible_targets:
        if col in df.columns:
            print(f"Target column: {col}")
            return col
    print("⚠️ Target column not found, checking available columns...")
    preview = df.columns.tolist()[:10]
    print(f"Available columns (first 10): {preview}")
    raise ValueError("No suitable target column found")


def preprocess_features(df: pd.DataFrame, target_col: str, config: PipelineConfig) -> DataBundle:
    print("=== Data Preprocessing ===")
    feature_columns = [col for col in df.columns if col not in config.meta_columns]
    X = df[feature_columns].copy()
    y = df[target_col].copy()

    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_columns].copy()
    print(f"Number of numeric features: {len(numeric_columns)}")

    missing_count = X_numeric.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values, filling with median")
        X_numeric = X_numeric.fillna(X_numeric.median())
    else:
        print("✅ No missing values")

    inf_count = np.isinf(X_numeric.values).sum()
    if inf_count > 0:
        print(f"Found {inf_count} infinite values, replacing them")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(X_numeric.median())
    else:
        print("✅ No infinite values")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str).str.strip())
    class_names = list(label_encoder.classes_)
    print("\nClass encoding:", {cls: idx for idx, cls in enumerate(class_names)})
    print(f"Number of classes: {len(class_names)}")

    print("\nFeature statistics:")
    print(f"Feature mean range: [{X_numeric.mean().min():.6f}, {X_numeric.mean().max():.6f}]")
    print(f"Feature variance range: [{X_numeric.var().min():.6f}, {X_numeric.var().max():.6f}]")
    print("✅ Data preprocessing completed")

    return DataBundle(
        df=df,
        target_col=target_col,
        feature_columns=feature_columns,
        X=X,
        y=y,
        X_numeric=X_numeric,
        numeric_columns=numeric_columns,
        y_encoded=y_encoded,
        label_encoder=label_encoder,
        class_names=class_names,
    )

