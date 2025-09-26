"""Data handling utilities for Problem 3 domain adaptation experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import detrend
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .paths import ROOT
except ImportError:
    from paths import ROOT

LABEL_TO_INDEX: Dict[str, int] = {"N": 0, "B": 1, "IR": 2, "OR": 3, "Unknown": -1}
INDEX_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_INDEX.items()}

WINDOW_SIZE: int = 2048
STEP_SIZE: int = 768  # Balanced overlap: 62.5% instead of 75%


@dataclass(frozen=True)
class SampleItem:
    """Represents one sliding-window sample extracted from a MAT file."""

    file_key: str
    file_path: Path
    start: int
    label: Optional[int]
    domain: int
    file_id: int


class BearingDataset(Dataset):
    """Dataset returning sliding windows from bearing vibration signals."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        domain_label: int,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        normalize: bool = True,
        cache_signals: bool = True,
    ) -> None:
        if metadata.empty:
            raise ValueError("Metadata dataframe is empty.")
        self.window_size = window_size
        self.step_size = step_size
        self.domain_label = domain_label
        self.normalize = normalize
        self.cache_signals = cache_signals
        self._metadata = metadata.reset_index(drop=True)
        self._cache: Dict[Path, np.ndarray] = {}
        self._items: List[SampleItem] = []
        self._file_id_to_key: Dict[int, str] = {}
        self._labels: Optional[np.ndarray] = None
        self._has_labels: bool = False
        self._build_index()
        self._prepare_label_cache()

    def _build_index(self) -> None:
        file_groups = self._metadata.groupby("file")
        file_id = 0

        # First pass: analyze class distribution
        class_file_counts = {}
        for file_key, group in file_groups:
            label_value = self._infer_label(group)
            if label_value is not None:
                label_str = INDEX_TO_LABEL[label_value]
                class_file_counts[label_str] = class_file_counts.get(label_str, 0) + 1

        # Calculate balanced sampling strategy
        if class_file_counts:
            min_files = min(class_file_counts.values())
            # Ensure sufficient samples for minority classes, especially B and N
            max_windows_per_file = {
                'N': min(200, int(3000 // class_file_counts.get('N', 1))),  # More for N class
                'B': min(180, int(2500 // class_file_counts.get('B', 1))),  # Preserve B class
                'IR': min(120, int(2000 // class_file_counts.get('IR', 1))),
                'OR': min(100, int(1800 // class_file_counts.get('OR', 1))),
            }
        else:
            max_windows_per_file = {'N': 200, 'B': 180, 'IR': 120, 'OR': 100}

        for file_key, group in file_groups:
            file_path = _resolve_mat_path(file_key, self._metadata)
            signal = self._load_and_cache_signal(file_path, group)
            total_length = signal.shape[0]
            if total_length < self.window_size:
                continue

            label_value = self._infer_label(group)
            label_str = INDEX_TO_LABEL.get(label_value, 'Unknown') if label_value is not None else 'Unknown'

            # Generate all possible starts
            all_starts = list(range(0, total_length - self.window_size + 1, self.step_size))

            # Apply class-specific sampling
            max_windows = max_windows_per_file.get(label_str, 100)
            if len(all_starts) > max_windows:
                # Use stratified sampling to maintain diversity
                step = len(all_starts) // max_windows
                starts = all_starts[::step][:max_windows]
                # Add some random samples to maintain variability
                remaining_starts = [s for s in all_starts if s not in starts]
                if remaining_starts:
                    additional_samples = min(10, len(remaining_starts))
                    starts.extend(np.random.choice(remaining_starts, additional_samples, replace=False))
            else:
                starts = all_starts

            for start in starts:
                self._items.append(
                    SampleItem(
                        file_key=file_key,
                        file_path=file_path,
                        start=start,
                        label=label_value,
                        domain=self.domain_label,
                        file_id=file_id,
                    )
                )
            self._file_id_to_key[file_id] = file_key
            file_id += 1

    def _prepare_label_cache(self) -> None:
        labels = [item.label for item in self._items]
        if labels and all(label is not None for label in labels):
            self._labels = np.asarray(labels, dtype=np.int64)
            self._has_labels = True
        else:
            self._labels = None
            self._has_labels = False

    def _infer_label(self, group: pd.DataFrame) -> Optional[int]:
        if "label_cls" not in group.columns:
            return None
        labels = group["label_cls"].dropna().unique()
        if len(labels) == 0:
            return None
        if len(labels) > 1:
            raise ValueError(f"Multiple labels found for file group: {labels}")
        label = labels[0]
        mapped = LABEL_TO_INDEX.get(str(label))
        if mapped is None:
            raise KeyError(f"Label {label!r} not recognised.")
        return mapped

    def _load_and_cache_signal(self, file_path: Path, group: pd.DataFrame) -> np.ndarray:
        if self.cache_signals and file_path in self._cache:
            return self._cache[file_path]
        signal = _load_signal_from_mat(file_path, group)
        signal = detrend(signal, type="linear")
        if self.normalize:
            std = np.std(signal)
            if std > 0:
                signal = (signal - np.mean(signal)) / std
            else:
                signal = signal - np.mean(signal)
        if self.cache_signals:
            self._cache[file_path] = signal
        return signal

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self._items[index]
        signal = self._fetch_window(item)
        tensor = torch.from_numpy(signal).unsqueeze(0).float()
        label = -1 if item.label is None else item.label
        return {
            "signal": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "domain": torch.tensor(item.domain, dtype=torch.long),
            "file_id": torch.tensor(item.file_id, dtype=torch.long),
        }

    def file_id_to_key(self) -> Dict[int, str]:
        return dict(self._file_id_to_key)

    def has_labels(self) -> bool:
        return self._has_labels

    def labels(self) -> np.ndarray:
        if not self._has_labels or self._labels is None:
            raise ValueError("Dataset has no labels available.")
        return self._labels

    def class_counts(self, minlength: int = len(LABEL_TO_INDEX)) -> np.ndarray:
        labels = self.labels()
        counts = np.bincount(labels, minlength=minlength)
        return counts

    def sample_weights(self) -> torch.Tensor:
        labels = self.labels()
        counts = self.class_counts(minlength=max(labels.max() + 1, len(LABEL_TO_INDEX)))
        inv_freq = np.zeros_like(counts, dtype=np.float64)
        mask = counts > 0
        inv_freq[mask] = 1.0 / counts[mask]
        weights = inv_freq[labels]
        return torch.from_numpy(weights.astype(np.float32))

    def _fetch_window(self, item: SampleItem) -> np.ndarray:
        signal = self._cache.get(item.file_path)
        if signal is None:
            group = self._metadata[self._metadata["file"] == item.file_key]
            signal = self._load_and_cache_signal(item.file_path, group)
        start = item.start
        end = start + self.window_size
        return signal[start:end]


def compute_class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = counts.astype(np.float64)
    # Only use the first 4 classes (N, B, IR, OR), ignore Unknown (-1)
    actual_counts = counts[:4]
    actual_counts[actual_counts == 0] = 1.0
    total = actual_counts.sum()
    num_classes = len(actual_counts)
    weights = total / (num_classes * actual_counts)
    return torch.from_numpy(weights.astype(np.float32))


def make_balanced_sampler(dataset: BearingDataset) -> WeightedRandomSampler:
    if not dataset.has_labels():
        raise ValueError("Balanced sampling requires a labeled dataset.")
    sample_weights = dataset.sample_weights()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def _resolve_mat_path(file_str: str, metadata: pd.DataFrame) -> Path:
    path = Path(file_str)
    if path.is_file():
        return path.resolve()
    csv_dir = Path(metadata.attrs.get("csv_dir", ROOT / "data" / "processed"))
    candidate = (csv_dir / file_str).resolve()
    if candidate.is_file():
        return candidate
    candidate = (ROOT / file_str).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"MAT file not found for entry: {file_str}")


def _load_signal_from_mat(file_path: Path, group: pd.DataFrame) -> np.ndarray:
    preferred_key = None
    if "filename" in group.columns:
        names = group["filename"].dropna().unique()
        if len(names) == 1:
            preferred_key = str(names[0]).replace(".mat", "")
    mat = loadmat(file_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    ordered_keys = _prioritize_keys(keys, preferred_key)
    for key in ordered_keys:
        data = mat[key]
        if not isinstance(data, np.ndarray):
            continue
        if data.ndim == 2 and data.shape[1] == 1:
            return np.asarray(data[:, 0], dtype=np.float64)
        if data.ndim == 1:
            return np.asarray(data, dtype=np.float64)
    raise KeyError(f"No suitable time series found in {file_path}")


def _prioritize_keys(keys: Iterable[str], preferred: Optional[str]) -> List[str]:
    result: List[str] = []
    if preferred:
        for key in keys:
            if key == preferred or key.endswith(preferred):
                result.append(key)
    for pattern in ("DE_time", "_DE", "drive", "A", "B", "C", "X", "Y"):
        for key in keys:
            if key in result:
                continue
            if pattern.lower() in key.lower():
                result.append(key)
    for key in keys:
        if key not in result:
            result.append(key)
    return result


def load_metadata(csv_path: Path) -> pd.DataFrame:
    csv_path = csv_path.resolve()
    df = pd.read_csv(csv_path)
    df.attrs["csv_dir"] = csv_path.parent
    if "file" not in df.columns:
        raise KeyError("CSV must contain a 'file' column pointing to MAT files.")
    return df


def stratified_train_val_split(
    metadata: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    unique_df = metadata.drop_duplicates(subset=["file"])
    labels = unique_df["label_cls"].apply(lambda x: LABEL_TO_INDEX[str(x)])
    train_files, val_files = train_test_split(
        unique_df["file"], test_size=val_ratio, random_state=seed, stratify=labels
    )
    train_df = metadata[metadata["file"].isin(train_files)].copy()
    val_df = metadata[metadata["file"].isin(val_files)].copy()
    train_df.attrs["csv_dir"] = metadata.attrs.get("csv_dir")
    val_df.attrs["csv_dir"] = metadata.attrs.get("csv_dir")
    return train_df, val_df


def create_dataloaders(
    source_csv: Path,
    target_csv: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    val_ratio: float = 0.2,
    seed: int = 42,
    balance_source: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    source_meta = load_metadata(source_csv)
    train_meta, val_meta = stratified_train_val_split(source_meta, val_ratio=val_ratio, seed=seed)
    target_meta = load_metadata(target_csv)

    train_dataset = BearingDataset(train_meta, domain_label=0)
    val_dataset = BearingDataset(val_meta, domain_label=0, cache_signals=False)
    target_dataset = BearingDataset(target_meta, domain_label=1)

    if balance_source:
        sampler = make_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    class_weights = compute_class_weights(train_dataset.class_counts())
    return train_loader, val_loader, target_loader, class_weights


def create_inference_loader(
    target_csv: Path,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    target_meta = load_metadata(target_csv)
    dataset = BearingDataset(target_meta, domain_label=1)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
