from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class PipelineConfig:
    data_candidates: Sequence[Path] = field(default_factory=lambda: (
        Path("data/processed/94feature.csv"),
        Path("../data/processed/94feature.csv"),
        Path("../../data/processed/94feature.csv"),
        Path("/home/yolo/mcm_project/data/processed/94feature.csv"),
    ))
    possible_targets: Sequence[str] = ("label_cls", "fault_type_orig")
    meta_columns: Sequence[str] = (
        "file",
        "filename",
        "fs_inferred",
        "fs_target",
        "rpm_mean",
        "fr_hz",
        "label_cls",
        "label_size_in",
        "label_load_hp",
        "label_or_pos",
        "fault_type_orig",
    )
    feature_variance_threshold: float = 1e-6
    max_top_features: int = 30
    min_top_features: int = 10
    train_test_split_seed: int = 42
    test_size: float = 0.2
    augmentation_window: float = 0.7
    augmentation_stride: float = 0.3
    augmentation_noise: float = 0.02
    scaler_mean_tol: float = 1e-5
    figure_dir: Path = Path("figs/问题2")
    data_dir: Path = Path("data/processed/问题2")

    def resolve_data_path(self) -> Path:
        for candidate in self.data_candidates:
            if candidate.exists():
                return candidate
        candidates = "\n".join(str(c) for c in self.data_candidates)
        raise FileNotFoundError(f"Feature data file 94feature.csv not found. Checked:\n{candidates}")


