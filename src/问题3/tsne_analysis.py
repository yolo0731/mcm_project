"""t-SNE analysis utilities for Problem 3."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import ROOT
    from .data import BearingDataset, INDEX_TO_LABEL, create_dataloaders, load_metadata
    from .models import DANNModel, SourceClassifier
    from .paths import ensure_output_dirs
    from .train_utils import set_seed
    from .visualization import plot_tsne_scatter
except ImportError:
    from paths import ROOT
    from data import BearingDataset, INDEX_TO_LABEL, create_dataloaders, load_metadata
    from models import DANNModel, SourceClassifier
    from paths import ensure_output_dirs
    from train_utils import set_seed
    from visualization import plot_tsne_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate t-SNE plots for feature alignment")
    parser.add_argument("--csv-source", type=Path, default=ROOT / "data" / "processed" / "94feature.csv")
    parser.add_argument("--csv-target", type=Path, default=ROOT / "data" / "processed" / "16feature.csv")
    parser.add_argument("--model-type", choices=["source", "dann"], default="dann")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figs-dir", type=Path, default=None)
    return parser.parse_args()


def _load_model(model_type: str, model_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    if model_type == "source":
        model = SourceClassifier()
    else:
        model = DANNModel()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _choose_model_path(model_type: str, processed_dir: Path, model_path: Path | None) -> Path:
    if model_path:
        return model_path
    default_name = "source_pretrained_model.pth" if model_type == "source" else "dann_model.pth"
    candidate = processed_dir / default_name
    if not candidate.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at {candidate}")
    return candidate


def _collect_features(
    model: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    domains_list: List[int] = []

    def _append_batch(batch: Dict[str, torch.Tensor], domain: int) -> None:
        signals = batch["signal"].to(device)
        with torch.no_grad():
            if isinstance(model, SourceClassifier):
                feats = model.feature_extractor(signals)
                logits = model.label_classifier(feats)
            else:
                logits, _, feats = model(signals, alpha=0.0)
            feats_np = feats.cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        features_list.append(feats_np)
        labels_list.extend(preds.tolist())
        domains_list.extend([domain] * feats_np.shape[0])

    total = 0
    for batch in source_loader:
        _append_batch(batch, domain=0)
        total += batch["signal"].size(0)
        if total >= max_samples // 2:
            break

    total_target = 0
    for batch in target_loader:
        _append_batch(batch, domain=1)
        total_target += batch["signal"].size(0)
        if total_target >= max_samples // 2:
            break

    features = np.concatenate(features_list, axis=0)
    labels = np.asarray(labels_list)
    domains = np.asarray(domains_list)
    return features, labels, domains


def plot_tsne(
    model_type: str = "dann",
    csv_source: Path | None = None,
    csv_target: Path | None = None,
    model_path: Path | None = None,
    figs_dir: Path | None = None,
    max_samples: int = 4000,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Path:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir, figs_dir_final = ensure_output_dirs(None, figs_dir)
    csv_source = csv_source or ROOT / "data" / "processed" / "94feature.csv"
    csv_target = csv_target or ROOT / "data" / "processed" / "16feature.csv"

    source_meta = load_metadata(csv_source)
    target_meta = load_metadata(csv_target)
    source_dataset = BearingDataset(source_meta, domain_label=0)
    target_dataset = BearingDataset(target_meta, domain_label=1)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    checkpoint_path = _choose_model_path(model_type, processed_dir, model_path)
    model = _load_model(model_type, checkpoint_path).to(device)

    features, labels, domains = _collect_features(model, source_loader, target_loader, device, max_samples)
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca")
    embedded = tsne.fit_transform(features)

    save_path = figs_dir_final / f"tsne_domain_class_sliding_{model_type}_en.png"
    plot_tsne_scatter(
        embedded,
        labels,
        domains,
        save_path,
        f"t-SNE ({model_type} model)",
    )
    return save_path


def main() -> None:
    args = parse_args()
    processed_dir, figs_dir = ensure_output_dirs(None, args.figs_dir)
    plot_tsne(
        model_type=args.model_type,
        csv_source=args.csv_source,
        csv_target=args.csv_target,
        model_path=args.model_path,
        figs_dir=figs_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
