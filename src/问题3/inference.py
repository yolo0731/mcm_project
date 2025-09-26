"""Inference script to classify target-domain signals using the trained model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import math
import numpy as np
from scipy.special import softmax
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from . import ROOT
    from .data import BearingDataset, INDEX_TO_LABEL, load_metadata
    from .models import DANNModel
    from .paths import ensure_output_dirs
    from .train_utils import set_seed
    from .visualization import plot_prediction_distribution, save_prediction_table
except ImportError:
    from paths import ROOT
    from data import BearingDataset, INDEX_TO_LABEL, load_metadata
    from models import DANNModel
    from paths import ensure_output_dirs
    from train_utils import set_seed
    from visualization import plot_prediction_distribution, save_prediction_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on target-domain signals")
    parser.add_argument("--csv-target", type=Path, default=ROOT / "data" / "processed" / "16feature.csv")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to trained DANN model checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--figs-dir", type=Path, default=None)
    parser.add_argument("--top-ratio", type=float, default=1.0, help="Top window fraction used for aggregation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for averaged logits")
    return parser.parse_args()


def load_model(model_path: Path, device: torch.device) -> DANNModel:
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model = DANNModel()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def aggregate_predictions(
    logits_per_file: Dict[int, List[np.ndarray]],
    file_mapping: Dict[int, str],
    metadata: pd.DataFrame,
    top_ratio: float = 0.75,  # Use more windows for better statistics
    temperature: float = 1.2,  # Better calibration while preserving class discrimination
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prob_rows = []
    for file_id, arr_list in logits_per_file.items():
        stacked_logits = np.stack(arr_list, axis=0)
        window_probs = softmax(stacked_logits, axis=1)
        window_conf = window_probs.max(axis=1)
        if top_ratio < 1.0 and stacked_logits.shape[0] > 1:
            k = max(1, int(round(stacked_logits.shape[0] * top_ratio)))
            top_idx = np.argsort(window_conf)[-k:]
            selected_logits = stacked_logits[top_idx]
        else:
            selected_logits = stacked_logits
        # Enhanced confidence calibration
        mean_logits = selected_logits.mean(axis=0) / max(temperature, 0.1)
        mean_probs = softmax(mean_logits)
        pred_idx = int(mean_probs.argmax())
        confidence_raw = float(mean_probs[pred_idx])

        # Better confidence calibration - avoid overconfidence but preserve discrimination
        confidence = min(confidence_raw, 0.97)  # Cap at 97% instead of 99.9%

        # Add reliability assessment
        window_probs_selected = softmax(selected_logits, axis=1)
        pred_consistency = np.mean([probs[pred_idx] for probs in window_probs_selected])
        uncertainty = -np.sum(mean_probs * np.log(mean_probs + 1e-8))  # Entropy
        reliability = 0.6 * confidence + 0.3 * pred_consistency + 0.1 * (1 - uncertainty/np.log(4))
        reliability = max(0, min(1, reliability))
        file_key = file_mapping[file_id]
        filename_candidates = metadata.loc[metadata["file"] == file_key, "filename"]
        if not filename_candidates.empty:
            display_name = filename_candidates.iloc[0]
        else:
            display_name = Path(file_key).name
        prob_rows.append(
            {
                "file": file_key,
                "filename": display_name,
                **{
                    f"prob_{INDEX_TO_LABEL[idx]}": float(prob)
                    for idx, prob in enumerate(mean_probs)
                },
            }
        )
        # Determine if prediction is reliable
        reliable_prediction = confidence >= 0.7 and reliability >= 0.6

        rows.append(
            {
                "file": file_key,
                "filename": display_name,
                "predicted_index": pred_idx,
                "predicted_label": INDEX_TO_LABEL[pred_idx],
                "confidence": confidence,
                "uncertainty": uncertainty,
                "reliability": reliability,
                "reliable_prediction": reliable_prediction,
                "num_windows": len(selected_logits),
            }
        )
    result_df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)
    prob_df = pd.DataFrame(prob_rows).sort_values("filename").reset_index(drop=True)
    return result_df, prob_df


def run_inference(args: argparse.Namespace) -> Path:
    top_ratio = max(0.1, min(1.0, args.top_ratio))
    temperature = max(0.05, args.temperature)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir, figs_dir = ensure_output_dirs(args.out_dir, args.figs_dir)

    target_meta = load_metadata(args.csv_target)
    dataset = BearingDataset(target_meta, domain_label=1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_path = args.model_path or (processed_dir / "dann_model.pth")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained DANN model not found at {model_path}")
    model = load_model(model_path, device)

    logits_by_file: Dict[int, List[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            signals = batch["signal"].to(device)
            logits, _, _ = model(signals, alpha=0.0)
            batch_logits = logits.cpu().numpy()
            for idx, file_id in enumerate(batch["file_id"].cpu().numpy().tolist()):
                logits_by_file[file_id].append(batch_logits[idx])

    file_mapping = dataset.file_id_to_key()
    predictions_df, probability_df = aggregate_predictions(
        logits_by_file,
        file_mapping,
        target_meta,
        top_ratio=top_ratio,
        temperature=temperature,
    )
    predictions_path = processed_dir / "problem3_target_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    prob_path = processed_dir / "problem3_target_probabilities.csv"
    probability_df.to_csv(prob_path, index=False)

    counts = {label: 0 for label in INDEX_TO_LABEL.values()}
    for label in predictions_df["predicted_label"]:
        counts[label] = counts.get(label, 0) + 1
    # plot_prediction_distribution(counts, figs_dir / "target_pred_distribution.png")  # Removed as requested
    save_prediction_table(
        predictions_df,
        figs_dir / "problem3_result_table_en.png",
        figs_dir / "problem3_result_table_en.pdf",
    )

    print(f"Saved predictions to {predictions_path}")
    return predictions_path


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
