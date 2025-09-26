"""Training utilities shared across source and DANN stages."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .models import DANNModel, SourceClassifier, TrainingStepOutput
except ImportError:
    from models import DANNModel, SourceClassifier, TrainingStepOutput


@dataclass
class EpochStats:
    losses: Dict[str, float]
    accuracy: float


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: Tensor, labels: Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.shape[0])


def train_source_epoch(
    model: SourceClassifier,
    data_loader: Iterable[Dict[str, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in data_loader:
        signals = batch["signal"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return EpochStats({"label": avg_loss}, accuracy)


def evaluate_source(
    model: SourceClassifier,
    data_loader: Iterable[Dict[str, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return EpochStats({"label": avg_loss}, accuracy)


def train_dann_epoch(
    model: DANNModel,
    source_loader: Iterable[Dict[str, Tensor]],
    target_loader: Iterable[Dict[str, Tensor]],
    label_criterion: nn.Module,
    domain_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    start_step: int,
    total_steps: int,
    alpha_max: float = 1.0,
) -> tuple[EpochStats, int]:
    model.train()
    total_label_loss = 0.0
    total_domain_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_domain_samples = 0

    target_iterator = iter(target_loader)
    current_step = start_step
    for _, source_batch in enumerate(source_loader):
        progress = current_step / max(1, total_steps)
        alpha = alpha_max * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
        try:
            target_batch = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_batch = next(target_iterator)

        # Source forward pass
        source_signals = source_batch["signal"].to(device)
        source_labels = source_batch["label"].to(device)
        target_signals = target_batch["signal"].to(device)

        optimizer.zero_grad()
        label_logits, domain_logits_src, _ = model(source_signals, alpha=alpha)
        label_loss = label_criterion(label_logits, source_labels)

        # Domain predictions for source and target
        _, domain_logits_tgt, _ = model(target_signals, alpha=alpha)
        domain_labels_src = torch.zeros(domain_logits_src.size(0), dtype=torch.long, device=device)
        domain_labels_tgt = torch.ones(domain_logits_tgt.size(0), dtype=torch.long, device=device)
        domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
        domain_labels = torch.cat([domain_labels_src, domain_labels_tgt], dim=0)
        domain_loss = domain_criterion(domain_logits, domain_labels)

        total_loss = label_loss + domain_loss * alpha
        total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = source_labels.size(0)
        total_label_loss += label_loss.item() * batch_size
        total_domain_loss += domain_loss.item() * domain_logits.size(0)
        total_domain_samples += domain_logits.size(0)
        total_correct += (label_logits.argmax(dim=1) == source_labels).sum().item()
        total_samples += batch_size

        current_step += 1

    avg_label_loss = total_label_loss / max(1, total_samples)
    avg_domain_loss = total_domain_loss / max(1, total_domain_samples)
    accuracy = total_correct / max(1, total_samples)
    return EpochStats({"label": avg_label_loss, "domain": avg_domain_loss}, accuracy), current_step


def evaluate_dann(
    model: DANNModel,
    data_loader: Iterable[Dict[str, Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> EpochStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            signals = batch["signal"].to(device)
            labels = batch["label"].to(device)
            logits, _, _ = model(signals, alpha=0.0)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return EpochStats({"label": avg_loss}, accuracy)


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config dictionary to JSON-serialisable values for safe checkpointing."""
    def _convert(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return type(value)(_convert(v) for v in value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return {k: _convert(v) for k, v in config.items()}
