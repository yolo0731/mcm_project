from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Optional pretty display in notebooks
    from IPython.display import display as ipy_display  # type: ignore
except ImportError:  # pragma: no cover - display not available in CLI
    ipy_display = None


def ensure_directory(path: Path) -> Path:
    """Create parent directories for the given path if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Persist a matplotlib figure to disk and close it to release memory."""
    ensure_directory(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def safe_display(obj: object, *, title: Optional[str] = None) -> None:
    """Best-effort display helper that falls back to plain printing."""
    if title:
        print(title)
    if ipy_display is not None:
        ipy_display(obj)
    else:
        print(obj)


def print_class_distribution(values: Iterable[int], class_names: Iterable[str]) -> None:
    for name, count in zip(class_names, values):
        print(f"  {name}: {count} samples")

