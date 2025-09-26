"""Path utilities for Problem 3 domain adaptation modules."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROCESSED_DIR: Path = ROOT / "data" / "processed" / "问题3"
DEFAULT_FIGS_DIR: Path = ROOT / "figs" / "问题3"


def ensure_output_dirs(processed_dir: Path | None = None, figs_dir: Path | None = None) -> Tuple[Path, Path]:
    """Ensure processed and figure directories exist and return their paths."""
    processed = (processed_dir or DEFAULT_PROCESSED_DIR).resolve()
    figures = (figs_dir or DEFAULT_FIGS_DIR).resolve()
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return processed, figures
