"""Checkpoint creation and best-result selection for autoresearch runs."""

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 1


def build_checkpoint(model: torch.nn.Module, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a portable inference checkpoint with model weights and run metadata."""
    model = getattr(model, "_orig_mod", model)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
) -> Path:
    """Save a checkpoint, creating its parent directory if needed."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_checkpoint(model, metadata), checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint onto CPU so it can be inspected without a GPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def should_replace_best(candidate_val_bpb: float, best_val_bpb: float | None) -> bool:
    """Return whether a candidate strictly improves the lower-is-better metric."""
    return best_val_bpb is None or candidate_val_bpb < best_val_bpb
