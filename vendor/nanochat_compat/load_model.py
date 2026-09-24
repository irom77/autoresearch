"""Load the exported autoresearch checkpoint with the pinned nanochat model."""

import json
from pathlib import Path

import torch

from nanochat.gpt import GPT, GPTConfig


def load_model(checkpoint_dir: str | Path, device: str = "cpu"):
    """Return an eval-mode nanochat GPT and its metadata."""
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "meta_000000.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    config = GPTConfig(**metadata["model_config"])
    model = GPT(config)
    model.init_weights()
    state_dict = torch.load(
        checkpoint_dir / "model_000000.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), metadata
