"""Export an autoresearch checkpoint in nanochat's native checkpoint layout."""

import argparse
import json
from pathlib import Path

import torch


def export_checkpoint(source: str | Path, destination: str | Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    metadata = dict(checkpoint["metadata"])
    metadata.setdefault("model_config", {})
    metadata["source_checkpoint"] = source.name
    metadata["format"] = "nanochat-native"

    model_dir = destination / "base_checkpoints" / "autoresearch"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["model_state_dict"], model_dir / "model_000000.pt")
    (model_dir / "meta_000000.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="nanochat_export")
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
