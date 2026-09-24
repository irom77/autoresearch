"""Keep the best local checkpoint and publish it to Hugging Face."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from checkpointing import load_checkpoint, should_replace_best


DEFAULT_REPO_ID = "niuk77/autoresearch"


def should_publish_candidate(
    candidate_metadata: dict[str, Any],
    best_metadata: dict[str, Any] | None,
) -> bool:
    """Return whether candidate metadata strictly improves the best result."""
    candidate_val_bpb = float(candidate_metadata["val_bpb"])
    best_val_bpb = None if best_metadata is None else float(best_metadata["val_bpb"])
    return should_replace_best(candidate_val_bpb, best_val_bpb)


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def publish_checkpoint(
    checkpoint_path: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
) -> bool:
    """Promote a better checkpoint locally and upload it to Hugging Face."""
    candidate_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(candidate_path)
    candidate_metadata = checkpoint["metadata"]
    output_dir = Path(checkpoint_dir) if checkpoint_dir else candidate_path.parent
    best_path = output_dir / "best.pt"
    best_metadata_path = output_dir / "best_metadata.json"
    best_metadata = _read_metadata(best_metadata_path)

    if not should_publish_candidate(candidate_metadata, best_metadata):
        print(
            f"Skipping {candidate_metadata['val_bpb']:.6f}; "
            f"best is {best_metadata['val_bpb']:.6f}."
        )
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, best_path)
    best_metadata_path.write_text(json.dumps(candidate_metadata, indent=2, sort_keys=True) + "\n")

    if not token:
        raise RuntimeError(
            "Best checkpoint saved locally, but HF_TOKEN is not configured; "
            "set HF_TOKEN before publishing."
        )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    commit = candidate_metadata.get("commit", "unknown")
    api.upload_file(
        path_or_fileobj=str(best_path),
        path_in_repo="best.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update best model: val_bpb={candidate_metadata['val_bpb']:.6f} ({commit})",
    )
    api.upload_file(
        path_or_fileobj=str(best_metadata_path),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update best model metadata",
    )
    print(f"Published best checkpoint to https://huggingface.co/{repo_id}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/candidate.pt")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", DEFAULT_REPO_ID))
    args = parser.parse_args()
    publish_checkpoint(
        args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
