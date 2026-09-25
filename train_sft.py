"""Bounded supervised fine-tuning for the autoresearch base checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from sft import build_batch


def save_sft_checkpoint(path: str | Path, model: torch.nn.Module, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def load_sft_checkpoint(
    path: str | Path,
    device: str | torch.device,
    model_factory: Callable[[], torch.nn.Module] | None = None,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    if model_factory is None:
        sys.path.insert(0, str(Path(__file__).parent / "vendor" / "nanochat_compat"))
        from nanochat.gpt import GPT, GPTConfig

        model_factory = lambda: GPT(GPTConfig(**metadata["model_config"]))
    model = model_factory()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), metadata


def _load_conversations(path: str | Path) -> list[list[dict[str, str]]]:
    conversations = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            messages = item.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"line {line_number}: expected a messages list")
            conversations.append(messages)
    if not conversations:
        raise ValueError("SFT dataset is empty")
    return conversations


def train_sft(
    model: torch.nn.Module,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    device: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_seq_len: int | None = None,
) -> tuple[int, int, float]:
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    random.seed(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    train_tokens = 0
    final_loss = float("nan")
    for step in range(steps):
        batch = [conversations[(step * batch_size + i) % len(conversations)] for i in range(batch_size)]
        input_ids, labels = build_batch(batch, tokenizer, device, max_length=max_seq_len)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_tokens += int((labels != -100).sum().item())
        final_loss = float(loss.detach().cpu())
    return steps, train_tokens, final_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=Path("nanochat_export/base_checkpoints/autoresearch"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path.home() / ".cache/autoresearch/tokenizer")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/autoresearch-sft.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "vendor" / "nanochat_compat"))
    from load_model import load_model
    from prepare import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, base_metadata = load_model(args.base_checkpoint, device=device)
    tokenizer = Tokenizer.from_directory(args.tokenizer_dir)
    conversations = _load_conversations(args.data)
    steps, train_tokens, final_loss = train_sft(
        model,
        tokenizer,
        conversations,
        device,
        args.steps,
        args.batch_size,
        args.learning_rate,
        args.seed,
        max_seq_len=base_metadata["model_config"]["sequence_len"],
    )
    metadata = {
        "format": "autoresearch-sft-v1",
        "base_checkpoint": str(args.base_checkpoint),
        "base_commit": base_metadata.get("commit"),
        "model_config": base_metadata["model_config"],
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "steps": steps,
        "train_tokens": train_tokens,
        "final_loss": final_loss,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
    }
    save_sft_checkpoint(args.output, model, metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
