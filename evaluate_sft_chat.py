"""Deterministic plain-text chat evaluation for an autoresearch SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def format_prompt(user_message: str) -> str:
    if not user_message.strip():
        raise ValueError("user_message must not be empty")
    return f"User: {user_message}\nAssistant: "


@torch.no_grad()
def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must not be negative")
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    generated = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model(input_ids)[:, -1, :]
        if temperature == 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1).item())
        token_ids.append(next_id)
        generated.append(next_id)
        if next_id == tokenizer.get_bos_token_id():
            break
    return tokenizer.decode(generated).strip()


def evaluate_prompts(model, tokenizer, prompts: list[str], device: str, output_path: Path, metadata: dict) -> dict:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = []
    for prompt in prompts:
        start = time.perf_counter()
        reply = generate_reply(model, tokenizer, prompt, metadata["max_new_tokens"], metadata["temperature"])
        elapsed = time.perf_counter() - start
        results.append({"prompt": prompt, "reply": reply, "seconds": elapsed})
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0
    report = {"metadata": metadata, "peak_vram_mb": peak_vram_mb, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/autoresearch-sft.pt"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path.home() / ".cache/autoresearch/tokenizer")
    parser.add_argument("--output", type=Path, default=Path("runs/sft_chat_eval.json"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from prepare import Tokenizer
    from train_sft import load_sft_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint_metadata = load_sft_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer.from_directory(args.tokenizer_dir)
    prompts = [
        format_prompt("Hello! What can you help me with?"),
        format_prompt("Explain why the sky is blue in one sentence."),
        format_prompt("Write a short cheerful greeting."),
    ]
    metadata = {
        "checkpoint": str(args.checkpoint),
        "base_commit": checkpoint_metadata.get("base_commit"),
        "sft_format": checkpoint_metadata.get("format"),
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }
    print(json.dumps(evaluate_prompts(model, tokenizer, prompts, device, args.output, metadata), ensure_ascii=False))


if __name__ == "__main__":
    main()
