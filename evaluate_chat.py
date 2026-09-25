"""Run deterministic completion and chat probes against the exported model."""

import json
import time
from pathlib import Path

import torch

from load_model import load_model
from prepare import Tokenizer


MODEL_DIR = Path("nanochat_export/base_checkpoints/autoresearch")
TOKENIZER_DIR = Path("/root/.cache/autoresearch/tokenizer")


def main() -> None:
    model, metadata = load_model(MODEL_DIR, device="cuda")
    tokenizer = Tokenizer.from_directory(TOKENIZER_DIR)
    vocab_size = metadata["model_config"]["vocab_size"]
    assert tokenizer.get_vocab_size() == vocab_size == 8192

    print(
        json.dumps(
            {
                "model_commit": metadata["commit"],
                "val_bpb": metadata["val_bpb"],
                "tokenizer_vocab_size": tokenizer.get_vocab_size(),
                "device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    prompts = [
        (
            "completion",
            "The sky is blue because",
            tokenizer.encode("The sky is blue because", prepend=tokenizer.get_bos_token_id()),
        ),
        (
            "prompted_completion",
            "User: Hello! What can you do?\nAssistant:",
            tokenizer.encode(
                "User: Hello! What can you do?\nAssistant:",
                prepend=tokenizer.get_bos_token_id(),
            ),
        ),
    ]
    for kind, prompt, tokens in prompts:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        generated = list(model.generate(tokens, max_tokens=48, temperature=0.0))
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "kind": kind,
                    "prompt": prompt,
                    "prompt_tokens": len(tokens),
                    "generated_tokens": len(generated),
                    "seconds": elapsed,
                    "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "response": tokenizer.decode(generated),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
