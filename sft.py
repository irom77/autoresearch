"""Tokenizer-preserving conversation rendering for autoresearch SFT."""

from __future__ import annotations

from typing import Protocol

import torch


class TokenizerLike(Protocol):
    def get_bos_token_id(self) -> int: ...

    def encode(self, text: str, prepend: int | None = None) -> list[int]: ...


def render_conversation(messages: list[dict[str, str]], tokenizer: TokenizerLike) -> tuple[list[int], list[int]]:
    """Render alternating user/assistant messages and mask prompt tokens.

    Labels use ``-100`` for tokens excluded from cross-entropy. Assistant
    content, but not its role prefix, is trainable.
    """
    if not messages or len(messages) % 2:
        raise ValueError("conversation must contain alternating user and assistant messages")

    input_ids = [tokenizer.get_bos_token_id()]
    labels = [-100]
    expected_role = "user"
    for message in messages:
        if message.get("role") != expected_role:
            raise ValueError("conversation must contain alternating user and assistant messages")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be a non-empty string")

        prefix = f"{expected_role.title()}: "
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))
        input_ids.extend(content_ids)
        labels.extend(content_ids if expected_role == "assistant" else [-100] * len(content_ids))
        input_ids.append(tokenizer.encode("\n")[0])
        labels.append(-100)
        expected_role = "assistant" if expected_role == "user" else "user"

    return input_ids, labels


def build_batch(
    conversations: list[list[dict[str, str]]],
    tokenizer: TokenizerLike,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render conversations into right-padded input and masked-label tensors."""
    rendered = [render_conversation(conversation, tokenizer) for conversation in conversations]
    max_length = max(len(input_ids) for input_ids, _ in rendered)
    pad_id = tokenizer.get_bos_token_id()
    input_rows, label_rows = [], []
    for input_ids, labels in rendered:
        padding = max_length - len(input_ids)
        input_rows.append(input_ids + [pad_id] * padding)
        label_rows.append(labels + [-100] * padding)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )
