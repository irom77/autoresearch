import json

import torch

from evaluate_sft_chat import format_prompt, generate_reply


class FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None):
        ids = [ord(char) % 10 for char in text]
        return ([prepend] if prepend is not None else []) + ids

    def decode(self, ids):
        return "reply"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        logits = torch.zeros((1, input_ids.shape[1], 10))
        logits[:, :, 3] = 1
        return logits


def test_format_prompt_uses_plain_text_chat_delimiters():
    assert format_prompt("Hello") == "User: Hello\nAssistant: "


def test_generate_reply_is_deterministic_at_zero_temperature():
    reply = generate_reply(FakeModel(), FakeTokenizer(), "User: Hi\nAssistant: ", 3, 0.0)
    assert reply == "reply"
