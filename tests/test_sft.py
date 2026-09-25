import pytest

from sft import build_batch, render_conversation


class FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None):
        ids = [ord(char) for char in text]
        return ([prepend] if prepend is not None else []) + ids


def test_render_conversation_masks_user_tokens_and_trains_assistant_tokens():
    input_ids, labels = render_conversation(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        FakeTokenizer(),
    )

    assert len(input_ids) == len(labels)
    assert any(label == -100 for label in labels)
    assert any(label != -100 for label in labels)
    first_assistant = labels.index(ord("H"))
    assert all(label == -100 for label in labels[:first_assistant])


def test_render_conversation_rejects_invalid_role_order():
    with pytest.raises(ValueError, match="alternating"):
        render_conversation(
            [{"role": "assistant", "content": "Hello"}],
            FakeTokenizer(),
        )


def test_build_batch_pads_inputs_and_labels():
    inputs, labels = build_batch(
        [
            [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}],
            [{"role": "user", "content": "C"}, {"role": "assistant", "content": "DD"}],
        ],
        FakeTokenizer(),
        "cpu",
    )

    assert inputs.shape == labels.shape
    assert inputs.shape[0] == 2
    assert (labels == -100).any()


def test_build_batch_truncates_to_context_length():
    inputs, labels = build_batch(
        [[{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]],
        FakeTokenizer(),
        "cpu",
        max_length=3,
    )

    assert inputs.shape == (1, 3)
    assert labels.shape == (1, 3)
