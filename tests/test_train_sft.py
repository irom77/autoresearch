import torch

from train_sft import load_sft_checkpoint, save_sft_checkpoint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def test_sft_checkpoint_round_trips_metadata_and_weights(tmp_path):
    path = tmp_path / "sft.pt"
    model = TinyModel()
    metadata = {
        "format": "autoresearch-sft-v1",
        "base_checkpoint": "base.pt",
        "tokenizer_vocab_size": 8192,
        "steps": 3,
        "train_tokens": 42,
        "model_config": {"factory": "test"},
    }

    save_sft_checkpoint(path, model, metadata)
    loaded_model, loaded_metadata = load_sft_checkpoint(
        path,
        "cpu",
        model_factory=TinyModel,
    )

    assert loaded_metadata == metadata
    assert torch.equal(loaded_model.weight, model.weight)
