import tempfile
import unittest
from pathlib import Path

import torch

from checkpointing import (
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    should_replace_best,
)
from publish_best import should_publish_candidate


class FakeModel:
    def state_dict(self):
        return {"weight": torch.tensor([1.0, 2.0])}


class CheckpointingTests(unittest.TestCase):
    def test_round_trip_preserves_weights_and_metadata(self):
        metadata = {"val_bpb": 1.004616, "commit": "65adfe4"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, FakeModel(), metadata)
            checkpoint = load_checkpoint(path)

        self.assertEqual(checkpoint["format_version"], 1)
        self.assertEqual(checkpoint["metadata"], metadata)
        torch.testing.assert_close(
            checkpoint["model_state_dict"]["weight"],
            torch.tensor([1.0, 2.0]),
        )

    def test_build_checkpoint_contains_metadata_and_weights(self):
        checkpoint = build_checkpoint(FakeModel(), {"val_bpb": 1.0})

        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["metadata"]["val_bpb"], 1.0)

    def test_only_lower_bpb_replaces_best(self):
        self.assertTrue(should_replace_best(1.0, None))
        self.assertTrue(should_replace_best(1.0, 1.1))
        self.assertFalse(should_replace_best(1.1, 1.0))
        self.assertFalse(should_replace_best(1.0, 1.0))

    def test_publisher_only_accepts_strict_improvement(self):
        self.assertTrue(should_publish_candidate({"val_bpb": 0.9}, {"val_bpb": 1.0}))
        self.assertFalse(should_publish_candidate({"val_bpb": 1.0}, {"val_bpb": 1.0}))
        self.assertFalse(should_publish_candidate({"val_bpb": 1.1}, {"val_bpb": 1.0}))


if __name__ == "__main__":
    unittest.main()
