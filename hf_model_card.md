---
library_name: pytorch
tags:
- autoresearch
- language-modeling
- gpt
---

# Autoresearch checkpoint

This repository contains the best saved checkpoint produced during the
2026-09-24 RunPod experiment window for the autoresearch project.

## Model

- Architecture: compact GPT-style language model
- Parameters: 50.3M
- Layers: 8
- Hidden size: 512
- Attention: 4 heads, 4 key/value heads
- Context length: 2,048 tokens
- Vocabulary: 8,192 tokens
- Attention window pattern: `SSSL`
- Training hardware: NVIDIA H100 80GB
- Training budget: 300 seconds

## Evaluation

The uploaded checkpoint's recorded validation score is:

```text
val_bpb: 1.012760
```

Lower `val_bpb` is better. The experiment log also contains the runtime,
throughput, MFU, and memory measurements.

The broader experiment search recorded a lower benchmark score of `1.004616`
for commit `65adfe4`; that run was completed before checkpoint saving was added,
so its weights are not the artifact uploaded here.

## Files

- `best.pt`: PyTorch checkpoint containing model weights and metadata
- `metadata.json`: training configuration and measured metrics
- `nanochat/base_checkpoints/autoresearch/model_000000.pt`: nanochat-native
  model weights
- `nanochat/base_checkpoints/autoresearch/meta_000000.json`: nanochat-native
  metadata

The checkpoint is intended for research and reproducibility. Load it with
`torch.load(..., weights_only=False)` and use the matching project `train.py`
model definition to reconstruct the model.

The native nanochat export matches the earlier nanochat architecture revision
`e85db6b`. The custom 8,192-token tokenizer is required separately; it could not
be recovered because the stopped RunPod pod could not be restarted due to host
GPU capacity.

## License

See the source project for licensing and dataset terms.
