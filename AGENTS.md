# Experiment instructions

## Before running

- Read [README.md](README.md) for the current RunPod status, roadmap, recovery, and monitoring commands.
- For deferred chat or nanochat evaluation work, follow [TODO.md](TODO.md).
- Keep the daily experiment window within the user’s stated budget; check the deadline before starting a run.
- Use one focused change per experiment. Keep `prepare.py` and the evaluation harness unchanged.
- Record the commit, `val_bpb`, runtime, tokens, MFU, and peak VRAM in `results.tsv`; update the nanochat-style summary in `leaderboard.tsv` and `README.md`.

## RunPod workflow

- Use `runpodctl ssh info <pod-id>` after every pod start or resume; IPs and ports can change.
- Run training in detached `screen` sessions and monitor the log plus `nvidia-smi` over SSH.
- Before stopping or removing a pod, copy logs and checkpoints locally. Remove the pod when work is complete to stop billing.
- After recovery on a new pod, run dependency setup and `uv run prepare.py` before training.

## Checkpoints and publishing

- Every completed run should save `checkpoints/candidate.pt` with metadata.
- Promote or publish only a strict improvement in lower `val_bpb`; preserve the current best otherwise.
- The Hugging Face repository is `niuk77/autoresearch`. Read the token from the local Hugging Face credential store; never print, commit, or place it in documentation.
- Keep the checkpoint, metadata, tokenizer, and model-card status documented before declaring the run complete.

## Completion

A run is complete only when the result is recorded, the best checkpoint is safely copied or published, the runbook reflects the state, tests pass, and the pod is stopped or removed.
