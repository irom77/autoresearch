# TODO: evaluate the trained model in nanochat

The current pod was removed. Resume this checklist only after GPU budget is
available; use [README.md](README.md) for provisioning, SSH monitoring, and
recovery commands.

## Required before evaluation

- [ ] Provision a new pod and run `uv run prepare.py`.
- [ ] Recover the exact custom tokenizer at `~/.cache/autoresearch/tokenizer/tokenizer.pkl`.
- [ ] Upload it to `niuk77/autoresearch` as `tokenizer/tokenizer.pkl`.
- [ ] Use the vendored nanochat compatibility layer under `vendor/nanochat_compat/`, pinned to revision `e85db6b`.
- [ ] Verify `load_model.py` loads `nanochat/base_checkpoints/autoresearch/model_000000.pt` with strict parameter matching.

## Chat evaluation

- [ ] Run a short interactive or scripted chat using the trained base model in the pod.
- [ ] Save prompts, responses, command, commit, model revision, and tokenizer revision under `runs/`.
- [ ] Monitor the session with SSH, `screen`, logs, and `nvidia-smi` as documented in `README.md`.

## Nanochat-style evaluation

- [ ] Run nanochat’s base-model evaluation scripts against the exported checkpoint.
- [ ] Record validation loss/BPB, task scores, generation samples, throughput, and GPU memory.
- [ ] Compare the results with the corresponding nanochat baseline using the same tokenizer and evaluation settings.
- [ ] Update the Hub model card and `README.md` with reproducible commands and results.

## Done when

The tokenizer is archived with the model, nanochat loads the checkpoint, a chat
run and benchmark evaluation are reproducible, and all results are recorded.
