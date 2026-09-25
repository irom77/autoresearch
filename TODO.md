# TODO: evaluate the trained model in nanochat

The current pod was removed. Resume this checklist only after GPU budget is
available; use [README.md](README.md) for provisioning, SSH monitoring, and
recovery commands.

## Required before evaluation

- [ ] Provision a new pod and run `uv run prepare.py`.
- [x] Recover the exact custom tokenizer at `~/.cache/autoresearch/tokenizer/tokenizer.pkl`.
- [x] Upload it to `niuk77/autoresearch` as `tokenizer/tokenizer.pkl`.
- [x] Upload `tokenizer/token_bytes.pt` alongside it for BPB evaluation.
- [x] Use the vendored nanochat compatibility layer under `vendor/nanochat_compat/`, pinned to revision `e85db6b`.
- [x] Verify `load_model.py` loads `nanochat/base_checkpoints/autoresearch/model_000000.pt` with strict parameter matching.

## Chat evaluation

- [x] Run a short scripted completion/chat-style probe using the trained base model in the pod.
- [x] Save prompts, responses, device, model revision, tokenizer vocabulary, timing, and VRAM under `runs/chat_eval_2026-09-25.log`.
- [ ] Add chat fine-tuning/SFT before judging conversational quality; the current artifact is a base model.
- [ ] Monitor the session with SSH, `screen`, logs, and `nvidia-smi` as documented in `README.md`.

## Nanochat-style evaluation

- [ ] Run nanochat’s full base-model evaluation scripts against the exported checkpoint. The current repository has only the pinned compatibility loader, so this requires vendoring or provisioning the matching nanochat evaluation scripts.
- [x] Record the checkpoint's validation BPB, generation samples, throughput, and GPU memory in the recovered-model probe log.
- [ ] Compare task scores with a corresponding nanochat baseline using identical tokenizer and evaluation settings.
- [ ] Update the Hub model card and `README.md` with reproducible commands and results.

## Done when

The tokenizer is archived with the model, the pinned loader strictly loads the
checkpoint, the base-model probe is reproducible, and the remaining full
nanochat evaluation is either completed or explicitly documented as unavailable.
