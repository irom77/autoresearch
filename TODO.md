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
- [x] Add a separate tokenizer-preserving SFT derivative before judging conversational quality.
- [x] Record deterministic SFT and base comparison probes with SSH/screen logs.

## Nanochat-style evaluation

- [x] Run the pinned nanochat CORE evaluator against the exported checkpoint; the bounded 100-example-per-task diagnostic produced CORE `0.058873` across all 22 tasks.
- [x] Record the checkpoint's validation BPB, generation samples, throughput, and GPU memory in the recovered-model probe log.
- [ ] Compare task scores with a corresponding nanochat baseline using identical tokenizer and evaluation settings.
- [x] Update `README.md` with the reproducible command, scope, and result.
- [x] Update the Hub model card with the CORE diagnostic result and SFT derivative.

## Done when

The tokenizer is archived with the model, the pinned loader strictly loads the
checkpoint, the base-model probe is reproducible, and the remaining full
nanochat evaluation is either completed or explicitly documented as unavailable.
