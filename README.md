# RunPod GPU Runbook

> Original project documentation: [PROJECT_README.md](PROJECT_README.md)


This repository is running autoresearch experiments on a RunPod H100. The best
benchmark result is `val_bpb=1.004616` from commit `65adfe4`. The pod was
stopped and removed to protect the remaining RunPod balance.

## Roadmap and goals

### Primary goal

Minimize `val_bpb` (validation bits per byte). Lower is better. The current best
result is `1.004616` from commit `65adfe4` with 10% learning-rate warmup.

### Experiment loop

Each experiment follows the same sequence:

1. Choose one focused change to `train.py`.
2. Commit the change on the remote branch `autoresearch/sep24-gpu`.
3. Run `uv run train.py` in a detached screen session for the fixed 300-second budget.
4. Extract `val_bpb`, runtime, token count, MFU, and peak VRAM from the log.
5. Compare against the current best result.
6. Keep an improvement; revert a regression or failed run.
7. Add the detailed result to `results.tsv`, update `leaderboard.tsv`, and update this runbook.

### Planned search areas

The initial search is intentionally incremental:

- Learning-rate warmup: 15%, then 20%, or revert if the trend stops improving.
- Attention window pattern: compare the current `SSSL` pattern with `L`.
- Optimizer rates: controlled changes to `MATRIX_LR`, `EMBEDDING_LR`, and `UNEMBEDDING_LR`.
- Optimizer regularization: weight decay and beta values.
- Model depth/width only after optimizer and schedule experiments plateau.

Only one major variable should change per experiment so results remain attributable.
The evaluation harness in `prepare.py` is the source of truth and must not be changed.

### Success criteria

- Lower `val_bpb` than the current best.
- No crash or out-of-memory failure.
- Peak VRAM remains within the H100's 80 GB capacity.
- The change is simple enough to justify its improvement.

## Autoresearch leaderboard

This table follows nanochat's [Time-to-GPT-2 Leaderboard](https://github.com/irom77/nanochat/blob/master/README_original.md)
format where it is useful: each row identifies the run, metric, change, date,
commit, and contributor. The entries below are **not** official nanochat
time-to-GPT-2 results. Autoresearch currently runs a fixed 300-second,
single-GPU experiment and reports `val_bpb`; it does not yet measure DCLM CORE
or the wall-clock time to exceed the GPT-2 CORE score. `time` therefore means
the fixed training budget, not time-to-GPT-2. [`leaderboard.tsv`](leaderboard.tsv)
is the machine-readable source for this table; [`results.tsv`](results.tsv)
retains the detailed experiment record.

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---:|---:|---:|---:|---|---|---|---|
| 0 | 300 s | 1.010783 | — | H100 baseline | Sep 24 2026 | `228791f` | @irom77 |
| 1 | 300 s | 1.005539 | — | 5% learning-rate warmup | Sep 24 2026 | `a27430b` | @irom77 |
| 2 | 300 s | **1.004616** | — | 10% learning-rate warmup; current best | Sep 24 2026 | `65adfe4` | @irom77 |
| 3 | 300 s | 1.013795 | — | 15% learning-rate warmup; discarded | Sep 24 2026 | `17219f5` | @irom77 |
| 4 | 300 s | 1.015563 | — | 20% learning-rate warmup; discarded | Sep 24 2026 | `446326e` | @irom77 |

Ranking is by lower `val_bpb` among valid runs. When a compatible nanochat
evaluation is available, add its CORE score and retain the same run identity;
do not substitute CORE for `val_bpb` or compare this single-GPU table directly
with nanochat's 8×H100 leaderboard.

### Stopping and finalization

Continue while experiments produce useful improvements and the two-hour daily
window remains available. Stop before `2026-09-24 16:54:51 UTC`, when progress
plateaus, or when the human requests a stop. Before stopping:

1. Confirm the best commit and copy its logs/results locally.
2. Ensure `results.tsv` and this roadmap reflect the final state.
3. Remove pod `2dgw5717mv8ypw` to stop the `$3.49/hr` charge.

## Current pod

- Pod name: `autoresearch-gpu`
- Pod ID: `2dgw5717mv8ypw`
- GPU: NVIDIA H100 80GB HBM3
- GPU memory: 81,559 MiB
- RunPod hourly rate at creation: `$3.49/hr`
- Pod state: `removed` (billing stopped at `2026-09-24 16:33:57 UTC`)
- SSH host: `103.207.149.105`
- SSH port: `19547` (ports may change when the pod resumes)
- SSH key: `/home/irom/.runpod/ssh/runpodctl-ssh-key`
- Project directory: `/workspace/autoresearch`
- Baseline session: `baseline`
- Baseline log: `/workspace/autoresearch/baseline.log`
- Experiment branch: `autoresearch/sep24-gpu`
- Current experiment session: `checkpoint_best2` (completed)
- Current experiment log: `/workspace/autoresearch/experiment4.log`
- Current branch state: best commit `65adfe4`; checkpoint trial completed
- Experiment window started: `2026-09-24 14:54:51 UTC`
- Experiment time limit today: `2 hours` (`2026-09-24 16:54:51 UTC`)
- Approximate experiment time used today: `~1 hour 39 minutes`; pod stopped before the 2-hour limit

## Cost choice

The live `runpodctl gpu list` check showed these relevant rates:

| GPU | VRAM | Community rate | Secure rate | Practical result here |
|---|---:|---:|---:|---|
| A100 SXM | 80 GB | `$1.39/hr` | `$1.59/hr` | Cheapest suitable unmodified baseline; availability/host storage failed during allocation |
| H100 SXM | 80 GB | `$2.69/hr` | `$3.49/hr` | Current pod; reference-class GPU for this repo |
| RTX 6000 Ada | 48 GB | `$0.74/hr` | `$0.84/hr` | Cheaper, but the default baseline OOMed during `torch.compile` |
| RTX A6000 | 48 GB | `$0.33/hr` | `$0.53/hr` | Cheaper, but likely insufficient for the default baseline without reducing batch size/model settings |

Therefore, H100 is not the cheapest option. A100 80 GB is the cheapest option
that should support the default configuration without changing the experiment, but
the available A100 hosts failed before startup with `no space left on device`.
The current H100 costs `$3.49/hr` at its selected secure rate. Rates and availability
are dynamic; re-run `runpodctl gpu list` before switching pods.

RunPod IPs and ports can change if the pod is recreated. Get the current values with:

```bash
runpodctl pod list
runpodctl ssh info 2dgw5717mv8ypw
```

Always run `runpodctl ssh info` after starting or resuming the pod. The SSH port
changed from `12074` to `19547` when this pod was resumed.

## Quick monitoring commands for every experiment

Set `EXPERIMENT` to the screen/log name you want to inspect. This works for
`experiment1`, `experiment2`, and future sessions such as `experiment3`.

```bash
# Choose one: experiment1, experiment2, experiment3, ...
EXPERIMENT=experiment2

# Follow experiment output
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "tail -f /workspace/autoresearch/${EXPERIMENT}.log"

# Watch GPU utilization and VRAM
ssh -t -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'watch -n 1 nvidia-smi'

# Check process/session status
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "screen -ls; ps -ef | grep -E '(${EXPERIMENT}|train.py)' | grep -v grep || true"

# Print final metrics after completion
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "grep -E '^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps):' /workspace/autoresearch/${EXPERIMENT}.log"
```

For the existing runs specifically:

```bash
EXPERIMENT=experiment1  # completed; best result so far: val_bpb=1.005539
EXPERIMENT=experiment2  # latest run
```

## Connect with SSH

From the repository directory on the local machine:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key \
  -p 19547 \
  -o StrictHostKeyChecking=no \
  root@103.207.149.105
```

Or use the command printed by `runpodctl ssh info`.

## Monitor the baseline

Check the latest training output without opening the interactive session:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'tail -f /workspace/autoresearch/baseline.log'
```

Check whether the detached session and Python process are still alive:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "screen -ls; ps -ef | grep -E 'train.py' | grep -v grep || true"
```

Monitor GPU utilization and memory:

```bash
ssh -t -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'watch -n 1 nvidia-smi'
```

A compact one-shot GPU check:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

The baseline is configured by the project to run for a fixed 300-second training
budget, excluding startup and compilation. When it finishes, extract the metrics:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "grep -E '^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):' /workspace/autoresearch/baseline.log"
```

## Monitor a named experiment

The same commands apply to the active experiment log:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'tail -f /workspace/autoresearch/<experiment-name>.log'
```

Extract completed experiment metrics:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "grep -E '^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):' /workspace/autoresearch/<experiment-name>.log"
```

View the experiment history:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'cd /workspace/autoresearch && git log --oneline --decorate -5 && cat results.tsv'
```

## Experiment history

The detailed machine-readable history is in [`results.tsv`](results.tsv); the
summary leaderboard above is the human-facing view. Peak VRAM was `44.0 GB`
for all five recorded runs.

## Checkpoints and Hugging Face publishing

Current training code saves a candidate checkpoint after each completed run:

```text
/workspace/autoresearch/checkpoints/candidate.pt
```

The checkpoint contains model weights, model configuration, `val_bpb`, commit,
runtime, token, and memory metadata. Promote and publish a candidate only when it
strictly improves the local best:

```bash
export HF_REPO_ID=niuk77/autoresearch
export HF_TOKEN=<your Hugging Face write token>
export AUTORESEARCH_COMMIT=$(git rev-parse --short HEAD)
export AUTORESEARCH_PUBLISH=1
uv run train.py
```

Or publish an existing candidate explicitly:

```bash
HF_REPO_ID=niuk77/autoresearch \
HF_TOKEN=<your Hugging Face write token> \
uv run publish_best.py --checkpoint checkpoints/candidate.pt
```

The publisher creates the model repository if needed and uploads `best.pt` and
`metadata.json`. Worse or equal candidates are skipped, so the Hugging Face repo
always remains at the best known `val_bpb`. If no `HF_TOKEN` is configured, the
best checkpoint is still saved locally under `checkpoints/best.pt`; upload it later
with the command above. Never commit the token or place it in this runbook.

### Final checkpoint status for 2026-09-24

- The checkpoint artifact is `checkpoints/best.pt` (about 159 MB).
- The checkpoint-producing rerun measured `val_bpb=1.012760`; it is preserved as
  the best saved checkpoint from this wrap-up, but it does not replace the better
  benchmark score `1.004616` recorded for commit `65adfe4`.
- Hugging Face publication did not occur because `HF_TOKEN` was absent. Configure
  a Hugging Face write token in a future pod session, then run the explicit publish
  command above; it will create `niuk77/autoresearch` and replace the remote
  `best.pt` only for a strict improvement.
- The local copy is in `checkpoints/`; this directory is ignored by git.
- After the session, the local checkpoint and metadata were uploaded to
  `https://huggingface.co/niuk77/autoresearch`, and the model card was added.

### Deferred tokenizer recovery

The model uses a custom 8,192-token tokenizer. The original tokenizer cache was
not copied before the pod was removed, and the pod could not be restarted because
RunPod had no available H100 on that host. Do not use nanochat's default tokenizer
with this model.

When budget is available, create a new pod or use a local machine and run:

```bash
cd /workspace/autoresearch
uv run prepare.py
```

Copy `~/.cache/autoresearch/tokenizer/tokenizer.pkl` locally, verify its vocabulary
size is 8,192, and upload it to the Hub as `tokenizer/tokenizer.pkl` in
`niuk77/autoresearch`. The repository now includes the pinned compatibility layer
under `vendor/nanochat_compat/`, copied from nanochat revision `e85db6b`; use its
`load_model.py` loader for the exported native checkpoint. The current H100 pod no
longer exists; provision a new pod and run the normal setup/monitoring steps if
GPU inference or additional experiments are needed.

### Pause and resume later

The pod can be stopped without deleting its persistent workspace:

```bash
runpodctl pod stop 2dgw5717mv8ypw
```

To resume the same pod later:

```bash
runpodctl pod start 2dgw5717mv8ypw
runpodctl pod list
runpodctl ssh info 2dgw5717mv8ypw
```

After a stop/resume, reinstall container-local utilities before launching a new
session, because packages such as `screen` may not persist across container restarts:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p <PORT> root@<IP> \
  'apt-get update && apt-get install -y screen python3-dev'
```

The tokenizer/data cache under `/root/.cache/autoresearch` may also be absent after
a stop/resume. Rebuild it before training:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'cd /workspace/autoresearch && uv run prepare.py'
```

After reconnecting, verify the best state and results:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p <PORT> root@<IP> \
  "cd /workspace/autoresearch && git log --oneline --decorate -5 && cat results.tsv"
```

If the stopped pod is unavailable, create a new GPU pod using the setup instructions
above, sync the local checkout, run `uv sync`, and run `uv run prepare.py`. The local
git branch and `results.tsv` are the recovery source of truth; do not start a new
experiment until the checkout is at the best commit `65adfe4`.

## Attach to or restart the session

Attach interactively:

```bash
ssh -t -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  'screen -r baseline'
```

Detach from screen without stopping training: press `Ctrl-A`, then `D`.

If the process has exited and you intentionally want to rerun it:

```bash
ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 root@103.207.149.105 \
  "screen -dmS baseline bash -lc 'cd /workspace/autoresearch && uv run train.py > baseline.log 2>&1'"
```

## Remote project setup

The remote environment was prepared with:

```bash
apt-get update && apt-get install -y rsync screen python3-dev
cd /workspace/autoresearch
uv sync
uv run prepare.py
```

`prepare.py` completed successfully and created the data shards and tokenizer under
`/root/.cache/autoresearch`. The locked environment contains `torch==2.9.1+cu128`.

To sync the current local checkout to a new pod, run locally:

```bash
rsync -rltvz \
  -e "ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547 -o StrictHostKeyChecking=no" \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="__pycache__" \
  --exclude=".cache" \
  --exclude=".vscode" \
  ./ root@103.207.149.105:/workspace/autoresearch/
```

## RunPod lifecycle and cleanup

Check pod status and current spend:

```bash
runpodctl pod list
runpodctl user
```

Stop billing when the run is complete and any results are copied back:

```bash
runpodctl pod remove 2dgw5717mv8ypw
```

Removing the pod is destructive to its remote container and volume. Copy back any
logs, checkpoints, or modified files first. For example:

```bash
rsync -avz \
  -e "ssh -i /home/irom/.runpod/ssh/runpodctl-ssh-key -p 19547" \
  root@103.207.149.105:/workspace/autoresearch/baseline.log ./
```

## Provisioning note

The first attempted RTX 6000 Ada pod started correctly, but the default baseline
hit CUDA OOM during `torch.compile` with only 48 GB VRAM. Two A100 allocations then
failed before container startup because the selected RunPod hosts reported
`no space left on device`. Those failed pods were removed. The current H100 pod is
the successful replacement and matches the project’s documented H100-class setup.
