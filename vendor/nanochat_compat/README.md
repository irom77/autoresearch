# Pinned nanochat compatibility layer

This is the minimal nanochat code needed to load the autoresearch checkpoint.
It is copied from nanochat commit `e85db6b4a4351eb562bec220b3bbcaad28be6722`
(January 16, 2026). The commit is pinned because later nanochat revisions changed
the model architecture.

The source is MIT licensed; see [LICENSE](LICENSE).

The exported checkpoint is generated separately by `export_nanochat.py` under
`nanochat_export/base_checkpoints/autoresearch/`. The original custom tokenizer
is still required for text generation and is intentionally not bundled here until
it is recovered.

Example from the repository root:

```bash
PYTHONPATH=vendor/nanochat_compat \
uv run python -c \
  'from load_model import load_model; load_model("nanochat_export/base_checkpoints/autoresearch")'
```
