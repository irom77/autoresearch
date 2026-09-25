# Chat SFT for the Autoresearch Base Model

## Goal

Create a separately named conversational derivative of the autoresearch base
checkpoint without changing the exact base tokenizer or overwriting the base
Hub artifact.

## Constraints

- Preserve the existing 8,192-token tokenizer and 50,332,176-parameter base
  checkpoint.
- Use plain text conversation delimiters (`User:` and `Assistant:`), because
  the tokenizer does not contain nanochat's nine chat-control tokens.
- Save SFT outputs separately from `autoresearch` as `autoresearch-sft`.
- Evaluate with fixed prompts and reproducible generation settings.
- Describe the result as custom-tokenizer SFT; do not present it as an official
  nanochat chat score.
- Remove the RunPod pod after artifacts and logs are recovered.

## Design

Add a small local SFT data/rendering module that reads JSONL conversations and
renders them into token IDs while masking the user/prompt tokens from the loss.
Add a training entry point that loads the native autoresearch checkpoint,
fine-tunes the model for a bounded token/step budget, and writes a standalone
checkpoint plus metadata. Add a chat evaluator that reloads that checkpoint,
generates deterministic replies for fixed prompts, and records timing, device,
memory, and outputs.

The initial dataset is a small, openly downloadable instruction/conversation
sample suitable for a reproducible smoke run. The runbook will record the exact
dataset source, revision, token count, steps, and hyperparameters. A larger SFT
run can be repeated later without changing the code path.

## Acceptance criteria

- Base checkpoint and tokenizer files remain byte-for-byte unchanged.
- SFT checkpoint reloads strictly and produces a response for every fixed
  prompt.
- The evaluation log contains prompts, responses, model identity, generation
  settings, timing, and VRAM.
- SFT model card clearly distinguishes the derivative from the base model and
  explains tokenizer compatibility.
- Tests cover conversation rendering, loss masking, and checkpoint metadata.
