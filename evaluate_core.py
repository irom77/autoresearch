"""Run nanochat's pinned CORE evaluator against the exported base model.

The evaluator is loaded from a checkout of nanochat revision e85db6b because
the autoresearch repository intentionally vendors only the checkpoint loader.
The model and tokenizer adapters below keep the evaluation compatible with the
autoresearch reserved-token tokenizer.
"""

import argparse
import importlib.util
import json
import sys
import csv
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch
import requests
import yaml


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CoreTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_bos_token_id(self):
        return self.tokenizer.get_bos_token_id()

    def __call__(self, texts, prepend=None):
        return self.tokenizer.encode(texts, prepend=prepend)


def evaluate_model(core_eval, model, tokenizer, device, max_per_task):
    bundle_dir = Path.home() / ".cache" / "nanochat" / "eval_bundle"
    if not bundle_dir.exists():
        bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "eval_bundle.zip"
            with requests.get(
                "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip",
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                with archive.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(tmp)
            shutil.move(str(Path(tmp) / "eval_bundle"), bundle_dir)

    config = yaml.safe_load((bundle_dir / "core.yaml").read_text())
    random_baselines = {}
    with (bundle_dir / "eval_meta_data.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results, centered = {}, {}
    for task in config["icl_tasks"]:
        label = task["label"]
        meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        data_path = bundle_dir / "eval_data" / meta["dataset_uri"]
        data = [json.loads(line) for line in data_path.read_text().splitlines()]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        accuracy = core_eval.evaluate_task(model, tokenizer, data, device, meta)
        results[label] = accuracy
        baseline = random_baselines[label]
        centered[label] = (accuracy - 0.01 * baseline) / (1.0 - 0.01 * baseline)
        print(f"{label}: accuracy={accuracy:.4f} centered={centered[label]:.4f}", flush=True)
    return {"results": results, "centered_results": centered,
            "core_metric": sum(centered.values()) / len(centered)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nanochat-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("nanochat_export/base_checkpoints/autoresearch"))
    parser.add_argument("--max-per-task", type=int, default=-1)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent / "vendor" / "nanochat_compat"))
    sys.path.insert(0, str(Path(__file__).parent))
    from load_model import load_model
    from prepare import Tokenizer

    core_eval = load_module(args.nanochat_source / "nanochat" / "core_eval.py", "pinned_core_eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_model(args.checkpoint, device=device)
    tokenizer = CoreTokenizer(Tokenizer.from_directory())
    model.max_seq_len = metadata["model_config"].get("sequence_len", 2048)

    print(json.dumps({
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "model_commit": metadata.get("commit"),
        "tokenizer_vocab_size": tokenizer.tokenizer.get_vocab_size(),
        "val_bpb": metadata.get("val_bpb"),
        "max_per_task": args.max_per_task,
    }), flush=True)
    result = evaluate_model(core_eval, model, tokenizer, device, args.max_per_task)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
