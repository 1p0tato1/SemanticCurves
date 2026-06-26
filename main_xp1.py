import argparse
import csv
import os

import numpy as np
from tqdm import tqdm
import random
import torch

from core import (
    load_model_and_tokenizer,
    raw_token_trajectory,
    METRICS,
    DATASET_LOADERS,
    DATASET_TASK,
    pearsonr_safe,
    auc_safe,
    get_layer_indices,
)

MAX_LEN = 128
DATASETS = ["stsb", "sick", "paws"]

# fiwed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_ALIASES = {
    "phi": "microsoft/phi-3-mini-4k-instruct",
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen3-0.6B",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_ALIASES.keys()),
        help="Short model name: 'phi', 'bert', or 'qwen'.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HF token if needed.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma-separated list of layers (e.g. '-1,-2,-3'). If empty, use automatic layer selection.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_layers",
        help="Directory where CSV results will be stored.",
    )
    return parser.parse_args()


def make_results_path(results_dir: str, hf_model_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    safe_model = hf_model_name.replace("/", "_")
    return os.path.join(results_dir, f"{safe_model}_xp1_layers.csv")


def append_results_csv(
    csv_path: str,
    model_name: str,
    layer: int,
    dataset_name: str,
    task_type: str,
    metrics_results,
):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model", "layer", "dataset", "task"] + list(metrics_results.keys())
            writer.writerow(header)

        row = [model_name, layer, dataset_name, task_type] + [
            metrics_results[m] for m in metrics_results.keys()
        ]
        writer.writerow(row)


def evaluate_dataset_xp1(
    dataset_name: str,
    tok,
    lm,
    is_decoder_only: bool,
    layer: int,
) -> dict:
    ds = DATASET_LOADERS[dataset_name]()
    task = DATASET_TASK[dataset_name]
    rows = []

    for ex in tqdm(ds, desc=f"XP1 | {dataset_name} | layer {layer}"):
        s1, s2, score = ex["text1"], ex["text2"], ex["score"]
        X = raw_token_trajectory(s1, tok, lm, is_decoder_only, MAX_LEN, layer)
        Y = raw_token_trajectory(s2, tok, lm, is_decoder_only, MAX_LEN, layer)

        vals = {metric_name: metric_fn(X, Y) for metric_name, metric_fn in METRICS.items()}
        rows.append((score, vals))

    scores = np.array([r[0] for r in rows], dtype=np.float64)

    results = {}
    for metric_name in METRICS:
        preds = np.array([r[1][metric_name] for r in rows], dtype=np.float64)

        if task == "regression":
            val = pearsonr_safe(preds, scores)
            results[metric_name] = val
        else:
            auc = auc_safe(scores, preds)
            results[metric_name] = auc

    return results


def main():
    args = parse_args()

    args.model = args.model.lower()
    if args.model not in MODEL_ALIASES:
        raise ValueError(f"Unknown model alias: {args.model}")
    hf_model_name = MODEL_ALIASES[args.model]

    tok, lm, is_decoder_only, config = load_model_and_tokenizer(
        hf_model_name,
        hf_token=args.hf_token,
    )

    layer_indices = get_layer_indices(config, args.layers)
    print(f"Evaluating model {hf_model_name} on layers: {layer_indices}")

    csv_path = make_results_path(args.results_dir, hf_model_name)

    for layer in layer_indices:
        for dataset_name in DATASETS:
            task = DATASET_TASK[dataset_name]
            metrics_results = evaluate_dataset_xp1(
                dataset_name,
                tok,
                lm,
                is_decoder_only,
                layer,
            )
            append_results_csv(
                csv_path,
                hf_model_name,
                layer,
                dataset_name,
                task,
                metrics_results,
            )
            print(f"[Model : {args.model}] layer {layer} dataset {dataset_name} -> {metrics_results}")


if __name__ == "__main__":
    main()