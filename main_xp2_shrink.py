import argparse
import csv
import os
from typing import List

import numpy as np
from tqdm import tqdm
import random
import torch

from core import (
    load_model_and_tokenizer,
    DATASET_LOADERS,
    DATASET_TASK,
    pearsonr_safe,
    mean_pool_cosine,
    endpoint_distance,
    l2_aligned_distance,
    linf_aligned_distance,
    h1_aligned_distance,
    dtw_distance,
    hausdorff_distance,
    chamfer_distance,
    trajectory_from_hidden_states,
    get_hidden_states_batch,
    auc_safe
)

MAX_LEN = 128
BATCH_SIZE = 16
DATASETS = ["stsb", "sick", "paws"]

# Fixed seed for reproducibility
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

# ========================================================
# XP2-specific constants
# ========================================================

TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
H1_LAMBDA = 0.5


def get_layer_indices_for_model(model_key: str):
    """
    Choose layers depending on the short model name.
    """
    model_key = model_key.lower()
    if model_key == "phi":
        return [-25, -10, -1, 0]
    elif model_key == "qwen":
        return [-22, -6, -1, 0]
    elif model_key == "bert":
        return [-23, -6, -1, 0]


# ============================================================
# Shrink + metric wrappers
# ============================================================

def shrink_traj(X: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Mean-preserving shrink around the token-wise mean.
    """
    mu = X.mean(dim=0, keepdim=True)
    return mu + (1.0 - tau) * (X - mu)


def to_float(x):
    return float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x)


def inverse_distance(x: torch.Tensor, eps: float = 1e-12) -> float:
    return 1.0 / (to_float(x) + eps)


def make_shrink_metrics(tau: float):
    """
    Build a dictionary of metrics after shrink for a given tau.
    """

    def cos_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return to_float(mean_pool_cosine(Xp, Yp))

    def inv_endpoint_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(endpoint_distance(Xp, Yp))

    def inv_l2_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(l2_aligned_distance(Xp, Yp))

    def inv_linf_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(linf_aligned_distance(Xp, Yp))

    def inv_h1_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(h1_aligned_distance(Xp, Yp, lam=H1_LAMBDA))

    def inv_dtw_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(dtw_distance(Xp, Yp))

    def inv_haus_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(hausdorff_distance(Xp, Yp))

    def inv_cham_f(X, Y):
        Xp = shrink_traj(X, tau)
        Yp = shrink_traj(Y, tau)
        return inverse_distance(chamfer_distance(Xp, Yp))

    return {
        "cos_f": cos_f,
        "inv_endpoint_f": inv_endpoint_f,
        "inv_l2_f": inv_l2_f,
        "inv_linf_f": inv_linf_f,
        "inv_h1_f": inv_h1_f,
        "inv_dtw_f": inv_dtw_f,
        "inv_haus_f": inv_haus_f,
        "inv_cham_f": inv_cham_f,
    }


# ============================================================
# Args & results
# ============================================================

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
        "--results_dir",
        type=str,
        default="results",
        help="Directory where CSV results will be stored.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for model forward passes.",
    )
    return parser.parse_args()


def make_results_path(results_dir: str, hf_model_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    safe_model = hf_model_name.replace("/", "_")
    return os.path.join(results_dir, f"{safe_model}_xp2_shrink.csv")


def append_results_csv(
    csv_path: str,
    model_name: str,
    layer: int,
    dataset_name: str,
    tau: float,
    task_type: str,  # Add this
    metric_corrs: dict,
):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model", "layer", "dataset", "task", "tau"] + list(metric_corrs.keys())
            writer.writerow(header)

        row = [model_name, layer, dataset_name, task_type, tau] + [
            metric_corrs[m] for m in metric_corrs.keys()
        ]
        writer.writerow(row)


# ============================================================
# Optimized evaluation with batch processing
# ============================================================

def process_batch_hidden_states(
        batch_texts: list,
        tok,
        lm,
        is_decoder_only: bool,
) -> list:
    """
    Process a batch of texts and return their hidden states.
    """
    # Get all hidden states for the batch
    hidden_states_list = get_hidden_states_batch(
        batch_texts, tok, lm, is_decoder_only, MAX_LEN
    )
    return hidden_states_list


def evaluate_dataset_xp2_shrink(
        dataset_name: str,
        tok,
        lm,
        is_decoder_only: bool,
        layer_indices: List[int],
        batch_size: int = 16,
) -> dict:
    """
    Optimized version: cache hidden states once and process in batches.
    Now with task-aware evaluation (Pearson for regression, AUC for binary).
    """
    ds = DATASET_LOADERS[dataset_name]()
    task_type = DATASET_TASK[dataset_name]  # Get task type
    print(f"Task type for {dataset_name}: {task_type}")

    print(f"XP2 SHRINK | Caching hidden states for {dataset_name} (batch_size={batch_size})")

    cached_data = []
    batch_size = min(batch_size, len(ds))

    for i in tqdm(range(0, len(ds), batch_size), desc=f"Batch processing {dataset_name}"):
        batch = ds[i:i + batch_size]

        texts1 = [ex["text1"] for ex in batch]
        texts2 = [ex["text2"] for ex in batch]
        scores = [ex["score"] for ex in batch]

        hs1_list, attn1_list, ids1_list = process_batch_hidden_states(
            texts1, tok, lm, is_decoder_only
        )
        hs2_list, attn2_list, ids2_list = process_batch_hidden_states(
            texts2, tok, lm, is_decoder_only
        )

        for j in range(len(batch)):
            cached_data.append((
                texts1[j],
                texts2[j],
                scores[j],
                hs1_list[j],
                attn1_list[j],
                ids1_list[j],
                hs2_list[j],
                attn2_list[j],
                ids2_list[j],
            ))

    results = {}

    for layer in layer_indices:
        print(f"XP2 SHRINK | Processing layer {layer} for {dataset_name}")

        layer_cache = []
        for data in tqdm(cached_data, desc=f"Building trajectories for layer {layer}"):
            (text1, text2, score,
             hs1, attn1, ids1,
             hs2, attn2, ids2) = data

            X = trajectory_from_hidden_states(hs1, attn1, ids1, tok, layer)
            Y = trajectory_from_hidden_states(hs2, attn2, ids2, tok, layer)
            layer_cache.append((score, X, Y))

        tau_to_metric_corrs = {}

        for tau in TAUS:
            shrink_metrics = make_shrink_metrics(tau)
            metric_values = {name: [] for name in shrink_metrics.keys()}
            scores = []

            for score, X, Y in tqdm(layer_cache, desc=f"tau={tau}"):
                scores.append(float(score))
                for name, fn in shrink_metrics.items():
                    val = fn(X, Y)
                    metric_values[name].append(val)

            scores_arr = np.array(scores, dtype=np.float64)

            metric_corrs = {}
            for name in shrink_metrics.keys():
                preds = np.array(metric_values[name], dtype=np.float64)

                if task_type == "regression":
                    corr = pearsonr_safe(preds, scores_arr)
                elif task_type == "binary":
                    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-12)
                    corr = auc_safe(scores_arr, preds_norm)
                else:
                    corr = pearsonr_safe(preds, scores_arr)

                metric_corrs[name] = corr

            tau_to_metric_corrs[tau] = metric_corrs

        results[layer] = tau_to_metric_corrs

    return results


# ============================================================
# Main
# ============================================================

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

    layer_indices = get_layer_indices_for_model(args.model)
    batch_size = args.batch_size

    print(f"Evaluating XP2 SHRINK for model {hf_model_name} on layers: {layer_indices}")
    print(f"TAUS = {TAUS}, H1_LAMBDA = {H1_LAMBDA}")
    print(f"Batch size = {batch_size}")

    csv_path = make_results_path(args.results_dir, hf_model_name)

    # Evaluate each dataset
    for dataset_name in DATASETS:
        task_type = DATASET_TASK[dataset_name]
        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'=' * 60}")

        layer_results = evaluate_dataset_xp2_shrink(
            dataset_name,
            tok,
            lm,
            is_decoder_only,
            layer_indices,
            batch_size,
        )

        # Write results to CSV
        for layer, tau_to_metric_corrs in layer_results.items():
            for tau, metric_corrs in tau_to_metric_corrs.items():
                append_results_csv(
                    csv_path,
                    hf_model_name,
                    layer,
                    dataset_name,
                    tau,
                    task_type,
                    metric_corrs,
                )
                print(f"[Model: {args.model}] layer {layer} dataset {dataset_name} tau={tau} -> {metric_corrs}")


if __name__ == "__main__":
    main()