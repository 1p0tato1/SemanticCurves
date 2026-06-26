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
    auc_safe,
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
    _aligned_pair,  # Need this for custom alignment
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

# Different alignment lengths to test
ALIGNMENT_LENGTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]   # T values
H1_LAMBDA = 0.5


def get_layer_indices_for_model(model_key: str):
    """Choose layers depending on the short model name."""
    model_key = model_key.lower()
    if model_key == "phi":
        return [-25, -10, -1, 0]
    else:
        return [-1, -2]


# ============================================================
# Custom aligned metrics with variable T
# ============================================================
def make_aligned_metrics(T: int):
    """
    Build a dictionary of metrics with a fixed alignment length T.
    This is like METRICS from core.py but with configurable T.
    """

    def l2_aligned_T(X: torch.Tensor, Y: torch.Tensor) -> float:
        """L2 distance with fixed T alignment."""
        Xr, Yr = _aligned_pair(X, Y, T)
        sq = ((Xr - Yr) ** 2).sum(dim=-1)
        dt = 1.0 / (T - 1) if T > 1 else 1.0
        integral = sq.sum() * dt
        return float(torch.sqrt(integral + 1e-12).cpu())

    def linf_aligned_T(X: torch.Tensor, Y: torch.Tensor) -> float:
        """L-infinity distance with fixed T alignment."""
        Xr, Yr = _aligned_pair(X, Y, T)
        return float(torch.linalg.norm(Xr - Yr, dim=-1).max().cpu())

    def h1_aligned_T(X: torch.Tensor, Y: torch.Tensor) -> float:
        """H1 distance with fixed T alignment."""
        Xr, Yr = _aligned_pair(X, Y, T)
        Xr = Xr.float()
        Yr = Yr.float()

        if not torch.isfinite(Xr).all() or not torch.isfinite(Yr).all():
            return float('inf')

        sq = ((Xr - Yr) ** 2).sum(dim=-1)
        dt = 1.0 / (T - 1) if T > 1 else 1.0
        int_pos = sq.sum() * dt

        if T > 1:
            VX = (Xr[1:] - Xr[:-1]) / dt
            VY = (Yr[1:] - Yr[:-1]) / dt
            sqv = ((VX - VY) ** 2).sum(dim=-1)
            int_vel = sqv.sum() * dt
        else:
            int_vel = 0.0

        val = int_pos + H1_LAMBDA * int_vel
        return float(torch.sqrt(torch.clamp(val, min=0.0) + 1e-12).cpu())

    def inv_l2_T(X, Y):
        return 1.0 / (l2_aligned_T(X, Y) + 1e-12)

    def inv_linf_T(X, Y):
        return 1.0 / (linf_aligned_T(X, Y) + 1e-12)

    def inv_h1_T(X, Y):
        return 1.0 / (h1_aligned_T(X, Y) + 1e-12)

    # Non-aligned metrics (independent of T)
    def cos_f(X, Y):
        return float(mean_pool_cosine(X, Y))

    def inv_endpoint_f(X, Y):
        return 1.0 / (float(endpoint_distance(X, Y).cpu()) + 1e-12)

    def inv_dtw_f(X, Y):
        return 1.0 / (float(dtw_distance(X, Y).cpu()) + 1e-12)

    def inv_haus_f(X, Y):
        return 1.0 / (float(hausdorff_distance(X, Y).cpu()) + 1e-12)

    def inv_cham_f(X, Y):
        return 1.0 / (float(chamfer_distance(X, Y).cpu()) + 1e-12)

    return {
        # Non-aligned metrics (these don't change with T)
        "cos_f": cos_f,
        "inv_endpoint_f": inv_endpoint_f,
        "inv_dtw_f": inv_dtw_f,
        "inv_haus_f": inv_haus_f,
        "inv_cham_f": inv_cham_f,

        # Aligned metrics (these depend on T)
        "inv_l2_f": inv_l2_T,
        "inv_linf_f": inv_linf_T,
        "inv_h1_f": inv_h1_T,
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
        default="results_layers",
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
    return os.path.join(results_dir, f"{safe_model}_xp2_alignment.csv")


def append_results_csv(
        csv_path: str,
        model_name: str,
        layer: int,
        dataset_name: str,
        T: int,  # Alignment length
        task_type: str,
        metric_corrs: dict,
):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model", "layer", "dataset", "task", "T"] + list(metric_corrs.keys())
            writer.writerow(header)

        row = [model_name, layer, dataset_name, task_type, T] + [
            metric_corrs[m] for m in metric_corrs.keys()
        ]
        writer.writerow(row)


# ============================================================
# Evaluation with batch processing
# ============================================================

def process_batch_hidden_states(
        batch_texts: list,
        tok,
        lm,
        is_decoder_only: bool,
) -> list:
    """Process a batch of texts and return their hidden states."""
    hidden_states_list = get_hidden_states_batch(
        batch_texts, tok, lm, is_decoder_only, MAX_LEN
    )
    return hidden_states_list


def evaluate_dataset_xp2_alignment(
        dataset_name: str,
        tok,
        lm,
        is_decoder_only: bool,
        layer_indices: List[int],
        batch_size: int = 16,
) -> dict:
    """
    Evaluate the effect of varying alignment length T.
    Cache hidden states once, then test different T values.
    """
    ds = DATASET_LOADERS[dataset_name]()
    task_type = DATASET_TASK[dataset_name]
    print(f"Task type for {dataset_name}: {task_type}")

    # Step 1: Cache all hidden states (same as XP2)
    print(f"XP2 ALIGNMENT | Caching hidden states for {dataset_name} (batch_size={batch_size})")

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
        print(f"XP2 ALIGNMENT | Processing layer {layer} for {dataset_name}")

        # Build trajectories for this layer
        layer_cache = []
        for data in tqdm(cached_data, desc=f"Building trajectories for layer {layer}"):
            (text1, text2, score,
             hs1, attn1, ids1,
             hs2, attn2, ids2) = data

            X = trajectory_from_hidden_states(hs1, attn1, ids1, tok, layer)
            Y = trajectory_from_hidden_states(hs2, attn2, ids2, tok, layer)
            layer_cache.append((score, X, Y))

        # For each alignment length T
        T_to_metric_corrs = {}

        for T in ALIGNMENT_LENGTHS:
            print(f"XP2 ALIGNMENT | Testing T={T} for layer {layer}")

            # Create metrics with this T
            aligned_metrics = make_aligned_metrics(T)
            metric_values = {name: [] for name in aligned_metrics.keys()}
            scores = []

            for score, X, Y in tqdm(layer_cache, desc=f"T={T}"):
                scores.append(float(score))
                for name, fn in aligned_metrics.items():
                    val = fn(X, Y)
                    metric_values[name].append(val)

            scores_arr = np.array(scores, dtype=np.float64)

            metric_corrs = {}
            for name in aligned_metrics.keys():
                preds = np.array(metric_values[name], dtype=np.float64)

                if task_type == "regression":
                    corr = pearsonr_safe(preds, scores_arr)
                elif task_type == "binary":
                    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-12)
                    corr = auc_safe(scores_arr, preds_norm)
                else:
                    corr = pearsonr_safe(preds, scores_arr)

                metric_corrs[name] = corr

            T_to_metric_corrs[T] = metric_corrs

        results[layer] = T_to_metric_corrs

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

    print(f"Evaluating XP2 ALIGNMENT for model {hf_model_name} on layers: {layer_indices}")
    print(f"ALIGNMENT_LENGTHS = {ALIGNMENT_LENGTHS}")
    print(f"Batch size = {batch_size}")

    csv_path = make_results_path(args.results_dir, hf_model_name)

    for dataset_name in DATASETS:
        task_type = DATASET_TASK[dataset_name]

        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {dataset_name} (task: {task_type})")
        print(f"{'=' * 60}")

        layer_results = evaluate_dataset_xp2_alignment(
            dataset_name,
            tok,
            lm,
            is_decoder_only,
            layer_indices,
            batch_size,
        )

        for layer, T_to_metric_corrs in layer_results.items():
            for T, metric_corrs in T_to_metric_corrs.items():
                append_results_csv(
                    csv_path,
                    hf_model_name,
                    layer,
                    dataset_name,
                    T,
                    task_type,
                    metric_corrs,
                )
                print(f"[Model: {args.model}] layer {layer} dataset {dataset_name} T={T} -> {metric_corrs}")


if __name__ == "__main__":
    main()