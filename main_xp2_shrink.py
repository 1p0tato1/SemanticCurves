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
    DATASET_LOADERS,
    DATASET_TASK,
    pearsonr_safe,
    # base metrics used inside shrink wrappers
    mean_pool_cosine,
    endpoint_distance,
    l2_aligned_distance,
    linf_aligned_distance,
    h1_aligned_distance,
    dtw_distance,
    hausdorff_distance,
    chamfer_distance,
)

MAX_LEN = 128

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

TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # tau grid in [0,1]
H1_LAMBDA = 0.5
layer_indices = [-1, -2]


def get_layer_indices_for_model(model_key: str):
    """
    Choose layers depending on the short model name.

    - For phi (phi-3-mini): use [-25, -10, 0]
    - For others: use the last two layers [-1, -2]
    """
    model_key = model_key.lower()
    if model_key == "phi":
        return [-25, -10, 0]
    else:
        return [-1, -2]


# ============================================================
# Shrink + metric wrappers
# ============================================================

def shrink_traj(X: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Mean-preserving shrink around the token-wise mean.

    mu = mean over tokens (per dimension)
    f_tau(X_i) = mu + (1 - tau) * (X_i - mu)

    tau in [0, 1]:
      tau = 0 -> identity (no change)
      tau = 1 -> trajectory collapses to mu
    """
    mu = X.mean(dim=0, keepdim=True)
    return mu + (1.0 - tau) * (X - mu)


def to_float(x):
    return float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x)


def inverse_distance(x: torch.Tensor, eps: float = 1e-12) -> float:
    return 1.0 / (to_float(x) + eps)


def make_shrink_metrics(tau: float):
    """
    Build a dictionary of metrics after shrink for a given tau,
    reusing the base distances from core.py.

    Returns a dict of callables that take (X, Y) and return floats.
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
        default="results_layers",
        help="Directory where CSV results will be stored (same as XP1 by default).",
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
    metric_corrs: dict,
):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["model", "layer", "dataset", "tau"] + list(metric_corrs.keys())
            writer.writerow(header)

        row = [model_name, layer, dataset_name, tau] + [
            metric_corrs[m] for m in metric_corrs.keys()
        ]
        writer.writerow(row)


# ============================================================
# Evaluation core (XP2 shrink)
# ============================================================

def evaluate_dataset_xp2_shrink(
    dataset_name: str,
    tok,
    lm,
    is_decoder_only: bool,
    layer: int,
) -> dict:
    """
    For a given dataset and layer:
      - cache trajectories (score, X, Y)
      - for each tau in TAUS:
          * compute shrink metrics for all pairs
          * compute Pearson correlation of each metric with scores
      - return a dict: tau -> {metric_name: correlation}
    """
    ds = DATASET_LOADERS[dataset_name]()
    task = DATASET_TASK[dataset_name]

    cache = []
    for ex in tqdm(ds, desc=f"XP2 SHRINK | cache {dataset_name} | layer {layer}"):
        s1, s2, score = ex["text1"], ex["text2"], ex["score"]
        X = raw_token_trajectory(s1, tok, lm, is_decoder_only, MAX_LEN, layer)
        Y = raw_token_trajectory(s2, tok, lm, is_decoder_only, MAX_LEN, layer)
        cache.append((score, X, Y))

    tau_to_metric_corrs = {}

    for tau in TAUS:
        shrink_metrics = make_shrink_metrics(tau)
        metric_values = {name: [] for name in shrink_metrics.keys()}
        scores = []

        for score, X, Y in tqdm(cache, desc=f"XP2 SHRINK | {dataset_name} | layer {layer} | tau {tau}"):
            scores.append(float(score))
            for name, fn in shrink_metrics.items():
                val = fn(X, Y)
                metric_values[name].append(val)

        scores_arr = np.array(scores, dtype=np.float64)

        metric_corrs = {}
        for name in shrink_metrics.keys():
            preds = np.array(metric_values[name], dtype=np.float64)
            corr = pearsonr_safe(preds, scores_arr)
            metric_corrs[name] = corr

        tau_to_metric_corrs[tau] = metric_corrs

    return tau_to_metric_corrs


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
    print(f"Evaluating XP2 SHRINK for model {hf_model_name} on layers : {layer_indices}")

    print(f"TAUS = {TAUS}, H1_LAMBDA = {H1_LAMBDA}")

    csv_path = make_results_path(args.results_dir, hf_model_name)

    for layer in layer_indices:
        for dataset_name in ["stsb"]: #["stsb", "sick", "paws"]
            tau_to_metric_corrs = evaluate_dataset_xp2_shrink(
                dataset_name,
                tok,
                lm,
                is_decoder_only,
                layer,
            )
            for tau, metric_corrs in tau_to_metric_corrs.items():
                append_results_csv(
                    csv_path,
                    hf_model_name,
                    layer,
                    dataset_name,
                    tau,
                    metric_corrs,
                )
                print(f"[Model: {args.model}] layer {layer} dataset {dataset_name} tau={tau} -> {metric_corrs}")


if __name__ == "__main__":
    main()