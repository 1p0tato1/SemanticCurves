import argparse
import datetime
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from core import (
    load_model_and_tokenizer,
    DATASET_LOADERS,
    DATASET_TASK,
    pearsonr_safe,
    auc_safe,
    _aligned_pair,
    mean_pool_cosine,
    l2_aligned_distance,
    h1_aligned_distance,
    chamfer_distance,
    raw_token_trajectory,
    get_hidden_states_batch,
    trajectory_from_hidden_states,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.backends.mps.is_available():
    device = "mps"

# ============================================================
# GLOBAL CONFIG
# ============================================================

# Model aliases and default model
MODEL_ALIASES = {
    "phi": "microsoft/phi-3-mini-4k-instruct",
    "bert": "bert-base-uncased",
    "qwen": "Qwen/Qwen3-0.6B",
}
DEFAULT_MODEL_ALIAS = "bert"

# Datasets to evaluate at test time
DEFAULT_DATASETS = ["stsb", "sick", "paws"]

# Default training / evaluation settings
DEFAULT_MAX_LEN     = 128
DEFAULT_NUM_EPOCHS  = 5
DEFAULT_TRAIN_BATCH = 4
DEFAULT_EVAL_BATCH  = 16
DEFAULT_T           = 10
DEFAULT_LR          = 5e-6
DEFAULT_WD          = 0.01

DEFAULT_RESULTS_DIR = "results"

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
        return [-6, -3, -1, 0]

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# Soft-DTW and helpers
# ============================================================

def softmin3(a, b, c, gamma):
    m = torch.stack([a, b, c], dim=0)
    return -gamma * torch.logsumexp(-m / gamma, dim=0)

def soft_dtw_distance(X: torch.Tensor, Y: torch.Tensor, gamma: float = 0.001, eps: float = 1e-12) -> torch.Tensor:
    a, d = X.shape
    b, _ = Y.shape

    D = torch.cdist(X, Y).clamp(max=20.0)  # [a, b]
    inf = torch.tensor(float("inf"), device=X.device)
    R = torch.full((a + 1, b + 1), inf, device=X.device)
    R[0, 0] = 0.0

    for i in range(1, a + 1):
        for j in range(1, b + 1):
            R[i, j] = D[i - 1, j - 1] + softmin3(
                R[i - 1, j],
                R[i, j - 1],
                R[i - 1, j - 1],
                gamma,
            )
    return R[a, b].clamp_min(0.0) + eps

def pearson_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Loss = 1 - Pearson correlation(x, y), with safety checks.
    """
    x = x - x.mean()
    y = y - y.mean()

    if torch.isnan(x).any() or torch.isinf(x).any():
        return x.sum() * 0.0
    if torch.isnan(y).any() or torch.isinf(y).any():
        return y.sum() * 0.0

    vx = (x ** 2).sum()
    vy = (y ** 2).sum()

    if torch.isnan(vx) or torch.isinf(vx) or torch.isnan(vy) or torch.isinf(vy):
        return (vx + vy) * 0.0

    if vx.item() <= eps or vy.item() <= eps:
        return (vx + vy) * 0.0

    denom = torch.sqrt(vx) * torch.sqrt(vy)
    corr = (x * y).sum() / (denom + eps)

    if torch.isnan(corr) or torch.isinf(corr):
        return corr * 0.0

    return 1.0 - corr

def token_trajectory_train(
    text: str,
    tok,
    lm,
    is_decoder_only: bool,
    max_len: int,
    layer: int,
) -> torch.Tensor:
    batch = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    forward_kwargs = dict(
        **batch,
        output_hidden_states=True,
        return_dict=True,
    )
    if is_decoder_only:
        forward_kwargs["use_cache"] = False

    outputs = lm(**forward_kwargs)

    H = outputs.hidden_states[layer][0]
    attn = batch["attention_mask"][0].bool()

    #new
    input_ids = batch["input_ids"][0]

    mask = attn.clone()

    special_ids = set(tok.all_special_ids)
    if len(special_ids) > 0:
        special_mask = torch.zeros_like(mask)
        for sid in special_ids:
            special_mask |= (input_ids == sid)
        mask &= ~special_mask

    X = H[mask]
    X = F.normalize(X, p=2, dim=-1, eps=1e-8)

    if X.shape[0] == 0:
        X = H[attn]
        X = F.normalize(X, p=2, dim=-1, eps=1e-8)

    return X

def get_trajectory_single(
    text: str,
    tok,
    lm,
    is_decoder_only: bool,
    max_len: int,
    layer: int,
) -> torch.Tensor:
    return raw_token_trajectory(text, tok, lm, is_decoder_only, max_len, layer)


# For MPS setup
def move_loss_tensors_for_backend(dists: torch.Tensor, scores: torch.Tensor):
    if device == "mps":
        return dists.float().cpu(), scores.float().cpu()
    return dists, scores

def metric_device_tensor(X, Y, device: str):
    if device == "mps":
        return X.float().cpu(), Y.float().cpu()
    return X, Y

# ============================================================
# Similarity computation for one pair
# ============================================================
def compute_metrics_from_trajectories(
    X: torch.Tensor,
    Y: torch.Tensor,
    T: int = DEFAULT_T,
) -> Dict[str, float]:
    X_aligned, Y_aligned = _aligned_pair(X, Y, T)
    X_metric, Y_metric = metric_device_tensor(X_aligned, Y_aligned, device)

    dist_soft = soft_dtw_distance(X_metric, Y_metric)
    softdtw = -dist_soft.item()

    euclid_l2 = -l2_aligned_distance(X, Y).item()
    h1 = -h1_aligned_distance(X, Y, lam=0.5).item()
    chamfer = -chamfer_distance(X, Y).item()

    X_mean = X.mean(dim=0)
    Y_mean = Y.mean(dim=0)
    cos = F.cosine_similarity(X_mean, Y_mean, dim=0).item()

    return {
        "softdtw": softdtw,
        "euclid_l2": euclid_l2,
        "h1": h1,
        "chamfer": chamfer,
        "cos": cos,
    }


def compute_pair_similarities_from_hidden_states(
    hs1,
    attn1,
    ids1,
    hs2,
    attn2,
    ids2,
    tok,
    layer: int,
    T: int = DEFAULT_T,
) -> Dict[str, float]:
    X = trajectory_from_hidden_states(hs1, attn1, ids1, tok, layer)
    Y = trajectory_from_hidden_states(hs2, attn2, ids2, tok, layer)
    return compute_metrics_from_trajectories(X, Y, T=T)


def compute_pair_similarities(
    s1: str,
    s2: str,
    tok,
    lm,
    is_decoder_only: bool,
    max_len: int,
    layer: int,
    T: int = DEFAULT_T,
) -> Dict[str, float]:
    X = get_trajectory_single(s1, tok, lm, is_decoder_only, max_len, layer)
    Y = get_trajectory_single(s2, tok, lm, is_decoder_only, max_len, layer)
    return compute_metrics_from_trajectories(X, Y, T=T)


# ============================================================
# Fine-tuning on STS-B with dev-based model selection
# ============================================================
def finetune_on_stsb(
    tok,
    lm,
    is_decoder_only: bool,
    max_len: int,
    layer: int,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    train_batch: int = DEFAULT_TRAIN_BATCH,
    T: int = DEFAULT_T,
    lr: float = DEFAULT_LR,
    wd: float = DEFAULT_WD,
):
    """
    Fine-tune the model on STS-B train split using
      - Soft-DTW distance between token trajectories
      - Similarity = -distance
      - Loss = 1 - Pearson correlation(similarity, human scores)

    Model selection:
      - After each epoch, evaluate on STS-B validation split
        using cosine similarity between mean-pooled embeddings.
      - Track Pearson correlation on dev.
      - Keep the checkpoint with best dev correlation.
    """
    from datasets import load_dataset

    dstrain = load_dataset("sentence-transformers/stsb", split="train")
    dsdev   = load_dataset("sentence-transformers/stsb", split="validation")

    train_loader = DataLoader(dstrain, batch_size=train_batch, shuffle=True)
    dev_loader   = DataLoader(dsdev,   batch_size=train_batch, shuffle=False)

    optimizer = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Fine-tuning on STS-B | epochs={num_epochs}, batch_size={train_batch}, total_steps={total_steps}")

    best_dev_corr = -float("inf")
    best_state_dict = None

    for epoch in range(num_epochs):
        lm.train()
        total_loss = 0.0
        dists_s = []
        scores_s = []

        for ex in tqdm(train_loader, desc=f"XP3 FT — Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            s1_batch = ex["sentence1"]
            s2_batch = ex["sentence2"]
            s = ex["score"].float().to(device)

            dists = []
            for s1_text, s2_text in zip(s1_batch, s2_batch):
                X = token_trajectory_train(s1_text, tok, lm, is_decoder_only, max_len, layer)
                Y = token_trajectory_train(s2_text, tok, lm, is_decoder_only, max_len, layer)

                X_aligned, Y_aligned = _aligned_pair(X, Y, T)
                X_loss, Y_loss = metric_device_tensor(X_aligned, Y_aligned, device)
                dist = soft_dtw_distance(X_loss, Y_loss)
                if torch.isnan(dist) or torch.isinf(dist):
                    dist = X.sum() * 0.0

                dists.append(dist)

            dists = torch.stack(dists)

            if torch.isnan(dists).any() or torch.isinf(dists).any():
                optimizer.zero_grad()
                continue

            dists_loss, s_loss = move_loss_tensors_for_backend(dists, s)

            sim = -dists_loss
            loss = pearson_loss(sim, s_loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(lm.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.detach().item()
            dists_s.append(sim.detach().cpu().numpy())
            scores_s.append(s_loss.detach().cpu().numpy())

        dists_s = np.concatenate(dists_s)
        scores_s = np.concatenate(scores_s)
        train_corr = np.corrcoef(scores_s, dists_s)[0, 1]
        avg_loss = total_loss / len(train_loader)

        lm.eval()
        with torch.no_grad():
            dev_sims = []
            dev_scores = []

            for ex in tqdm(dev_loader, desc=f"Dev eval — Epoch {epoch + 1}", leave=False):
                s1_batch = list(ex["sentence1"])
                s2_batch = list(ex["sentence2"])
                s_dev = ex["score"].float()

                hs1_list, attn1_list, ids1_list = get_hidden_states_batch(
                    s1_batch, tok, lm, is_decoder_only, max_len
                )
                hs2_list, attn2_list, ids2_list = get_hidden_states_batch(
                    s2_batch, tok, lm, is_decoder_only, max_len
                )

                for hs1, attn1, ids1, hs2, attn2, ids2, score_val in zip(
                        hs1_list, attn1_list, ids1_list,
                        hs2_list, attn2_list, ids2_list,
                        s_dev,
                ):
                    X = trajectory_from_hidden_states(hs1, attn1, ids1, tok, layer)
                    Y = trajectory_from_hidden_states(hs2, attn2, ids2, tok, layer)
                    sim_cos = mean_pool_cosine(X, Y)

                    dev_sims.append(sim_cos)
                    dev_scores.append(score_val.item())

            dev_sims = np.array(dev_sims, dtype=np.float64)
            dev_scores = np.array(dev_scores, dtype=np.float64)
            dev_corr = pearsonr_safe(dev_sims, dev_scores)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | "
            f"Train r (Soft-DTW sim): {train_corr:.4f} | "
            f"Dev r (cosine mean-pool): {dev_corr:.4f}"
        )

        if dev_corr > best_dev_corr:
            best_dev_corr = dev_corr
            best_state_dict = lm.state_dict().copy()
            print(f"New best dev corr: {best_dev_corr:.4f} (saving checkpoint in memory)")

    if best_state_dict is not None:
        lm.load_state_dict(best_state_dict)
        print(f"Loaded best checkpoint with dev corr = {best_dev_corr:.4f}")
    else:
        print("Warning: no best checkpoint stored, using last epoch weights.")

    lm.eval()
    return lm

# ============================================================
# Evaluation on test datasets
# ============================================================
def evaluate_pipeline_on_dataset(
    dataset_name: str,
    tok,
    lm,
    is_decoder_only: bool,
    layer: int,
    max_len: int,
    T: int = DEFAULT_T,
    eval_batch: int = DEFAULT_EVAL_BATCH,
) -> Dict[str, float]:

    ds = DATASET_LOADERS[dataset_name]()
    task_type = DATASET_TASK[dataset_name]

    soft_sims = []
    euclid_l2_sims = []
    h1_sims = []
    chamfer_sims = []
    cos_sims = []
    scores = []

    for start in tqdm(
            range(0, len(ds), eval_batch),
            desc=f"XP3 | dataset={dataset_name} | layer={layer}",
    ):
        batch_examples = ds[start:start + eval_batch]
        s1_batch = [ex["text1"] for ex in batch_examples]
        s2_batch = [ex["text2"] for ex in batch_examples]
        score_batch = [float(ex["score"]) for ex in batch_examples]

        hs1_list, attn1_list, ids1_list = get_hidden_states_batch(
            s1_batch, tok, lm, is_decoder_only, max_len
        )
        hs2_list, attn2_list, ids2_list = get_hidden_states_batch(
            s2_batch, tok, lm, is_decoder_only, max_len
        )

        for hs1, attn1, ids1, hs2, attn2, ids2, score in zip(
                hs1_list, attn1_list, ids1_list,
                hs2_list, attn2_list, ids2_list,
                score_batch,
        ):
            sims = compute_pair_similarities_from_hidden_states(
                hs1, attn1, ids1,
                hs2, attn2, ids2,
                tok,
                layer,
                T=T,
            )

            soft_sims.append(sims["softdtw"])
            euclid_l2_sims.append(sims["euclid_l2"])
            h1_sims.append(sims["h1"])
            chamfer_sims.append(sims["chamfer"])
            cos_sims.append(sims["cos"])
            scores.append(score)

    soft_sims = np.array(soft_sims, dtype=np.float64)
    euclid_l2_sims = np.array(euclid_l2_sims, dtype=np.float64)
    h1_sims = np.array(h1_sims, dtype=np.float64)
    chamfer_sims = np.array(chamfer_sims, dtype=np.float64)
    cos_sims = np.array(cos_sims, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)

    metric_corrs = {}

    if task_type == "regression":
        metric_corrs["softdtw"] = pearsonr_safe(soft_sims, scores)
        metric_corrs["euclid_l2"] = pearsonr_safe(euclid_l2_sims, scores)
        metric_corrs["h1"] = pearsonr_safe(h1_sims, scores)
        metric_corrs["chamfer"] = pearsonr_safe(chamfer_sims, scores)
        metric_corrs["cos"] = pearsonr_safe(cos_sims, scores)

    else :
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-12)

        metric_corrs["softdtw"] = auc_safe(scores, norm(soft_sims))
        metric_corrs["euclid_l2"] = auc_safe(scores, norm(euclid_l2_sims))
        metric_corrs["h1"] = auc_safe(scores, norm(h1_sims))
        metric_corrs["chamfer"] = auc_safe(scores, norm(chamfer_sims))
        metric_corrs["cos"] = auc_safe(scores, norm(cos_sims))

    return metric_corrs

# ============================================================
# CSV saving
# ============================================================
def make_results_path(results_dir: str, hf_model_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    safe_model = hf_model_name.replace("/", "_")
    return os.path.join(results_dir, f"{safe_model}_XP3_ft.csv")

def append_results_csv(
    csv_path: str,
    model_name: str,
    layer: int,
    dataset_name: str,
    task_type: str,
    metric_corrs: Dict[str, float],
):
    file_exists = os.path.isfile(csv_path)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["timestamp", "model", "layer", "dataset", "task"] + list(metric_corrs.keys())
            writer.writerow(header)

        row = [timestamp, model_name, layer, dataset_name, task_type] + [
            metric_corrs[m] for m in metric_corrs.keys()
        ]
        writer.writerow(row)

# ============================================================
# CLI and main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        choices=list(MODEL_ALIASES.keys()),
        help="Short model name: 'phi', 'bert', or 'qwen'.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token if needed.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for CSV results and fine-tuned model.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of epochs for fine-tuning on STS-B.",
    )
    parser.add_argument(
        "--train_batch",
        type=int,
        default=DEFAULT_TRAIN_BATCH,
        help="Batch size for fine-tuning.",
    )
    parser.add_argument(
        "--eval_batch",
        type=int,
        default=DEFAULT_EVAL_BATCH,
        help="Batch size for dev/test evaluation.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=DEFAULT_T,
        help="Target length for trajectory alignment in Soft-DTW.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="If set, fine-tune the model on STS-B and save it. If not set, try to load the fine-tuned model.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    model_alias = args.model.lower()
    hf_model_name = MODEL_ALIASES[model_alias]

    print(f"Device = {device}")
    print(f"Base model: {hf_model_name}")

    tok, lm, is_decoder_only, config = load_model_and_tokenizer(
        hf_model_name,
        hf_token=args.hf_token,
    )

    # Automatic layer choice based on model alias
    layer_indices = get_layer_indices_for_model(model_alias)
    print(f"Layers to use (auto for {model_alias}): {layer_indices}")

    save_dir = os.path.join(args.results_dir, hf_model_name.replace("/", "_") + "_xp3_ft")
    os.makedirs(save_dir, exist_ok=True)

    if args.finetune:
        ft_layer = layer_indices[0]
        print(f"Fine-tuning on STS-B using layer {ft_layer} ")
        lm = finetune_on_stsb(
            tok,
            lm,
            is_decoder_only,
            max_len=args.max_len,
            layer=ft_layer,
            num_epochs=args.num_epochs,
            train_batch=args.train_batch,
            T=args.T,
            lr=DEFAULT_LR,
            wd=DEFAULT_WD,
        )
        print(f"Saving fine-tuned model to {save_dir}")
        lm.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    else:
        if not os.path.isdir(save_dir):
            print(f"Fine-tuned model directory not found: {save_dir}")
            print("You must run this script with --finetune first to create the fine-tuned model.")
            return
        print(f"Loading fine-tuned model from {save_dir}")
        tok = AutoTokenizer.from_pretrained(save_dir, use_fast=True)
        if is_decoder_only:
            lm = AutoModelForCausalLM.from_pretrained(save_dir, dtype=torch.float32).to(device)
        else:
            lm = AutoModel.from_pretrained(save_dir, dtype=torch.float32).to(device)

        lm = lm.float()
        lm.eval()

    csv_path = make_results_path(args.results_dir, hf_model_name)

    for dataset_name in DEFAULT_DATASETS:
        task_type = DATASET_TASK[dataset_name]

        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {dataset_name} (task={task_type})")
        print(f"{'=' * 60}")

        for layer in layer_indices:
            metric_corrs = evaluate_pipeline_on_dataset(
                dataset_name,
                tok,
                lm,
                is_decoder_only,
                layer,
                args.max_len,
                T=args.T,
                eval_batch=args.eval_batch,
            )

            append_results_csv(
                csv_path,
                hf_model_name,
                layer,
                dataset_name,
                task_type,
                metric_corrs,
            )

            print(f"[Model: {hf_model_name}] layer {layer} dataset {dataset_name} -> {metric_corrs}")

if __name__ == "__main__":
    main()
