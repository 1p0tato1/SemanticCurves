import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HTTP_PROXY"] = "http://cache.univ-st-etienne.fr:3128/"
os.environ["HTTPS_PROXY"] = "http://cache.univ-st-etienne.fr:3128/"

# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(model_name: str, hf_token: str = None):
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)

    is_decoder_only = bool(getattr(config, "is_decoder", False)) and not bool(
        getattr(config, "is_encoder_decoder", False)
    )

    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    common_kwargs = {}
    if device == "cuda":
        common_kwargs["torch_dtype"] = torch.float16

    if is_decoder_only:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            device_map="auto" if device == "cuda" else None,
            **common_kwargs,
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            token=hf_token,
            **common_kwargs,
        )
        model.to(device)

    # if tokenizer size changed because of added pad token
    if len(tok) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tok))

    model.eval()
    return tok, model, is_decoder_only, config


# ============================================================
# Token trajectory (raw, layer-specific)
# ============================================================

@torch.no_grad()
def raw_token_trajectory(
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

    H = outputs.hidden_states[layer][0]          # [L, D]
    attn = batch["attention_mask"][0].bool()     # [L]
    input_ids = batch["input_ids"][0]            # [L]

    mask = attn.clone()

    # Remove common special tokens when they exist
    special_ids = set(tok.all_special_ids)
    if len(special_ids) > 0:
        special_mask = torch.zeros_like(mask)
        for sid in special_ids:
            special_mask |= (input_ids == sid)
        mask &= ~special_mask

    X = H[mask]
    X = F.normalize(X, p=2, dim=-1)

    # Fallback: if everything was removed, keep attention-masked tokens
    if X.shape[0] == 0:
        X = H[attn]
        X = F.normalize(X, p=2, dim=-1)

    return X


# ============================================================
# Metrics
# ============================================================

def mean_pool_cosine(X: torch.Tensor, Y: torch.Tensor) -> float:
    x = X.mean(dim=0)
    y = Y.mean(dim=0)
    return F.cosine_similarity(x, y, dim=0).item()


def _resample_unit_interval_linear(x: torch.Tensor, T: int) -> torch.Tensor:
    assert x.dim() == 2, "x must be (L, d)"
    L, d = x.shape
    if L == T:
        return x
    if L == 1:
        return x.repeat(T, 1)

    s = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype)
    pos = s * (L - 1)

    i0 = torch.floor(pos).long().clamp(0, L - 2)
    i1 = i0 + 1
    w = (pos - i0.to(x.dtype)).unsqueeze(-1)

    x0 = x[i0]
    x1 = x[i1]
    return (1.0 - w) * x0 + w * x1


def _aligned_pair(X: torch.Tensor, Y: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return _resample_unit_interval_linear(X, T), _resample_unit_interval_linear(Y, T)


def _trapz_unit(values: torch.Tensor) -> torch.Tensor:
    T = values.shape[-1]
    if T < 2:
        raise ValueError("Need at least 2 points for integration.")
    dt = 1.0 / (T - 1)
    return 0.5 * (values[..., :-1] + values[..., 1:]).sum(dim=-1) * dt


def endpoint_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(X[-1] - Y[-1])


def hausdorff_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    D = torch.cdist(X.float(), Y.float())
    hXY = D.min(dim=1).values.max()
    hYX = D.min(dim=0).values.max()
    return torch.maximum(hXY, hYX)


def chamfer_distance(X: torch.Tensor, Y: torch.Tensor, squared: bool = False) -> torch.Tensor:
    D = torch.cdist(X.float(), Y.float())
    if squared:
        D = D * D
    return D.min(dim=1).values.mean() + D.min(dim=0).values.mean()


def l2_aligned_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    T = min(X.shape[0], Y.shape[0])
    Xr, Yr = _aligned_pair(X, Y, T)
    sq = ((Xr - Yr) ** 2).sum(dim=-1)
    integral = _trapz_unit(sq)
    return torch.sqrt(integral + 1e-12)


def linf_aligned_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    T = min(X.shape[0], Y.shape[0])
    Xr, Yr = _aligned_pair(X, Y, T)
    return torch.linalg.norm(Xr - Yr, dim=-1).max()


def h1_aligned_distance(X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    T = min(X.shape[0], Y.shape[0])
    Xr, Yr = _aligned_pair(X, Y, T)
    Xr = Xr.float()
    Yr = Yr.float()

    if not torch.isfinite(Xr).all() or not torch.isfinite(Yr).all():
        return torch.tensor(float("inf"), device=X.device)

    sq = ((Xr - Yr) ** 2).sum(dim=-1)
    int_pos = _trapz_unit(sq)

    dt = 1.0 / (T - 1)
    VX = (Xr[1:] - Xr[:-1]) / dt
    VY = (Yr[1:] - Yr[:-1]) / dt
    sqv = ((VX - VY) ** 2).sum(dim=-1)
    int_vel = sqv.sum() * dt

    val = int_pos + lam * int_vel
    return torch.sqrt(torch.clamp(val, min=0.0) + 1e-12)


def dtw_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xf = X.float()
    Yf = Y.float()
    a, _ = Xf.shape
    b, _ = Yf.shape
    D = torch.cdist(Xf, Yf)

    dp = torch.full((a + 1, b + 1), float("inf"), device=X.device, dtype=torch.float32)
    dp[0, 0] = 0.0

    for i in range(1, a + 1):
        for j in range(1, b + 1):
            dp[i, j] = D[i - 1, j - 1] + torch.min(
                torch.stack([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]])
            )

    return torch.sqrt(dp[a, b] + 1e-12)


def inv_distance(fn):
    def wrapped(X, Y):
        return 1.0 / (fn(X, Y).item() + 1e-12)
    return wrapped

"""
# no alignment metrics
METRICS = {
    "cos": mean_pool_cosine,
    "inv_endpoint": inv_distance(endpoint_distance),
    "inv_haus": inv_distance(hausdorff_distance),
    "inv_cham": inv_distance(chamfer_distance),
}
"""

METRICS = {
    "cos": mean_pool_cosine,
    "inv_endpoint": inv_distance(endpoint_distance),
    "inv_l2": inv_distance(l2_aligned_distance),
    "inv_linf": inv_distance(linf_aligned_distance),
    "inv_h1": lambda X, Y: 1.0 / (h1_aligned_distance(X, Y, lam=0.5).item() + 1e-12),
    "inv_dtw": inv_distance(dtw_distance),
    "inv_haus": inv_distance(hausdorff_distance),
    "inv_cham": inv_distance(chamfer_distance),
}


# ============================================================
# Dataset loaders
# ============================================================

def normalize_stsb(score: float) -> float:
    return score / 5.0


def normalize_sick(score: float) -> float:
    return (score - 1.0) / 4.0


def normalize_binary(score: float) -> float:
    return float(score)

def load_stsb(split: str = "test") -> List[Dict]:
    ds = load_dataset("sentence-transformers/stsb", split=split)
    return [
        {
            "text1": ex["sentence1"],
            "text2": ex["sentence2"],
            "score": float(ex["score"]),
        }
        for ex in ds
    ]

def split_sickr_fixed(seed: int = 42):
    ds = load_dataset("mteb/sickr-sts", split="test")
    ds = ds.shuffle(seed=seed)

    n = len(ds)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    train_ds = ds.select(range(0, n_train))
    dev_ds = ds.select(range(n_train, n_train + n_dev))
    test_ds = ds.select(range(n_train + n_dev, n))

    return {
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds,
    }

def load_sickr(split: str = "test", seed: int = 42) -> List[Dict]:
    split_map = split_sickr_fixed(seed=seed)
    ds = split_map[split]

    return [
        {
            "text1": ex["sentence1"],
            "text2": ex["sentence2"],
            "score": normalize_sick(float(ex["score"])),
        }
        for ex in ds
    ]

def load_paws(split: str = "test") -> List[Dict]:
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
    return [
        {
            "text1": ex["sentence1"],
            "text2": ex["sentence2"],
            "score": normalize_binary(float(ex["label"])),
        }
        for ex in ds]

DATASET_LOADERS = {
    "stsb": lambda: load_stsb("test"),
    "sick": lambda: load_sickr("test", seed=42),
    "paws": lambda: load_paws("test"),
}

DATASET_TASK = {
    "stsb": "regression",
    "sick": "regression",
    "paws": "binary",
}


# ============================================================
# Evaluation
# ============================================================

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def get_layer_indices(config, user_layers_str: str) -> List[int]:
    """
    If user_layers_str is non-empty (e.g., '-1,-2,-3'), parse that.
    Otherwise, automatically use all transformer layers in negative indexing.
    """
    user_layers_str = user_layers_str.strip()
    if user_layers_str:
        layers = []
        for part in user_layers_str.split(","):
            part = part.strip()
            if part:
                layers.append(int(part))
        return layers

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        # Fallback: last layer only
        return [-1]

    # Example: for 12 layers, this returns [-1, -2, ..., -12]
    return [-(i + 1) for i in range(num_layers)]

# ============================================================
# Helpers
# ============================================================
# Add to core.py

@torch.no_grad()
def get_hidden_states_batch(
        texts: List[str],
        tok,
        lm,
        is_decoder_only: bool,
        max_len: int,
) -> Tuple[List[tuple], List[torch.Tensor], List[torch.Tensor]]:
    """
    Get hidden states for multiple texts in one batch.

    Returns:
        - hidden_states_list: List of hidden state tuples (one per text)
        - attention_masks_list: List of attention masks
        - input_ids_list: List of input IDs
    """
    batch = tok(
        texts,
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

    batch_size = batch["input_ids"].shape[0]
    hidden_states_list = []
    attention_masks_list = []
    input_ids_list = []

    for i in range(batch_size):
        text_hidden_states = tuple(hs[i:i + 1] for hs in outputs.hidden_states)
        hidden_states_list.append(text_hidden_states)
        attention_masks_list.append(batch["attention_mask"][i].bool())
        input_ids_list.append(batch["input_ids"][i])

    return hidden_states_list, attention_masks_list, input_ids_list

@torch.no_grad()
def trajectory_from_hidden_states(
    hidden_states,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tok,
    layer: int,
) -> torch.Tensor:
    """
    Build a normalized trajectory for ONE chosen layer,
    using stored hidden_states instead of running the model again.
    """
    H = hidden_states[layer][0]  # [L, D]
    attn = attention_mask        # [L]
    ids = input_ids              # [L]

    mask = attn.clone()

    special_ids = set(tok.all_special_ids)
    if len(special_ids) > 0:
        special_mask = torch.zeros_like(mask)
        for sid in special_ids:
            special_mask |= (ids == sid)
        mask &= ~special_mask

    X = H[mask]
    X = F.normalize(X, p=2, dim=-1)

    if X.shape[0] == 0:
        X = H[attn]
        X = F.normalize(X, p=2, dim=-1)

    return X