"""Phase 4: Encoder training — three objectives, three axes of variation.

This script trains a learnable encoder on top of the frozen LLM's patch
temporal features ([B, M, S]).  It evaluates three prediction objectives:

    A) Task / semantic prediction  — can the encoder recover condition labels?
       Metric: macro AUROC (target > 0.70)

    B) Prompt reconstruction       — can the encoder recover the LLM embedding?
       Metric: mean cosine similarity (target > 0.60)

    C) Alpha (IS intensity) prediction — can the encoder predict sparsity level?
       Metric: R² (target > 0.80; this is the easy control)

The encoder architecture and training objective weights are swept via W&B
(configs/sweep/encoder_architecture.yaml).

Data pipeline
-------------
LLM inference is deterministic for fixed (prompt, alpha).  Features are
extracted once, cached to disk, and reused across all sweep trials that
share the same (model_name, patch_depth, heads_per_patch).  Training only
runs the small encoder + 3 heads — no LLM gradients.

Feature tensor layout (cache file)
-----------------------------------
  temporal_features : [N, M, S_max]   float32  — from forward_temporal()
  text_embeddings   : [P, d_model]    float32  — LLM mean hidden state per prompt
  task_labels       : [N]             int64    — condition index (-1 = unknown)
  vividness         : [N]             float32  — NaN where unavailable
  alpha_values      : [N]             float32  — known sparsity level
  prompt_indices    : [N]             int64    — which prompt (0..P-1)
  meta              : dict            — M, S_max, d_model, prompt count

Usage
-----
# Synthetic (quick validation)
python experiments/train_encoder.py --encoder-type transformer

# Real data
python experiments/train_encoder.py \\
    --encoder-type transformer --max-epochs 100 \\
    --wandb-project eegscrfp

# W&B sweep agent runs this automatically
wandb agent <sweep-id>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.sparse_attention import create_sparse_model
from src.data.tokenizer import TextTokenizer
from src.data.dataset import create_default_prompts
from src.metrics.network_patches import PatchSampler, PatchFeatureExtractor
from src.encoders.patch_encoder import build_patch_encoder

_log = logging.getLogger("train_encoder")

# Alpha values used to generate the training dataset.
# Each prompt is run at all alpha levels → N_prompts × N_alpha samples.
_ALPHA_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# Feature collection & caching
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_features(
    model,
    tokenizer,
    prompts: List[str],
    task_labels_per_prompt: Optional[np.ndarray],   # [P] int, -1 = unknown
    vividness_per_prompt: Optional[np.ndarray],      # [P] float
    patch_depth: int,
    heads_per_patch: int,
    device: torch.device,
    batch_size: int = 16,
    alpha_values: List[float] = _ALPHA_VALUES,
) -> Dict:
    """Run the frozen LLM and extract [M, S] temporal features for all
    (prompt, alpha) combinations.

    Returns a dict ready for torch.save(), with keys:
        temporal_features  [N, M, S_max]
        text_embeddings    [P, d_model]
        task_labels        [N] int64
        vividness          [N] float32
        alpha_values       [N] float32
        prompt_indices     [N] int64
        meta               dict
    """
    n_layers = model.model.config.n_layer
    n_heads  = model.model.config.n_head
    d_model  = model.model.config.n_embd
    P = len(prompts)

    sampler = PatchSampler(
        n_layers=n_layers, n_heads=n_heads,
        patch_depth=patch_depth, heads_per_patch=heads_per_patch,
    )
    patches  = sampler.full_coverage_patches()
    M        = len(patches)
    extractor = PatchFeatureExtractor(patches).to(device)

    _log.info(f"Collecting features: {P} prompts × {len(alpha_values)} alpha levels "
              f"→ {P * len(alpha_values)} samples")
    _log.info(f"Patch config: depth={patch_depth}, heads={heads_per_patch}, M={M}")

    # ── Step 1: collect per-prompt attention maps at alpha=1.0 for calibration ─
    model.set_sparsity_level(1.0)
    cal_maps: List = []
    for i in range(0, P, batch_size):
        batch = prompts[i: i + batch_size]
        tok = tokenizer.tokenize(batch)
        out = model(
            tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
            return_attention_maps=True,
        )
        attn_maps = out["attention_maps"]
        B = tok["input_ids"].shape[0]
        for b in range(B):
            cal_maps.append([layer[b:b+1] for layer in attn_maps])

    extractor.calibrate(cal_maps)
    _log.info(f"Calibrated extractor: reference shape {extractor._references.shape}")

    # ── Step 2: collect text embeddings (prompt reconstruction target) ─────────
    text_embeds = []
    model.set_sparsity_level(1.0)
    for i in range(0, P, batch_size):
        batch = prompts[i: i + batch_size]
        tok = tokenizer.tokenize(batch)
        out = model(
            tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
            return_hidden_states=True,
        )
        # Mean-pool final hidden state over non-padding tokens → [B, d_model]
        hs = out["hidden_states"][-1]           # [B, S, d_model]
        mask = tok["attention_mask"].to(device).unsqueeze(-1).float()
        emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        text_embeds.append(emb.cpu())
    text_embeddings = torch.cat(text_embeds, dim=0)   # [P, d_model]

    # ── Step 3: extract temporal features for all (prompt, alpha) pairs ────────
    all_temporal: List[torch.Tensor] = []   # each [M, S_i]
    all_alpha:    List[float] = []
    all_pidx:     List[int] = []
    S_max = 0

    for alpha in alpha_values:
        model.set_sparsity_level(alpha)
        for i in range(0, P, batch_size):
            batch = prompts[i: i + batch_size]
            tok = tokenizer.tokenize(batch)
            out = model(
                tok["input_ids"].to(device),
                attention_mask=tok["attention_mask"].to(device),
                return_attention_maps=True,
            )
            feats = extractor.forward_temporal(out["attention_maps"])  # [B, M, S]
            B, _, S = feats.shape
            S_max = max(S_max, S)
            for b in range(B):
                all_temporal.append(feats[b].cpu())   # [M, S]
                all_alpha.append(alpha)
                all_pidx.append(i + b)

    # Pad all samples to S_max
    padded = torch.zeros(len(all_temporal), M, S_max)
    for k, feat in enumerate(all_temporal):
        S_k = feat.shape[1]
        padded[k, :, :S_k] = feat

    N = len(all_temporal)
    # Expand per-prompt labels to per-sample labels
    if task_labels_per_prompt is not None:
        task_labels_n = torch.tensor(
            [task_labels_per_prompt[p] for p in all_pidx], dtype=torch.long
        )
    else:
        task_labels_n = torch.full((N,), -1, dtype=torch.long)

    if vividness_per_prompt is not None:
        vividness_n = torch.tensor(
            [vividness_per_prompt[p] for p in all_pidx], dtype=torch.float32
        )
    else:
        vividness_n = torch.full((N,), float("nan"))

    return {
        "temporal_features":  padded,                               # [N, M, S_max]
        "text_embeddings":    text_embeddings,                      # [P, d_model]
        "task_labels":        task_labels_n,                        # [N]
        "vividness":          vividness_n,                          # [N]
        "alpha_values":       torch.tensor(all_alpha, dtype=torch.float32),  # [N]
        "prompt_indices":     torch.tensor(all_pidx, dtype=torch.long),      # [N]
        "meta": {
            "M": M, "S_max": S_max, "d_model": d_model,
            "n_prompts": P, "n_alpha": len(alpha_values),
            "patch_depth": patch_depth, "heads_per_patch": heads_per_patch,
        },
    }


def get_cache_path(
    cache_dir: str,
    model_name: str,
    patch_depth: int,
    heads_per_patch: int,
    n_prompts: int,
) -> Path:
    tag = f"{model_name}_d{patch_depth}_h{heads_per_patch}_p{n_prompts}"
    return Path(cache_dir) / f"patch_features_{tag}.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Prediction heads
# ─────────────────────────────────────────────────────────────────────────────

class EncoderWithHeads(nn.Module):
    """Encoder + 3 prediction heads.

    Heads:
        task_head:    embed_dim → n_task_classes  (cross-entropy)
        recon_head:   embed_dim → d_model         (1 - cosine_sim)
        alpha_head:   embed_dim → 1               (MSE)
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        n_task_classes: int,
        d_model: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.task_head  = nn.Linear(embed_dim, max(2, n_task_classes))
        self.recon_head = nn.Linear(embed_dim, d_model)
        self.alpha_head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, M, S] patch temporal features.
        Returns:
            dict with keys task_logits, recon_embed, alpha_pred.
        """
        z = self.encoder(x)                      # [B, embed_dim]
        return {
            "task_logits":  self.task_head(z),   # [B, n_classes]
            "recon_embed":  self.recon_head(z),  # [B, d_model]
            "alpha_pred":   self.alpha_head(z).squeeze(-1),  # [B]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    preds: Dict[str, torch.Tensor],
    task_labels:  torch.Tensor,   # [B] long, -1 = ignore
    text_targets: torch.Tensor,   # [B, d_model]
    alpha_targets: torch.Tensor,  # [B] float
    w_task: float,
    w_recon: float,
    w_alpha: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute blended loss and return per-component scalars for logging."""
    losses: Dict[str, float] = {}
    total = torch.zeros(1, device=preds["task_logits"].device)

    # Task classification (ignore samples with label -1)
    task_mask = task_labels >= 0
    if w_task > 0 and task_mask.any():
        L_task = F.cross_entropy(
            preds["task_logits"][task_mask], task_labels[task_mask]
        )
        total = total + w_task * L_task
        losses["loss_task"] = L_task.item()

    # Prompt reconstruction (cosine similarity maximisation)
    if w_recon > 0:
        pred_norm = F.normalize(preds["recon_embed"], dim=-1)
        tgt_norm  = F.normalize(text_targets, dim=-1)
        L_recon = 1.0 - (pred_norm * tgt_norm).sum(-1).mean()
        total = total + w_recon * L_recon
        losses["loss_recon"] = L_recon.item()

    # Alpha prediction
    if w_alpha > 0:
        L_alpha = F.mse_loss(preds["alpha_pred"], alpha_targets)
        total = total + w_alpha * L_alpha
        losses["loss_alpha"] = L_alpha.item()

    losses["loss_total"] = total.item()
    return total, losses


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    classes = np.unique(y_true[y_true >= 0])
    if len(classes) < 2:
        return float("nan")
    mask = y_true >= 0
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true[mask], y_score[mask, 1]))
        return float(roc_auc_score(
            y_true[mask], y_score[mask], multi_class="ovr", average="macro"
        ))
    except Exception:
        return float("nan")


def _cosine_sim(pred: np.ndarray, tgt: np.ndarray) -> float:
    p = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
    t = tgt  / (np.linalg.norm(tgt,  axis=-1, keepdims=True) + 1e-8)
    return float((p * t).sum(-1).mean())


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated training
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder_cv(
    data: Dict,
    encoder_type: str,
    embed_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    n_attn_heads: int,
    dropout: float,
    weight_decay: float,
    lr: float,
    batch_size: int,
    max_epochs: int,
    w_task: float,
    w_recon: float,
    w_alpha: float,
    n_folds: int,
    device: torch.device,
    wandb_run=None,
) -> Dict:
    """K-fold cross-validated encoder training.

    Returns dict with fold-averaged metrics.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    X: torch.Tensor  = data["temporal_features"]    # [N, M, S_max]
    text_embs        = data["text_embeddings"]       # [P, d_model]
    task_labels      = data["task_labels"]           # [N] long
    vividness        = data["vividness"]             # [N] float
    alpha_vals       = data["alpha_values"]          # [N] float
    prompt_idx       = data["prompt_indices"]        # [N] long
    meta             = data["meta"]

    N, M, S = X.shape
    d_model = text_embs.shape[1]
    n_classes = int((task_labels[task_labels >= 0]).max().item() + 1) if (task_labels >= 0).any() else 2

    _log.info(f"Dataset: N={N}, M={M}, S={S}, d_model={d_model}, n_classes={n_classes}")

    # Build per-sample text embeddings (look up by prompt index)
    text_targets_all = text_embs[prompt_idx]   # [N, d_model]

    # Choose CV splitter
    has_task = (task_labels >= 0).any().item()
    if has_task:
        y_cv = task_labels.numpy()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        split_iter = cv.split(np.arange(N), y_cv)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        split_iter = cv.split(np.arange(N))

    fold_metrics: List[Dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(split_iter):
        _log.info(f"  Fold {fold_i + 1}/{n_folds} — train={len(train_idx)}, val={len(val_idx)}")

        # ── Build model ───────────────────────────────────────────────────────
        encoder = build_patch_encoder(
            encoder_type=encoder_type,
            M=M, S=S,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            n_attn_heads=n_attn_heads,
            dropout=dropout,
        )
        model = EncoderWithHeads(
            encoder=encoder,
            embed_dim=embed_dim,
            n_task_classes=n_classes,
            d_model=d_model,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

        # ── DataLoaders ───────────────────────────────────────────────────────
        def _make_loader(idx, shuffle):
            ds = TensorDataset(
                X[idx],
                text_targets_all[idx],
                task_labels[idx],
                alpha_vals[idx],
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        train_loader = _make_loader(train_idx, shuffle=True)
        val_loader   = _make_loader(val_idx,   shuffle=False)

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_auroc = float("nan")
        for epoch in range(max_epochs):
            model.train()
            for xb, tt, tl, av in train_loader:
                xb, tt, tl, av = xb.to(device), tt.to(device), tl.to(device), av.to(device)
                opt.zero_grad()
                preds = model(xb)
                loss, _ = compute_loss(preds, tl, tt, av, w_task, w_recon, w_alpha)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_task_logits, all_recon, all_alpha_pred = [], [], []
        all_task_labels_np, all_text_tgt, all_alpha_tgt = [], [], []

        with torch.no_grad():
            for xb, tt, tl, av in val_loader:
                preds = model(xb.to(device))
                all_task_logits.append(preds["task_logits"].cpu().numpy())
                all_recon.append(preds["recon_embed"].cpu().numpy())
                all_alpha_pred.append(preds["alpha_pred"].cpu().numpy())
                all_task_labels_np.append(tl.numpy())
                all_text_tgt.append(tt.numpy())
                all_alpha_tgt.append(av.numpy())

        task_logits_np = np.concatenate(all_task_logits)
        task_labels_np = np.concatenate(all_task_labels_np)
        recon_np = np.concatenate(all_recon)
        text_tgt_np = np.concatenate(all_text_tgt)
        alpha_pred_np = np.concatenate(all_alpha_pred)
        alpha_tgt_np = np.concatenate(all_alpha_tgt)

        from scipy.special import softmax
        task_prob_np = softmax(task_logits_np, axis=-1)

        fm = {
            "fold": fold_i,
            "encoder/task_auroc":     _auroc(task_labels_np, task_prob_np),
            "encoder/recon_cosine":   _cosine_sim(recon_np, text_tgt_np),
            "encoder/alpha_r2":       _r2(alpha_tgt_np, alpha_pred_np),
        }
        fold_metrics.append(fm)
        _log.info(
            f"    AUROC={fm['encoder/task_auroc']:.3f}  "
            f"cos={fm['encoder/recon_cosine']:.3f}  "
            f"R²={fm['encoder/alpha_r2']:.3f}"
        )

        if wandb_run is not None:
            wandb_run.log({f"{k}_fold{fold_i}": v for k, v in fm.items() if k != "fold"})

    # ── Aggregate across folds ────────────────────────────────────────────────
    def _agg(key):
        vals = [f[key] for f in fold_metrics if not np.isnan(f[key])]
        return float(np.mean(vals)) if vals else float("nan")

    results = {
        "encoder/task_auroc":     _agg("encoder/task_auroc"),
        "encoder/recon_cosine":   _agg("encoder/recon_cosine"),
        "encoder/alpha_r2":       _agg("encoder/alpha_r2"),
        "n_folds": n_folds,
        "n_samples": N,
        "M": M,
        "S": S,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # W&B agent passes args with underscores (--batch_size); argparse expects
    # hyphens (--batch-size).  Normalise before parsing so both styles work.
    sys.argv = [
        ("--" + a[2:a.index("=")].replace("_", "-") + a[a.index("="):])
        if a.startswith("--") and "=" in a
        else ("--" + a[2:].replace("_", "-"))
        if a.startswith("--")
        else a
        for a in sys.argv
    ]

    parser = argparse.ArgumentParser(
        description="Train encoder on LLM patch temporal features"
    )
    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model-name",       default="gpt2")
    parser.add_argument("--data-dir",         default=os.environ.get("EEGSCRFP_DATA_DIR"))
    parser.add_argument("--num-prompts",      type=int, default=60,
                        help="Synthetic prompt count (ignored if --data-dir given)")
    parser.add_argument("--cache-dir",
                        default=os.environ.get("EEGSCRFP_OUTPUT_DIR", "outputs/features"))
    parser.add_argument("--no-cache",         action="store_true",
                        help="Recompute features even if cache exists")
    # ── Patch config ──────────────────────────────────────────────────────────
    parser.add_argument("--patch-depth",      type=int, default=1)
    parser.add_argument("--heads-per-patch",  type=int, default=1)
    parser.add_argument("--alpha",            type=float, default=1.0,
                        help="Alpha for calibration (training uses all _ALPHA_VALUES)")
    # ── Encoder ───────────────────────────────────────────────────────────────
    parser.add_argument("--encoder-type",     default="transformer",
                        choices=["linear", "mlp", "transformer", "eeg_viewer"])
    parser.add_argument("--embed-dim",        type=int,   default=256)
    parser.add_argument("--hidden-dim",       type=int,   default=256)
    parser.add_argument("--n-hidden-layers",  type=int,   default=2)
    parser.add_argument("--n-attn-heads",     type=int,   default=4)
    parser.add_argument("--dropout",          type=float, default=0.1)
    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--max-epochs",       type=int,   default=100)
    parser.add_argument("--n-cv-folds",       type=int,   default=5)
    # ── Objective weights ─────────────────────────────────────────────────────
    parser.add_argument("--objective-task-weight",  type=float, default=1.0)
    parser.add_argument("--objective-recon-weight", type=float, default=0.5)
    parser.add_argument("--objective-alpha-weight", type=float, default=0.1)
    # ── W&B ───────────────────────────────────────────────────────────────────
    parser.add_argument("--wandb-project",    default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info(f"Device: {device}")

    # ── W&B init (before everything so sweep config overrides args) ────────────
    # Also triggers when run by wandb agent (WANDB_SWEEP_ID env var is set).
    wandb_run = None
    _wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT", "eegscrfp")
    _in_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    if args.wandb_project or _in_sweep:
        try:
            import wandb
            wandb_run = wandb.init(
                project=_wandb_project,
                name=f"encoder_{args.encoder_type}",
                config=vars(args),
            )
            # Sync sweep-injected config back to args
            for k, v in wandb_run.config.items():
                arg_key = k.replace("-", "_")
                if hasattr(args, arg_key):
                    setattr(args, arg_key, v)
            _log.info(f"W&B config synced: {dict(wandb_run.config)}")
        except Exception as e:
            _log.warning(f"W&B init failed: {e}")

    # ── Load prompts and labels ────────────────────────────────────────────────
    task_labels_per_prompt: Optional[np.ndarray] = None
    vividness_per_prompt:   Optional[np.ndarray] = None

    if args.data_dir:
        from src.data.narrative_loader import load_narrative_records
        prompts, raw_vividness = load_narrative_records(args.data_dir)
        _log.info(f"Loaded {len(prompts)} narrative prompts")
        vividness_per_prompt = np.array(raw_vividness, dtype=np.float32)

        try:
            import pandas as pd
            csv_files = sorted(Path(args.data_dir).rglob("*_trialinfo*.csv"))
            cond_list = []
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                if "order" in df.columns:
                    df = df.sort_values("order").reset_index(drop=True)
                if "condition" in df.columns:
                    cond_list.extend(df["condition"].tolist())
            if cond_list and len(cond_list) == len(prompts):
                unique = sorted(set(c for c in cond_list if isinstance(c, str)))
                cmap = {c: i for i, c in enumerate(unique)}
                task_labels_per_prompt = np.array(
                    [cmap.get(c, -1) for c in cond_list], dtype=np.int64
                )
                _log.info(f"Task labels: {len(unique)} classes")
        except Exception as e:
            _log.debug(f"Could not load condition labels: {e}")
    else:
        prompts = create_default_prompts(args.num_prompts)
        # For synthetic data: use alpha bucket as proxy task label
        # (actual task labels come from the data collection step)
        _log.info(f"Using {len(prompts)} synthetic prompts")

    # ── Feature cache ─────────────────────────────────────────────────────────
    cache_path = get_cache_path(
        args.cache_dir, args.model_name,
        args.patch_depth, args.heads_per_patch, len(prompts),
    )

    if cache_path.exists() and not args.no_cache:
        _log.info(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        _log.info(f"  M={data['meta']['M']}, S_max={data['meta']['S_max']}, "
                  f"N={data['temporal_features'].shape[0]}")
    else:
        _log.info("Extracting features (no cache found or --no-cache set)...")
        model = create_sparse_model(args.model_name).to(device).eval()
        tokenizer = TextTokenizer(model_name=args.model_name)

        t0 = time.time()
        data = collect_features(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            task_labels_per_prompt=task_labels_per_prompt,
            vividness_per_prompt=vividness_per_prompt,
            patch_depth=args.patch_depth,
            heads_per_patch=args.heads_per_patch,
            device=device,
            batch_size=args.batch_size,
        )
        _log.info(f"Feature extraction: {time.time() - t0:.1f}s")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache_path)
        _log.info(f"Features cached to {cache_path}")

        # Free GPU memory before training
        del model
        torch.cuda.empty_cache()

    # For synthetic prompts with no condition labels, synthesise task labels
    # from alpha buckets so the task head has something to train on.
    if (data["task_labels"] < 0).all():
        n_alpha = len(_ALPHA_VALUES)
        N = data["temporal_features"].shape[0]
        # Assign label = alpha bucket index (repeating pattern over prompts)
        data["task_labels"] = torch.arange(N, dtype=torch.long) % n_alpha
        _log.info("No task labels found — using alpha bucket as synthetic task proxy")

    # ── Train encoder ─────────────────────────────────────────────────────────
    _log.info(f"\nTraining {args.encoder_type} encoder "
              f"(embed_dim={args.embed_dim}, depth={args.n_hidden_layers})...")
    t0 = time.time()

    results = train_encoder_cv(
        data=data,
        encoder_type=args.encoder_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        n_attn_heads=args.n_attn_heads,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        w_task=args.objective_task_weight,
        w_recon=args.objective_recon_weight,
        w_alpha=args.objective_alpha_weight,
        n_folds=args.n_cv_folds,
        device=device,
        wandb_run=wandb_run,
    )
    elapsed = time.time() - t0

    # ── Log results ───────────────────────────────────────────────────────────
    _log.info("\n" + "=" * 60)
    _log.info("ENCODER TRAINING — RESULTS")
    _log.info(f"  Encoder type : {args.encoder_type}")
    _log.info(f"  Elapsed      : {elapsed:.1f}s")
    _log.info(f"  Task AUROC   : {results['encoder/task_auroc']:.3f}  (target >0.70)")
    _log.info(f"  Recon cosine : {results['encoder/recon_cosine']:.3f}  (target >0.60)")
    _log.info(f"  Alpha R²     : {results['encoder/alpha_r2']:.3f}  (target >0.80)")
    _log.info("=" * 60)

    if wandb_run is not None:
        wandb_run.log(results)
        wandb_run.summary.update(results)
        wandb_run.finish()

    # Save JSON summary
    out_dir = Path(os.environ.get("EEGSCRFP_OUTPUT_DIR", "outputs/encoder"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.encoder_type}_d{args.embed_dim}_l{args.n_hidden_layers}"
    with open(out_dir / f"results_{tag}.json", "w") as f:
        json.dump({**results, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
