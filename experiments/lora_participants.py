"""LoRA participant generalisation experiment.

Simulates N synthetic 'participants' by injecting rank-r random deltas into
GPT-2 attention projections.  Each participant has a distinct attention
routing pattern, modelling inter-individual variability.

The experiment trains an encoder on a subset of participants and measures
generalisation to held-out participants.  The key question:

    Does cross-participant AUROC drop significantly as LoRA rank increases?

If the drop is small (< 0.10), the encoder has learned participant-invariant
features.  If large, it has over-fitted to idiosyncratic routing in the train
participants.

Metrics
-------
    lora/within_auroc  — trained & tested on train participants (oracle upper bound)
    lora/cross_auroc   — trained on train, tested on held-out participants
    lora/auroc_drop    — within − cross; target < 0.10

Usage
-----
    # Smoke test (no real data needed)
    python experiments/lora_participants.py --rank 4 --n-participants 6

    # W&B sweep
    wandb sweep configs/sweep/lora_participants.yaml
    wandb agent <sweep-id>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_default_prompts
from src.data.tokenizer import TextTokenizer
from src.metrics.network_patches import PatchSampler, PatchFeatureExtractor
from src.encoders.patch_encoder import build_patch_encoder
from experiments.train_encoder import _auroc, _ALPHA_VALUES

_log = logging.getLogger("lora_participants")

# Maps the sweep's lora_target_modules string to the weight key substrings
# that appear in GPT-2's state dict (model.transformer.h.*.attn.c_attn.weight etc.)
_TARGET_MODULE_MAP: Dict[str, List[str]] = {
    "qv":  ["c_attn"],           # Q+K+V combined projection (dominates routing)
    "all": ["c_attn", "c_proj"], # also output projection
}


# ─────────────────────────────────────────────────────────────────────────────
# LoRA delta utilities (pure functions — no model dependency)
# ─────────────────────────────────────────────────────────────────────────────

def lora_delta(
    in_features: int,
    out_features: int,
    rank: int,
    seed: int,
    scale: float = 0.01,
) -> torch.Tensor:
    """Return a [in_features, out_features] rank-r perturbation.

    Uses A @ B decomposition with A drawn from N(0, scale/√rank) and B
    similarly, so the expected Frobenius norm of the delta is independent of
    rank.  This keeps participant diversity comparable across rank settings.

    Args:
        in_features:  First dimension of the weight matrix.
        out_features: Second dimension of the weight matrix.
        rank:         Decomposition rank (must be ≥ 1).
        seed:         Deterministic seed for reproducibility.
        scale:        Controls the magnitude of the perturbation.

    Returns:
        Tensor of shape [in_features, out_features].
    """
    if rank < 1:
        raise ValueError(f"rank must be ≥ 1, got {rank}")
    g = torch.Generator()
    g.manual_seed(seed)
    sigma = scale / math.sqrt(rank)
    A = torch.randn(in_features,  rank, generator=g) * sigma
    B = torch.randn(rank, out_features, generator=g) * sigma
    return A @ B   # [in_features, out_features], matrix rank ≤ r


def participant_state_dict(
    base_state_dict: Dict[str, torch.Tensor],
    rank: int,
    target_module_names: List[str],
    participant_id: int,
    scale: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Return a copy of base_state_dict with rank-r LoRA deltas applied.

    For rank=0 the original dict is returned (no copy, no modification).
    Different participant_ids produce independent random deltas for the same
    weight key, so each participant has its own routing pattern.

    Only 2-D weight tensors whose key contains any of the target_module_names
    are perturbed; biases and other tensors are left unchanged.

    Args:
        base_state_dict:     Base model state dict (e.g. GPT-2).
        rank:                LoRA rank.  0 = no perturbation.
        target_module_names: List of key substrings (e.g. ["c_attn"]).
        participant_id:      Used to derive per-participant seeds.
        scale:               Perturbation magnitude (same as lora_delta).

    Returns:
        Dict with the same keys as base_state_dict.
    """
    if rank == 0:
        return base_state_dict

    sd = {k: v.clone() for k, v in base_state_dict.items()}
    for key, tensor in sd.items():
        if tensor.dim() != 2:
            continue
        if not any(mod in key for mod in target_module_names):
            continue
        in_f, out_f = tensor.shape
        # Combine participant ID and key hash for a unique, deterministic seed
        seed = (participant_id * 104_729 + hash(key)) & 0xFFFF_FFFF
        delta = lora_delta(in_f, out_f, rank, seed, scale)
        sd[key] = tensor + delta

    return sd


# ─────────────────────────────────────────────────────────────────────────────
# Participant split
# ─────────────────────────────────────────────────────────────────────────────

def split_participants(
    n_total: int,
    test_fraction: float,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Split participant IDs 0…n_total−1 into train and test sets.

    Guarantees at least one participant in each set.

    Returns:
        (train_ids, test_ids) — non-overlapping, union = {0…n_total−1}.
    """
    if n_total < 2:
        raise ValueError(f"Need at least 2 participants to split, got {n_total}")
    n_test = max(1, min(n_total - 1, round(n_total * test_fraction)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total).tolist()
    return perm[n_test:], perm[:n_test]


# ─────────────────────────────────────────────────────────────────────────────
# Per-participant feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_features_for_participants(
    base_model,
    tokenizer,
    prompts: List[str],
    participant_ids: List[int],
    rank: int,
    target_module_names: List[str],
    sampler: PatchSampler,
    device: torch.device,
    alpha: float = 1.0,
    batch_size: int = 16,
    scale: float = 0.01,
) -> Dict[int, torch.Tensor]:
    """Extract patch temporal features for each participant.

    Loads the LoRA-patched weights for each participant, runs LLM inference on
    all prompts, and extracts [P, M, S] features from the patch extractor.
    The base model weights are restored after the loop.

    Args:
        base_model:           SparseAttentionWrapper loaded from create_sparse_model().
        tokenizer:            TextTokenizer instance.
        prompts:              List of P prompt strings (same across all participants).
        participant_ids:      Which participant IDs to collect.
        rank:                 LoRA rank (0 = base model, all participants identical).
        target_module_names:  Weight key substrings to perturb.
        sampler:              PatchSampler defining the patch geometry.
        device:               Compute device.
        alpha:                Sparsity level used for feature extraction.
        batch_size:           Number of prompts per LLM forward pass.
        scale:                LoRA perturbation magnitude.

    Returns:
        Dict mapping participant_id → tensor of shape [P, M, S].
    """
    patches = sampler.full_coverage_patches()
    extractor = PatchFeatureExtractor(patches).to(device)

    # Calibrate on the base model
    base_model.set_sparsity_level(alpha)
    cal_maps: List = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        tok = tokenizer.tokenize(batch)
        out = base_model(
            tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
            return_attention_maps=True,
        )
        for b in range(len(batch)):
            cal_maps.append([layer[b: b + 1] for layer in out["attention_maps"]])
    extractor.calibrate(cal_maps)

    base_sd = base_model.model.state_dict()
    features_by_participant: Dict[int, torch.Tensor] = {}

    for pid in participant_ids:
        psd = participant_state_dict(base_sd, rank, target_module_names, pid, scale)
        base_model.model.load_state_dict(psd)
        base_model.set_sparsity_level(alpha)

        all_feats: List[torch.Tensor] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i: i + batch_size]
            tok = tokenizer.tokenize(batch)
            out = base_model(
                tok["input_ids"].to(device),
                attention_mask=tok["attention_mask"].to(device),
                return_attention_maps=True,
            )
            feats = extractor.forward_temporal(out["attention_maps"])  # [B, M, S]
            all_feats.append(feats.cpu())

        features_by_participant[pid] = torch.cat(all_feats, dim=0)  # [P, M, S]
        _log.debug(f"  Participant {pid}: shape {features_by_participant[pid].shape}")

    # Restore base model
    base_model.model.load_state_dict(base_sd)
    return features_by_participant


# ─────────────────────────────────────────────────────────────────────────────
# Encoder for participant experiment (task head only — no text reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

class _TaskEncoder(nn.Module):
    """Patch encoder + task head.  No recon or alpha objectives."""

    def __init__(self, encoder: nn.Module, embed_dim: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.task_head = nn.Linear(embed_dim, max(2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.task_head(self.encoder(x).task_vec)   # [B, n_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    features_by_participant: Dict[int, torch.Tensor],   # pid → [P, M, S]
    train_ids: List[int],
    test_ids: List[int],
    task_labels: torch.Tensor,   # [P] int64 — same prompts across all participants
    encoder_type: str = "transformer",
    embed_dim: int = 256,
    hidden_dim: int = 256,
    n_hidden_layers: int = 2,
    n_attn_heads: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 50,
    device: Optional[torch.device] = None,
    wandb_run=None,
) -> Dict:
    """Train encoder on train participants, evaluate within + cross AUROC.

    Within-AUROC: trained and evaluated on train participants (oracle).
    Cross-AUROC:  trained on train participants, evaluated on held-out test.

    Args:
        features_by_participant: Pre-computed features for every participant.
        train_ids:               Participant IDs used for training.
        test_ids:                Held-out participant IDs.
        task_labels:             Per-prompt condition labels (shared across participants).
        ...                      Encoder + optimisation hyperparameters.
        device:                  Defaults to CPU.
        wandb_run:               Optional W&B run for logging.

    Returns:
        Dict with keys lora/within_auroc, lora/cross_auroc, lora/auroc_drop.
    """
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = torch.device("cpu")

    P = task_labels.shape[0]

    def _stack(ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack features for a list of participant IDs → (X [N,M,S], y [N])."""
        xs = [features_by_participant[pid] for pid in ids]
        X = torch.cat(xs, dim=0)
        y = task_labels.repeat(len(ids))
        return X, y

    X_train, y_train = _stack(train_ids)
    X_test,  y_test  = _stack(test_ids)
    _, M, S = X_train.shape
    n_classes = int(task_labels.max().item() + 1)

    _log.info(
        f"  X_train {X_train.shape}, X_test {X_test.shape}, "
        f"M={M}, S={S}, n_classes={n_classes}"
    )

    # Build encoder
    encoder = build_patch_encoder(
        encoder_type=encoder_type,
        M=M, S=S,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        n_attn_heads=n_attn_heads,
        dropout=dropout,
    )
    model = _TaskEncoder(encoder, embed_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            mask = yb >= 0
            if not mask.any():
                continue
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits[mask], yb[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.eval()

    def _eval_auroc(X: torch.Tensor, y: torch.Tensor) -> float:
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                logits_list.append(model(X[i: i + 64].to(device)).cpu().numpy())
        logits_np = np.concatenate(logits_list)
        from scipy.special import softmax
        probs = softmax(logits_np, axis=-1)
        return _auroc(y.numpy(), probs)

    within_auroc = _eval_auroc(X_train, y_train)
    cross_auroc  = _eval_auroc(X_test,  y_test)
    drop = (
        within_auroc - cross_auroc
        if not (np.isnan(within_auroc) or np.isnan(cross_auroc))
        else float("nan")
    )

    results = {
        "lora/within_auroc":        within_auroc,
        "lora/cross_auroc":         cross_auroc,
        "lora/auroc_drop":          drop,
        "n_train_participants":     len(train_ids),
        "n_test_participants":      len(test_ids),
        "n_prompts":                P,
        "M":                        M,
        "S":                        S,
    }

    if wandb_run is not None:
        wandb_run.log(results)
        wandb_run.summary.update(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Normalise W&B agent's underscore args to hyphens
    sys.argv = [
        ("--" + a[2: a.index("=")].replace("_", "-") + a[a.index("="):])
        if a.startswith("--") and "=" in a
        else ("--" + a[2:].replace("_", "-"))
        if a.startswith("--")
        else a
        for a in sys.argv
    ]

    parser = argparse.ArgumentParser(
        description="LoRA participant generalisation experiment"
    )
    # ── Participants ──────────────────────────────────────────────────────────
    parser.add_argument("--n-participants",  type=int,   default=10)
    parser.add_argument("--test-fraction",   type=float, default=0.25)
    parser.add_argument("--lora-rank",       type=int,   default=4)
    parser.add_argument("--lora-target-modules", default="qv", choices=list(_TARGET_MODULE_MAP))
    parser.add_argument("--lora-scale",      type=float, default=0.01)
    # ── Feature extraction ────────────────────────────────────────────────────
    parser.add_argument("--model-name",      default="gpt2")
    parser.add_argument("--num-prompts",     type=int, default=40)
    parser.add_argument("--data-dir",        default=os.environ.get("EEGSCRFP_DATA_DIR"))
    parser.add_argument("--patch-depth",     type=int, default=1)
    parser.add_argument("--heads-per-patch", type=int, default=1)
    parser.add_argument("--alpha",           type=float, default=1.0)
    # ── Encoder ───────────────────────────────────────────────────────────────
    parser.add_argument("--encoder-type",    default="transformer",
                        choices=["linear", "mlp", "transformer", "eeg_viewer"])
    parser.add_argument("--embed-dim",       type=int,   default=256)
    parser.add_argument("--hidden-dim",      type=int,   default=256)
    parser.add_argument("--n-hidden-layers", type=int,   default=2)
    parser.add_argument("--n-attn-heads",    type=int,   default=4)
    parser.add_argument("--dropout",         type=float, default=0.1)
    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--max-epochs",      type=int,   default=50)
    parser.add_argument("--n-cv-folds",      type=int,   default=5)
    # ── W&B ───────────────────────────────────────────────────────────────────
    parser.add_argument("--wandb-project",   default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info(f"Device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run = None
    _in_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    if args.wandb_project or _in_sweep:
        try:
            import wandb
            _project = args.wandb_project or os.environ.get("WANDB_PROJECT", "eegscrfp")
            wandb_run = wandb.init(project=_project, config=vars(args))
            for k, v in wandb_run.config.items():
                key = k.replace("-", "_")
                if hasattr(args, key):
                    setattr(args, key, v)
        except Exception as e:
            _log.warning(f"W&B init failed: {e}")

    # ── Prompts ───────────────────────────────────────────────────────────────
    prompts = create_default_prompts(args.num_prompts)
    # Synthetic task labels: hash each prompt into one of 4 categories
    task_labels = torch.tensor(
        [hash(p) % 4 for p in prompts], dtype=torch.long
    )
    _log.info(f"Prompts: {len(prompts)}, participants: {args.n_participants}")

    # ── Split participants ────────────────────────────────────────────────────
    train_ids, test_ids = split_participants(
        args.n_participants, args.test_fraction
    )
    _log.info(f"Train participants: {train_ids}")
    _log.info(f"Test  participants: {test_ids}")

    target_keys = _TARGET_MODULE_MAP[args.lora_target_modules]

    # ── Feature extraction ────────────────────────────────────────────────────
    from src.model.sparse_attention import create_sparse_model

    _log.info(f"Loading {args.model_name}...")
    base_model = create_sparse_model(args.model_name).to(device).eval()
    tokenizer = TextTokenizer(model_name=args.model_name)

    n_layers = base_model.model.config.n_layer
    n_heads  = base_model.model.config.n_head
    sampler  = PatchSampler(
        n_layers=n_layers, n_heads=n_heads,
        patch_depth=args.patch_depth, heads_per_patch=args.heads_per_patch,
    )

    all_ids = train_ids + test_ids
    _log.info(f"Extracting features for {len(all_ids)} participants...")
    features = collect_features_for_participants(
        base_model=base_model,
        tokenizer=tokenizer,
        prompts=prompts,
        participant_ids=all_ids,
        rank=args.lora_rank,
        target_module_names=target_keys,
        sampler=sampler,
        device=device,
        alpha=args.alpha,
        batch_size=16,
        scale=args.lora_scale,
    )

    del base_model
    torch.cuda.empty_cache()

    # ── Run experiment ────────────────────────────────────────────────────────
    results = run_experiment(
        features_by_participant=features,
        train_ids=train_ids,
        test_ids=test_ids,
        task_labels=task_labels,
        encoder_type=args.encoder_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        n_attn_heads=args.n_attn_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        device=device,
        wandb_run=wandb_run,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    _log.info("\n" + "=" * 60)
    _log.info("LORA PARTICIPANT GENERALISATION — RESULTS")
    _log.info(f"  LoRA rank              : {args.lora_rank}")
    _log.info(f"  Target modules         : {args.lora_target_modules}")
    _log.info(f"  Train / test            : {len(train_ids)} / {len(test_ids)}")
    _log.info(f"  Within AUROC           : {results['lora/within_auroc']:.3f}  (oracle)")
    _log.info(f"  Cross  AUROC           : {results['lora/cross_auroc']:.3f}  (generalisation)")
    _log.info(f"  AUROC drop             : {results['lora/auroc_drop']:.3f}  (target < 0.10)")
    _log.info("=" * 60)

    out_dir = Path(os.environ.get("EEGSCRFP_OUTPUT_DIR", "outputs/lora"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"rank{args.lora_rank}_{args.lora_target_modules}_p{args.n_participants}"
    with open(out_dir / f"results_{tag}.json", "w") as f:
        json.dump({**results, "args": vars(args)}, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
