"""Pre-flight diagnostic: are the features sensitive to prompt content?

Measures the *prompt-signal ratio* for each feature set:

    prompt_signal_ratio = std(features across prompts at fixed alpha)
                        / std(features across alpha at fixed prompt)

Ratio > 0.5 means prompt content drives as much variation as sparsity level.
Ratio < 0.2 means the features mainly measure sparsity, not content.

Runs on both the existing 6-scalar features and the new CKA features,
so you can see directly whether CKA is more content-sensitive.

Usage
-----
# With synthetic prompts (quick sanity check)
python scripts/audit_feature_variance.py --num-prompts 30

# With real narrative data (the meaningful test)
python scripts/audit_feature_variance.py --data-dir /data/EEG
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.sparse_attention import create_sparse_model
from src.data.tokenizer import TextTokenizer
from src.data.dataset import create_default_prompts
from src.metrics.pathway_metrics import compute_pathway_features
from src.metrics.cka_metrics import extract_cka_features, cka_feature_dim

PATHWAY_NAMES = [
    "routing_sparsity", "path_competition_index", "path_efficiency",
    "routing_entropy", "inter_head_divergence", "layer_stability",
]


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    return logging.getLogger("audit")


@torch.no_grad()
def collect(model, prompts, alpha_values, tokenizer, device, batch_size=32):
    """Collect pathway and CKA features for all (prompt, alpha) combinations.

    Returns:
        pathway_feats: [A, N, 6]
        cka_feats:     [A, N, D_cka]
    """
    A, N = len(alpha_values), len(prompts)
    pathway_all = []
    cka_all = []

    for alpha in alpha_values:
        model.set_sparsity_level(float(alpha))
        batch_path, batch_cka = [], []

        for start in range(0, N, batch_size):
            batch = prompts[start: start + batch_size]
            tokens = tokenizer.tokenize(batch)
            input_ids = tokens["input_ids"].to(device)
            attn_mask = tokens["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                return_attention_maps=True,
                return_hidden_states=True,
            )
            path_feats = compute_pathway_features(out["attention_maps"])  # [B, 6]
            cka_feats = extract_cka_features(out["hidden_states"])        # [B, D]

            batch_path.append(path_feats.cpu().float().numpy())
            batch_cka.append(cka_feats.cpu().float().numpy())

        pathway_all.append(np.concatenate(batch_path, axis=0))   # [N, 6]
        cka_all.append(np.concatenate(batch_cka, axis=0))         # [N, D]

    return np.stack(pathway_all), np.stack(cka_all)


def variance_decomposition(feats: np.ndarray):
    """Decompose total variance into prompt-driven vs alpha-driven components.

    Uses a two-way variance decomposition (no interaction term):
        SS_total  = SS_prompt + SS_alpha + SS_residual
        eta²_prompt = SS_prompt / SS_total
        eta²_alpha  = SS_alpha  / SS_total

    Args:
        feats: [A, N, D]
    Returns:
        eta2_prompt: [D]  fraction of variance explained by prompt identity
        eta2_alpha:  [D]  fraction of variance explained by alpha level
        abs_std_prompt: [D]  absolute std across prompts (raw sensitivity)
    """
    A, N, D = feats.shape
    grand_mean = feats.mean(axis=(0, 1), keepdims=True)         # [1,1,D]
    prompt_means = feats.mean(axis=0, keepdims=True)            # [1,N,D]
    alpha_means = feats.mean(axis=1, keepdims=True)             # [A,1,D]

    ss_total  = ((feats - grand_mean) ** 2).sum(axis=(0, 1))    # [D]
    ss_prompt = A * ((prompt_means - grand_mean) ** 2).sum(axis=(0, 1))
    ss_alpha  = N * ((alpha_means  - grand_mean) ** 2).sum(axis=(0, 1))

    eta2_prompt = ss_prompt / (ss_total + 1e-30)
    eta2_alpha  = ss_alpha  / (ss_total + 1e-30)
    abs_std_prompt = feats.mean(axis=0).std(axis=0)             # std of per-alpha means

    return eta2_prompt, eta2_alpha, abs_std_prompt


def print_report(pathway_feats, cka_feats, alpha_values, logger):
    logger.info("")
    logger.info("=" * 70)
    logger.info("FEATURE VARIANCE AUDIT")
    logger.info("  eta²_prompt = fraction of variance driven by prompt content")
    logger.info("  eta²_alpha  = fraction of variance driven by sparsity level")
    logger.info("  Target: eta²_prompt > 0.3  (features are content-sensitive)")
    logger.info("=" * 70)

    mid = len(alpha_values) // 2

    # ── Pathway statistics ────────────────────────────────────────────────
    p_eta_prompt, p_eta_alpha, _ = variance_decomposition(pathway_feats)
    logger.info(f"\n[6-scalar pathway metrics]")
    logger.info(f"  {'Metric':<28s}  {'eta²_prompt':>12}  {'eta²_alpha':>10}  note")
    logger.info(f"  {'-'*28}  {'-'*12}  {'-'*10}  ----")
    for i, name in enumerate(PATHWAY_NAMES):
        ep, ea = p_eta_prompt[i], p_eta_alpha[i]
        note = "OK" if ep > 0.3 else ("WARN" if ep > 0.1 else "POOR — mostly sparsity")
        logger.info(f"  {name:<28s}  {ep:>12.3f}  {ea:>10.3f}  {note}")

    # ── CKA features ─────────────────────────────────────────────────────
    c_eta_prompt, c_eta_alpha, _ = variance_decomposition(cka_feats)
    D_cka = cka_feats.shape[-1]
    logger.info(f"\n[CKA features]  ({D_cka} dimensions total)")
    logger.info(f"  eta²_prompt — min={c_eta_prompt.min():.3f}  "
                f"median={np.median(c_eta_prompt):.3f}  "
                f"max={c_eta_prompt.max():.3f}")
    logger.info(f"  eta²_alpha  — min={c_eta_alpha.min():.3f}  "
                f"median={np.median(c_eta_alpha):.3f}  "
                f"max={c_eta_alpha.max():.3f}")
    logger.info(f"  % dims with eta²_prompt > 0.3:  "
                f"{(c_eta_prompt > 0.3).mean() * 100:.1f}%")
    logger.info(f"  % dims where prompt > alpha:  "
                f"{(c_eta_prompt > c_eta_alpha).mean() * 100:.1f}%")

    # ── Participation ratio ───────────────────────────────────────────────
    from sklearn.decomposition import PCA
    def part_ratio(X):
        k = min(X.shape[0] - 1, X.shape[1])
        if k < 1:
            return 1.0
        pca = PCA(n_components=k)
        pca.fit(X)
        lam = pca.explained_variance_
        return float(lam.sum() ** 2 / (lam ** 2).sum())

    mid_p = pathway_feats[mid]
    mid_c = cka_feats[mid]
    logger.info(f"\n[Effective dimensionality at alpha={alpha_values[mid]:.2f}]")
    logger.info(f"  Pathway (6-scalar):    participation ratio = {part_ratio(mid_p):.2f} / 6")
    logger.info(f"  CKA ({D_cka}-dim):  participation ratio = {part_ratio(mid_c):.2f} / {D_cka}")

    logger.info("")
    logger.info("Verdict:")
    med_p_ep = float(np.median(p_eta_prompt))
    med_c_ep = float(np.median(c_eta_prompt))
    logger.info(f"  Pathway scalar median eta²_prompt : {med_p_ep:.3f}")
    logger.info(f"  CKA median eta²_prompt            : {med_c_ep:.3f}")
    if med_c_ep > med_p_ep + 0.1:
        logger.info("  >> CKA is substantially more content-sensitive.")
    elif med_c_ep > med_p_ep:
        logger.info("  >> CKA is moderately more content-sensitive.")
    else:
        logger.info("  >> Similar content sensitivity.")
    if max(med_p_ep, med_c_ep) < 0.1:
        logger.warning("  >> BOTH feature sets are dominated by alpha variation. "
                       "Switch to real narrative prompts for meaningful results.")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--alpha-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = create_sparse_model(args.model_name).to(device).eval()
    tokenizer = TextTokenizer(model_name=args.model_name)

    if args.data_dir:
        from src.data.narrative_loader import load_narrative_records
        prompts, _ = load_narrative_records(args.data_dir)
        logger.info(f"Loaded {len(prompts)} narrative prompts from {args.data_dir}")
    else:
        prompts = create_default_prompts(args.num_prompts)
        logger.info(f"Using {len(prompts)} synthetic prompts")

    alpha_values = np.linspace(0.05, 1.0, args.alpha_steps)
    logger.info(f"Collecting features: {len(prompts)} prompts × {len(alpha_values)} alpha values...")

    pathway_feats, cka_feats = collect(
        model, prompts, alpha_values, tokenizer, device, args.batch_size
    )
    logger.info(f"pathway_feats: {pathway_feats.shape}  cka_feats: {cka_feats.shape}")

    print_report(pathway_feats, cka_feats, alpha_values, logger)


if __name__ == "__main__":
    main()
