"""CLI for the three representation analysis avenues.

Requires a completed alpha sweep (either run fresh or load from cache).

Usage
-----
# Run sweep + all three avenues (synthetic prompts, quick)
python scripts/run_representation_analysis.py --alpha-steps 30 --num-prompts 20

# With real data, reuse existing phase-transition sweep cache
python scripts/run_representation_analysis.py \\
    --data-dir /data/EEG \\
    --sweep-cache ./cache/phase_transition/alpha_sweep.npz \\
    --wandb-project eeg-phase-transition

# Skip contrastive training (faster, no Avenue 1)
python scripts/run_representation_analysis.py \\
    --data-dir /data/EEG \\
    --sweep-cache ./cache/phase_transition/alpha_sweep.npz \\
    --no-contrastive
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import get_default_config
from src.model.sparse_attention import create_sparse_model
from src.data.tokenizer import TextTokenizer
from src.data.dataset import create_default_prompts
from src.projection.eeg_projector import EEGProjector
from experiments.phase_transition import collect_alpha_sweep
from experiments.representation_analysis import (
    run_representation_analysis,
    analyze_circuit_pca,
    analyze_information_retention,
    train_contrastive_encoder,
    evaluate_vividness_probe,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    return logging.getLogger("repr_analysis")


def print_summary(output: dict, logger: logging.Logger):
    logger.info("")
    logger.info("=" * 65)
    logger.info("REPRESENTATION ANALYSIS — SUMMARY")
    logger.info("=" * 65)

    # PCA
    pca = output.get("pca", {})
    evr = pca.get("explained_variance_ratio", [])
    cum = pca.get("cumulative_explained_variance", [])
    if evr:
        pcs_90 = next((i + 1 for i, v in enumerate(cum) if v >= 0.90), len(evr))
        logger.info(f"AVENUE 2 — Circuit PCA")
        logger.info(f"  Participation ratio  : {pca.get('participation_ratio_full', 'N/A'):.2f}")
        logger.info(f"  PCs for 90% variance : {pcs_90}")
        logger.info(f"  Explained variance   : {[f'{v*100:.1f}%' for v in evr]}")
        fa = pca.get("fingerprint_top1_accuracy")
        if fa is not None:
            logger.info(f"  Prompt fingerprint (top-1 accuracy across alpha): {fa:.3f}")
        if "pc_vividness_correlation" in pca:
            logger.info("  PC ↔ vividness Spearman r:")
            for pc_r in pca["pc_vividness_correlation"]:
                flag = "  *" if abs(pc_r["spearman_r"]) > 0.3 else ""
                logger.info(f"    PC{pc_r['pc']}:  r={pc_r['spearman_r']:+.3f}  p={pc_r['p']:.3f}{flag}")
        if "cumulative_r2_vividness" in pca:
            logger.info(f"  Cumulative R² (vividness) w/ 1..6 PCs: "
                        f"{[f'{v:.3f}' for v in pca['cumulative_r2_vividness']]}")

    # Retention
    ret = output.get("retention", {})
    if ret:
        logger.info("")
        logger.info(f"AVENUE 3 — Information Retention (pathway → EEG)")
        d = ret.get("retention_at_dense")
        s = ret.get("retention_at_sparse")
        if d is not None:
            logger.info(f"  Retention at dense alpha (1.0) : {d:.3f}")
            logger.info(f"  Retention at sparse alpha      : {s:.3f}")
        ped = ret.get("pathway_eff_dim", [])
        eed = ret.get("eeg_eff_dim", [])
        if ped and eed:
            logger.info(f"  Pathway eff. dim range  : [{min(ped):.2f}, {max(ped):.2f}]")
            logger.info(f"  EEG eff. dim range      : [{min(eed):.2f}, {max(eed):.2f}]")

    # Contrastive probe
    probe = output.get("probe", {})
    if probe:
        logger.info("")
        logger.info(f"AVENUE 1 — Contrastive Encoder Vividness Probe")
        for k, v in sorted(probe.items()):
            logger.info(f"  {v['n_labeled']:4d} labels: R²={v['r2']:.3f}  "
                        f"Spearman r={v['spearman_r']:.3f}")

    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Representation analysis: PCA, retention, self-supervised"
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--data-dir", default=None,
                        help="Narrative EEG data directory (optional)")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Synthetic prompts when --data-dir not set")
    parser.add_argument("--alpha-steps", type=int, default=30)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sweep-cache", default=None,
                        help="Path to existing alpha_sweep.npz (skips LLM sweep)")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--contrastive-epochs", type=int, default=100)
    parser.add_argument("--no-contrastive", action="store_true",
                        help="Skip contrastive encoder training (Avenue 1)")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default="representation_analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load sweep from cache if available ────────────────────────────────
    sweep_data = None
    if args.sweep_cache and Path(args.sweep_cache).exists():
        logger.info(f"Loading sweep cache from {args.sweep_cache}")
        npz = np.load(args.sweep_cache, allow_pickle=False)
        sweep_data = {k: npz[k] for k in npz.files}
        logger.info(f"  Loaded: alpha shape={sweep_data['alpha'].shape}, "
                    f"features shape={sweep_data['features_all'].shape}")

    # ── Load prompts ──────────────────────────────────────────────────────
    vividness = None
    conditions = None

    if args.data_dir:
        from src.data.narrative_loader import load_narrative_records, _find_trial_info_csvs
        logger.info(f"Loading narrative prompts from {args.data_dir}")
        prompts, vividness_raw = load_narrative_records(args.data_dir)
        vividness = vividness_raw
        logger.info(f"  {len(prompts)} prompts, vividness range "
                    f"[{min(vividness):.2f}, {max(vividness):.2f}]")
        # Attempt to load condition labels
        try:
            import pandas as pd
            csvs = _find_trial_info_csvs(args.data_dir)
            conds = []
            for csv_path in csvs:
                df = pd.read_csv(csv_path)
                if "order" in df.columns:
                    df = df.sort_values("order").reset_index(drop=True)
                if "condition" in df.columns and "vividness_rating" in df.columns:
                    for _, row in df.iterrows():
                        if not pd.isna(row.get("vividness_rating", float("nan"))):
                            conds.append(str(row.get("condition", "unknown")))
            if len(conds) == len(prompts):
                conditions = conds
                logger.info(f"  Conditions: {sorted(set(conditions))}")
        except Exception as e:
            logger.warning(f"  Could not load conditions: {e}")
    else:
        logger.info(f"Using {args.num_prompts} synthetic prompts")
        prompts = create_default_prompts(args.num_prompts)

    # ── Run LLM sweep if not cached ───────────────────────────────────────
    if sweep_data is None:
        logger.info(f"Loading {args.model_name} for alpha sweep...")
        model = create_sparse_model(args.model_name, sparsity_type="topk")
        model.to(device).eval()
        tokenizer = TextTokenizer(model_name=args.model_name)

        alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
        sweep_data = collect_alpha_sweep(
            model, prompts, alpha_grid, tokenizer,
            batch_size=args.batch_size, device=device,
        )
        # Optionally cache it for re-use
        if args.sweep_cache:
            Path(args.sweep_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.sweep_cache, **sweep_data)
            logger.info(f"Sweep cached to {args.sweep_cache}")

        # Free model memory before analysis
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Build EEG projector for Avenue 3 ──────────────────────────────────
    config = get_default_config()
    eeg_projector = EEGProjector(
        input_dim=6,
        output_channels=config.projection.output_channels,
        add_noise=False,
    ).to(device).eval()

    # ── Run all three avenues ──────────────────────────────────────────────
    output = run_representation_analysis(
        sweep_data,
        eeg_projector=eeg_projector,
        vividness=vividness,
        conditions=conditions,
        train_contrastive=not args.no_contrastive,
        contrastive_epochs=args.contrastive_epochs,
        device=device,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    print_summary(output, logger)

    # ── Save figures ──────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from experiments.representation_analysis import (
            _pca_explained_variance_figure,
            _pc_vividness_figure,
            _retention_figure,
            _eff_dim_figure,
            _contrastive_probe_figure,
        )
        pca = output.get("pca", {})
        evr = np.array(pca.get("explained_variance_ratio", []))
        cum = np.array(pca.get("cumulative_explained_variance", []))
        if len(evr):
            fig = _pca_explained_variance_figure(evr, cum)
            fig.savefig(results_dir / "repr_pca_variance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        if "pc_vividness_correlation" in pca:
            fig = _pc_vividness_figure(pca["pc_vividness_correlation"])
            fig.savefig(results_dir / "repr_pc_vividness.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        ret = output.get("retention", {})
        if "pathway_r2_vividness" in ret:
            alpha = np.array(ret["alpha"])
            fig = _retention_figure(
                alpha,
                np.array(ret["pathway_r2_vividness"]),
                np.array(ret["eeg_r2_vividness"]),
                ret["retention_vividness"],
            )
            fig.savefig(results_dir / "repr_retention.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        if "pathway_eff_dim" in ret:
            alpha = np.array(ret["alpha"])
            fig = _eff_dim_figure(
                alpha,
                np.array(ret["pathway_eff_dim"]),
                np.array(ret["eeg_eff_dim"]),
            )
            fig.savefig(results_dir / "repr_eff_dim.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        if "probe" in output and output["probe"]:
            fig = _contrastive_probe_figure(output["probe"])
            fig.savefig(results_dir / "repr_contrastive_probe.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info(f"Figures saved to {results_dir}/")
    except Exception as e:
        logger.warning(f"Could not save figures: {e}")

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary = {
        "pca": {
            k: v for k, v in output.get("pca", {}).items()
            if k not in ("ref_pcs",)           # skip large arrays
        },
        "retention": {
            k: v for k, v in output.get("retention", {}).items()
            if k not in ("alpha",)
        } if output.get("retention") else None,
        "probe": output.get("probe"),
    }
    with open(results_dir / "representation_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if x != x else x)
    logger.info(f"Results saved to {results_dir}/representation_results.json")


if __name__ == "__main__":
    main()
