"""CLI entry point for the phase transition analysis.

Usage examples
--------------
# Synthetic prompts (quick sanity check)
python scripts/run_phase_transition.py --alpha-steps 30

# Real narrative data with W&B logging
python scripts/run_phase_transition.py \\
    --data-dir /data/EEG \\
    --alpha-steps 40 \\
    --batch-size 64 \\
    --wandb-project eeg-phase-transition

# Full run with cache for the LLM forward passes
python scripts/run_phase_transition.py \\
    --data-dir /data/EEG \\
    --cache-dir ./cache/phase_transition \\
    --alpha-steps 40 \\
    --wandb-project eeg-phase-transition
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import get_default_config
from src.model.sparse_attention import create_sparse_model
from src.data.tokenizer import TextTokenizer
from src.data.dataset import create_default_prompts
from src.projection.eeg_projector import EEGProjector
from experiments.phase_transition import (
    run_phase_transition_analysis,
    METRIC_NAMES,
    METRIC_DISPLAY,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    return logging.getLogger("phase_transition")


def print_summary(results: dict, logger: logging.Logger):
    transition = results["transition"]
    stats = results["stats"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE TRANSITION ANALYSIS — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  alpha_star  : {transition['alpha_star']:.3f}")
    logger.info(f"  confidence  : {transition['confidence']:.2f}")
    logger.info(f"  method 1 (max gradient)  : alpha={transition['method1']['alpha']:.3f}")
    logger.info(f"  method 2 (inflection pt) : alpha={transition['method2']['alpha']:.3f}")
    logger.info(f"  method 3 (consensus)     : alpha={transition['method3']['alpha']:.3f}")
    logger.info("")
    logger.info("Spearman r (alpha vs metric mean):")
    for name in METRIC_NAMES:
        r = stats.get(f"spearman_r_{name}", float("nan"))
        p = stats.get(f"spearman_p_{name}", float("nan"))
        flag = "  *" if abs(r) > 0.5 else ""
        logger.info(f"  {name:<28s}  r={r:+.3f}  p={p:.3f}{flag}")
    logger.info("")
    logger.info("Nonlinearity (quadratic - linear R²):")
    for name in METRIC_NAMES:
        nl = stats.get(f"nonlinearity_{name}", float("nan"))
        flag = "  ** nonlinear" if nl > 0.1 else ""
        logger.info(f"  {name:<28s}  ΔR²={nl:+.3f}{flag}")
    logger.info("")
    logger.info("Mann-Whitney p (pre vs post alpha_star):")
    for name in METRIC_NAMES:
        p = stats.get(f"mannwhitney_p_{name}", float("nan"))
        sig = "  *" if p < 0.05 else ""
        logger.info(f"  {name:<28s}  p={p:.4f}{sig}")

    if results.get("viv_rs") is not None:
        logger.info("")
        logger.info("Max |Spearman r| with vividness across alpha:")
        for mi, name in enumerate(METRIC_NAMES):
            m = stats.get(f"vividness_spearman_max_{name}", float("nan"))
            logger.info(f"  {name:<28s}  |r|_max={m:.3f}")

    if results.get("auroc") is not None:
        auroc = results["auroc"]
        valid = auroc[~np.isnan(auroc)]
        if len(valid):
            logger.info("")
            logger.info(f"Task AUROC — max={valid.max():.3f}  "
                        f"at full density (alpha=1): {auroc[-1]:.3f}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Phase transition analysis for attention sparsity"
    )
    parser.add_argument("--model-name", default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to narrative EEG data directory (optional)")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Number of synthetic prompts (ignored when --data-dir set)")
    parser.add_argument("--alpha-steps", type=int, default=40,
                        help="Number of alpha values to sweep (default: 40)")
    parser.add_argument("--alpha-min", type=float, default=0.05,
                        help="Minimum alpha (default: 0.05)")
    parser.add_argument("--alpha-max", type=float, default=1.0,
                        help="Maximum alpha (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Prompts per LLM forward pass (default: 32)")
    parser.add_argument("--eeg-projector", action="store_true",
                        help="Also collect EEG projection statistics")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory to cache sweep results (avoids re-running LLM)")
    parser.add_argument("--results-dir", default="./results",
                        help="Where to save results JSON and plots (default: ./results)")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (omit to skip logging)")
    parser.add_argument("--wandb-run-name", default="phase_sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logging()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_name} ...")
    model = create_sparse_model(model_name=args.model_name, sparsity_type="topk")
    model.to(device).eval()
    tokenizer = TextTokenizer(model_name=args.model_name)

    # ── Optionally build EEG projector ────────────────────────────────────
    eeg_projector: EEGProjector | None = None
    if args.eeg_projector:
        config = get_default_config()
        eeg_projector = EEGProjector(
            input_dim=6,
            output_channels=config.projection.output_channels,
            add_noise=False,
        )
        eeg_projector.to(device).eval()

    # ── Load prompts and labels ────────────────────────────────────────────
    vividness = None
    conditions = None

    if args.data_dir:
        from src.data.narrative_loader import load_narrative_records
        logger.info(f"Loading narrative prompts from {args.data_dir} ...")
        prompts, vividness_raw = load_narrative_records(args.data_dir)
        vividness = vividness_raw
        logger.info(f"  {len(prompts)} prompts loaded")

        # Try loading condition labels from CSVs
        try:
            import pandas as pd
            from src.data.narrative_loader import _find_trial_info_csvs
            from pathlib import Path as _Path
            csvs = _find_trial_info_csvs(args.data_dir)
            conditions_raw = []
            for csv_path in csvs:
                df = pd.read_csv(csv_path)
                if "order" in df.columns:
                    df = df.sort_values("order").reset_index(drop=True)
                if "condition" in df.columns and "vividness_rating" in df.columns:
                    for _, row in df.iterrows():
                        if not pd.isna(row.get("vividness_rating", float("nan"))):
                            conditions_raw.append(str(row.get("condition", "unknown")))
            if len(conditions_raw) == len(prompts):
                conditions = conditions_raw
                logger.info(f"  Conditions loaded: {sorted(set(conditions))}")
        except Exception as e:
            logger.warning(f"  Could not load conditions: {e}")
    else:
        logger.info(f"Using {args.num_prompts} synthetic prompts")
        prompts = create_default_prompts(args.num_prompts)

    # ── Check / use sweep cache ────────────────────────────────────────────
    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "alpha_sweep.npz"

    # ── Run analysis ───────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load cached sweep data if available
    sweep_data = None
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached sweep from {cache_path}")
        npz = np.load(str(cache_path), allow_pickle=False)
        sweep_data = {k: npz[k] for k in npz.files}

    if sweep_data is None:
        from experiments.phase_transition import collect_alpha_sweep
        alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
        sweep_data = collect_alpha_sweep(
            model, prompts, alpha_grid, tokenizer,
            eeg_projector=eeg_projector,
            batch_size=args.batch_size,
            device=device,
        )
        if cache_path:
            save_kwargs = {k: v for k, v in sweep_data.items()}
            np.savez(str(cache_path), **save_kwargs)
            logger.info(f"Sweep data cached to {cache_path}")

    # Run the rest of the analysis on the (possibly cached) sweep_data
    from experiments.phase_transition import (
        detect_phase_transition,
        vividness_correlation_sweep,
        task_auroc_sweep,
        run_statistical_tests,
        log_to_wandb,
    )

    transition = detect_phase_transition(sweep_data["alpha"], sweep_data["features_mean"])

    viv_rs = None
    if vividness is not None:
        viv_rs = vividness_correlation_sweep(
            sweep_data["features_all"], np.array(vividness, dtype=np.float32)
        )

    auroc = None
    if conditions is not None:
        label_arr = np.array(conditions)
        unique = sorted(set(label_arr.tolist()))
        label_map = {v: i for i, v in enumerate(unique)}
        label_int = np.array([label_map[v] for v in label_arr], dtype=np.int32)
        auroc = task_auroc_sweep(sweep_data["features_all"], label_int)

    stats = run_statistical_tests(
        sweep_data["alpha"], sweep_data["features_mean"],
        alpha_star=transition["alpha_star"],
        vividness=np.array(vividness) if vividness is not None else None,
        features_all=sweep_data.get("features_all"),
    )

    results = {
        "sweep_data": sweep_data,
        "transition": transition,
        "stats": stats,
        "viv_rs": viv_rs,
        "auroc": auroc,
    }

    print_summary(results, logger)

    # ── Save results JSON ──────────────────────────────────────────────────
    serialisable = {
        "alpha_star": float(transition["alpha_star"]),
        "confidence": float(transition["confidence"]),
        "method1_alpha": float(transition["method1"]["alpha"]),
        "method2_alpha": float(transition["method2"]["alpha"]),
        "method3_alpha": float(transition["method3"]["alpha"]),
        "stats": {k: float(v) for k, v in stats.items()
                  if not (isinstance(v, float) and np.isnan(v))},
        "alpha_grid": sweep_data["alpha"].tolist(),
        "features_mean": sweep_data["features_mean"].tolist(),
        "features_std": sweep_data["features_std"].tolist(),
    }
    if auroc is not None:
        serialisable["auroc"] = [float(x) if not np.isnan(x) else None for x in auroc]

    out_path = results_dir / "phase_transition_results.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # ── Save matplotlib figures locally ───────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from experiments.phase_transition import (
            _make_metric_curves_figure,
            _make_derivative_figure,
            _make_consensus_figure,
            _make_vividness_figure,
            _make_auroc_figure,
        )
        figs = {
            "metric_curves": _make_metric_curves_figure(
                sweep_data["alpha"], sweep_data["features_mean"],
                sweep_data["features_std"], transition["alpha_star"]),
            "derivatives": _make_derivative_figure(
                sweep_data["alpha"], transition["d1"], transition["d2"],
                transition["alpha_star"]),
            "consensus": _make_consensus_figure(
                sweep_data["alpha"], transition["consensus"], transition["alpha_star"]),
        }
        if viv_rs is not None:
            figs["vividness"] = _make_vividness_figure(
                sweep_data["alpha"], viv_rs, transition["alpha_star"])
        if auroc is not None:
            figs["task_auroc"] = _make_auroc_figure(
                sweep_data["alpha"], auroc, transition["alpha_star"])

        for name, fig in figs.items():
            path = results_dir / f"phase_{name}.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"  Saved {path}")
    except Exception as e:
        logger.warning(f"Could not save figures: {e}")

    # ── W&B logging ────────────────────────────────────────────────────────
    if args.wandb_project:
        log_to_wandb(
            sweep_data, transition, stats,
            viv_rs=viv_rs, auroc=auroc,
            project=args.wandb_project,
            run_name=args.wandb_run_name,
        )


if __name__ == "__main__":
    main()
