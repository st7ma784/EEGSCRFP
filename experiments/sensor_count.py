"""Phase 3: Sensor count experiment — patch-based degradation curve.

Sensor model
------------
Each "sensor" is a NetworkPatch: a contiguous window of adjacent transformer
layers sampled with a random subset of attention heads.  The patch pools
attention weights across its (layers × heads) to produce a per-input
**floating score**: how much does this input's routing through the patch
deviate from the population average?

  Low floating score  → fixed routing  (patch treats all inputs the same)
  High floating score → floating path  (patch responds to this input)

At full coverage (M = full_coverage_count) every (layer, head) pair is
sampled by exactly one patch.  We sweep M from 1 to full_coverage and
plot task-prediction AUROC / vividness R² as a function of sensor count.

The elbow of the degradation curve gives the minimum useful sensor count —
the M below which prediction degrades sharply.

Experiment design
-----------------
For GPT-2 (12 layers, 12 heads, depth=3, heads_per_patch=4):
  full_coverage_count = 12 patches
  M values = [1, 2, 4, 8, 12]

For each M:
  - Repeat K=5 times with different random patch subsets
  - Extract [N, M] floating-score feature matrix for all prompts
  - Predict task (logistic regression, AUROC) with 5-fold CV
  - Predict vividness (ridge regression, R²) with 5-fold CV
  - Record mean ± std over repeats

Outputs
-------
  - PNG degradation curves (AUROC and R² vs M)
  - JSON summary of all results
  - W&B run with interactive plots

Usage
-----
# Synthetic prompts (quick sanity check)
python experiments/sensor_count.py --num-prompts 60

# Real narrative data
python experiments/sensor_count.py --data-dir /data/EEG --wandb-project eegscrfp
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.sparse_attention import create_sparse_model
from src.data.tokenizer import TextTokenizer
from src.data.dataset import create_default_prompts
from src.metrics.network_patches import PatchSampler, PatchFeatureExtractor

_log = logging.getLogger("sensor_count")


# ─────────────────────────────────────────────────────────────────────────────
# Feature collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_attention_maps(
    model,
    prompts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 16,
    alpha: float = 1.0,
) -> List[List[torch.Tensor]]:
    """Run all prompts through the model and collect per-layer attention maps.

    Returns:
        attention_maps_list: N-length list; each element is a list of L tensors
            [1, H, S, S] — one per layer.  Sequence dimension S is padded to
            the longest sequence in the batch; here we return single-sample
            lists so S is consistent with that prompt only.
    """
    model.set_sparsity_level(float(alpha))
    all_maps: List[List[torch.Tensor]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start: start + batch_size]
        tokens = tokenizer.tokenize(batch)
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_attention_maps=True,
        )
        attn_maps = out["attention_maps"]   # list of L tensors [B, H, S, S]

        B = input_ids.shape[0]
        for b in range(B):
            # Unpack each sample to [1, H, S, S] per layer
            sample_maps = [layer[b: b + 1] for layer in attn_maps]
            all_maps.append(sample_maps)

    return all_maps


def extract_floating_scores(
    extractor: PatchFeatureExtractor,
    attention_maps_list: List[List[torch.Tensor]],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Apply the extractor to all N samples → [N, M] float32 array."""
    N = len(attention_maps_list)
    M = extractor.M
    all_scores = np.zeros((N, M), dtype=np.float32)

    for start in range(0, N, batch_size):
        batch_maps = attention_maps_list[start: start + batch_size]
        B = len(batch_maps)
        # Stack into [B, H, S, S] per layer — all samples in this batch must
        # have the same sequence length; handle variable S by zero-padding.
        L = len(batch_maps[0])
        S_max = max(m[0].shape[-1] for m in batch_maps)
        H = batch_maps[0][0].shape[1]

        batched: List[torch.Tensor] = []
        for li in range(L):
            layer_stack = []
            for sample in batch_maps:
                t = sample[li]   # [1, H, S_i, S_i]
                S_i = t.shape[-1]
                if S_i < S_max:
                    pad = S_max - S_i
                    t = torch.nn.functional.pad(t, (0, pad, 0, pad))
                layer_stack.append(t)
            batched.append(torch.cat(layer_stack, dim=0).to(device))  # [B,H,S,S]

        with torch.no_grad():
            scores = extractor(batched).cpu().numpy()   # [B, M]
        all_scores[start: start + B] = scores

    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

def _predict_task_auroc(
    X: np.ndarray,
    y: np.ndarray,   # int labels
    n_folds: int = 5,
) -> Tuple[float, float]:
    """Stratified k-fold macro-AUROC for logistic regression.

    Returns (mean_auroc, std_auroc).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan"), 0.0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    aurocs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=0)
        try:
            clf.fit(X_tr, y[train_idx])
            if len(classes) == 2:
                prob = clf.predict_proba(X_te)[:, 1]
                auroc = roc_auc_score(y[test_idx], prob)
            else:
                prob = clf.predict_proba(X_te)
                auroc = roc_auc_score(
                    y[test_idx], prob, multi_class="ovr", average="macro"
                )
            aurocs.append(auroc)
        except Exception:
            pass

    if not aurocs:
        return float("nan"), 0.0
    return float(np.mean(aurocs)), float(np.std(aurocs))


def _predict_vividness_r2(
    X: np.ndarray,
    y: np.ndarray,   # float ratings
    n_folds: int = 5,
) -> Tuple[float, float]:
    """K-fold R² for ridge regression on vividness ratings.

    Returns (mean_r2, std_r2).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    valid = np.isfinite(y)
    if valid.sum() < n_folds * 2:
        return float("nan"), 0.0
    X, y = X[valid], y[valid]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    r2s = []
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        reg = Ridge(alpha=1.0)
        try:
            reg.fit(X_tr, y[train_idx])
            r2 = r2_score(y[test_idx], reg.predict(X_te))
            r2s.append(r2)
        except Exception:
            pass

    if not r2s:
        return float("nan"), 0.0
    return float(np.mean(r2s)), float(np.std(r2s))


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_sensor_count_experiment(
    model,
    prompts: List[str],
    tokenizer,
    device: torch.device,
    task_labels: Optional[np.ndarray] = None,
    vividness_labels: Optional[np.ndarray] = None,
    patch_depth: int = 3,
    heads_per_patch: int = 4,
    n_repeats: int = 5,
    batch_size: int = 16,
    alpha: float = 1.0,
    wandb_run=None,
) -> Dict:
    """Run the full sensor count sweep and return a results dict.

    Args:
        model:           Frozen sparse-attention LLM with set_sparsity_level()
        prompts:         N prompt strings
        tokenizer:       matching tokeniser
        device:          torch device
        task_labels:     [N] int array (None → skip AUROC)
        vividness_labels:[N] float array (None → skip R²)
        patch_depth:     number of adjacent layers per patch
        heads_per_patch: number of attention heads per patch
        n_repeats:       how many random M-patch subsets to average over
        batch_size:      LLM forward-pass batch size
        alpha:           sparsity level (1.0 = dense, run at full fidelity)
        wandb_run:       active wandb run (optional)

    Returns:
        results: dict with keys m_values, auroc_mean, auroc_std,
                 r2_mean, r2_std, full_coverage_count
    """
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    sampler = PatchSampler(
        n_layers=n_layers,
        n_heads=n_heads,
        patch_depth=patch_depth,
        heads_per_patch=heads_per_patch,
    )
    full_coverage = sampler.full_coverage_patches()
    M_full = len(full_coverage)
    m_values = sampler.m_values_for_experiment()

    _log.info(f"Model: {n_layers} layers × {n_heads} heads")
    _log.info(f"Full-coverage patch count: {M_full}")
    _log.info(f"M values to sweep: {m_values}")

    # ── Step 1: collect attention maps at alpha ────────────────────────────
    _log.info(f"Collecting attention maps for {len(prompts)} prompts at alpha={alpha}...")
    attn_maps_list = collect_attention_maps(
        model, prompts, tokenizer, device, batch_size=batch_size, alpha=alpha
    )

    # ── Step 2: calibrate extractor on full coverage ───────────────────────
    _log.info("Calibrating full-coverage extractor (population baseline)...")
    full_extractor = PatchFeatureExtractor(full_coverage)
    full_extractor.calibrate(attn_maps_list)

    # ── Step 3: sweep M ────────────────────────────────────────────────────
    auroc_means, auroc_stds = [], []
    r2_means, r2_stds = [], []

    for M in m_values:
        _log.info(f"M={M} ({n_repeats} repeats)...")
        rep_auroc, rep_r2 = [], []

        for rep in range(n_repeats):
            # Sample M patches from the full-coverage set (without replacement
            # when M <= M_full; otherwise allow overlap via random_patches)
            rng = np.random.default_rng(rep * 1000 + M)
            if M <= M_full:
                chosen = rng.choice(M_full, size=M, replace=False).tolist()
                patches = [full_coverage[i] for i in chosen]
            else:
                patches = sampler.random_patches(M)

            extractor = PatchFeatureExtractor(patches)
            # Share calibration reference from full extractor where possible
            refs = []
            for p in patches:
                # Re-use calibrated reference if patch matches a full-coverage patch
                ref = None
                for fi, fp in enumerate(full_coverage):
                    if (p.layer_indices == fp.layer_indices
                            and p.head_indices == fp.head_indices):
                        if full_extractor._references is not None:
                            ref = full_extractor._references[fi]
                        break
                refs.append(ref)

            # Fill any missing refs by calibrating directly
            missing = [i for i, r in enumerate(refs) if r is None]
            if missing:
                missing_patches = [patches[i] for i in missing]
                tmp = PatchFeatureExtractor(missing_patches)
                tmp.calibrate(attn_maps_list)
                for k, i in enumerate(missing):
                    refs[i] = tmp._references[k]
            extractor._references = refs

            X = extract_floating_scores(extractor, attn_maps_list, device)

            if task_labels is not None:
                a_mean, _ = _predict_task_auroc(X, task_labels)
                rep_auroc.append(a_mean)

            if vividness_labels is not None:
                r_mean, _ = _predict_vividness_r2(X, vividness_labels)
                rep_r2.append(r_mean)

        auroc_means.append(float(np.nanmean(rep_auroc)) if rep_auroc else float("nan"))
        auroc_stds.append(float(np.nanstd(rep_auroc)) if rep_auroc else 0.0)
        r2_means.append(float(np.nanmean(rep_r2)) if rep_r2 else float("nan"))
        r2_stds.append(float(np.nanstd(rep_r2)) if rep_r2 else 0.0)

        _log.info(
            f"  M={M:3d}  AUROC={auroc_means[-1]:.3f}±{auroc_stds[-1]:.3f}"
            f"  R²={r2_means[-1]:.3f}±{r2_stds[-1]:.3f}"
        )

    results = {
        "m_values": m_values,
        "full_coverage_count": M_full,
        "auroc_mean": auroc_means,
        "auroc_std": auroc_stds,
        "r2_mean": r2_means,
        "r2_std": r2_stds,
        "n_prompts": len(prompts),
        "patch_depth": patch_depth,
        "heads_per_patch": heads_per_patch,
        "n_repeats": n_repeats,
        "alpha": alpha,
    }

    if wandb_run is not None:
        _log_to_wandb(wandb_run, results)

    return results


def _log_to_wandb(run, results: Dict) -> None:
    """Log degradation curves and summary metrics to W&B."""
    try:
        import wandb

        m_values = results["m_values"]
        auroc_means = results["auroc_mean"]
        r2_means = results["r2_mean"]

        # Log per-M values as a table (W&B line plot)
        for M, auroc, r2 in zip(m_values, auroc_means, r2_means):
            run.log({
                "sensor_count/n_patches": M,
                "sensor_count/task_auroc": auroc,
                "sensor_count/vividness_r2": r2,
            })

        # Log summary stats
        run.summary.update({
            "sensor_count/full_coverage_count": results["full_coverage_count"],
            "sensor_count/max_auroc": max(
                (v for v in auroc_means if not np.isnan(v)), default=float("nan")
            ),
            "sensor_count/max_r2": max(
                (v for v in r2_means if not np.isnan(v)), default=float("nan")
            ),
        })

        # Log plotted curves as W&B table
        table = wandb.Table(
            columns=["n_patches", "task_auroc", "auroc_std", "vividness_r2", "r2_std"],
            data=[
                [m, a, ae, r, re]
                for m, a, ae, r, re in zip(
                    m_values,
                    auroc_means, results["auroc_std"],
                    r2_means, results["r2_std"],
                )
            ],
        )
        run.log({"sensor_count/degradation_table": table})
        _log.info("Results logged to W&B.")
    except Exception as e:
        _log.warning(f"W&B logging failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_degradation_curves(results: Dict, out_path: Path) -> None:
    """Save a two-panel PNG of the degradation curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        m = np.array(results["m_values"], dtype=float)
        auroc = np.array(results["auroc_mean"])
        auroc_std = np.array(results["auroc_std"])
        r2 = np.array(results["r2_mean"])
        r2_std = np.array(results["r2_std"])
        M_full = results["full_coverage_count"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Sensor count degradation  (depth={results['patch_depth']}, "
            f"heads/patch={results['heads_per_patch']}, "
            f"N={results['n_prompts']} prompts)",
            fontsize=11,
        )

        # AUROC
        ax = axes[0]
        if not np.all(np.isnan(auroc)):
            ax.errorbar(m, auroc, yerr=auroc_std, marker="o", capsize=4,
                        label="Task AUROC")
            ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance (0.5)")
            ax.axvline(M_full, color="C1", ls=":", lw=1.2,
                       label=f"Full coverage (M={M_full})")
            ax.set_xlabel("Number of patches M")
            ax.set_ylabel("Task AUROC (macro)")
            ax.set_ylim(0.4, 1.05)
            ax.legend(fontsize=8)
            ax.set_title("Task prediction")
        else:
            ax.text(0.5, 0.5, "No task labels", ha="center", va="center",
                    transform=ax.transAxes)

        # R²
        ax = axes[1]
        if not np.all(np.isnan(r2)):
            ax.errorbar(m, r2, yerr=r2_std, marker="s", color="C2", capsize=4,
                        label="Vividness R²")
            ax.axhline(0.0, color="gray", ls="--", lw=0.8, label="Baseline R²=0")
            ax.axvline(M_full, color="C1", ls=":", lw=1.2,
                       label=f"Full coverage (M={M_full})")
            ax.set_xlabel("Number of patches M")
            ax.set_ylabel("Vividness R²")
            ax.legend(fontsize=8)
            ax.set_title("Vividness prediction")
        else:
            ax.text(0.5, 0.5, "No vividness labels", ha="center", va="center",
                    transform=ax.transAxes)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        _log.info(f"Saved degradation curve plot to {out_path}")
    except ImportError:
        _log.warning("matplotlib not available — skipping plot")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sensor count experiment: patch-based degradation curve"
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--data-dir",
                        default=os.environ.get("EEGSCRFP_DATA_DIR"),
                        help="Path to narrative EEG data (default: $EEGSCRFP_DATA_DIR)")
    parser.add_argument("--num-prompts", type=int, default=60,
                        help="Synthetic prompts (ignored if --data-dir given)")
    parser.add_argument("--patch-depth", type=int, default=3)
    parser.add_argument("--heads-per-patch", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LLM sparsity level for feature extraction")
    parser.add_argument("--output-dir",
                        default=os.environ.get("EEGSCRFP_OUTPUT_DIR",
                                               "outputs/sensor_count"))
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    model = create_sparse_model(args.model_name).to(device).eval()
    tokenizer = TextTokenizer(model_name=args.model_name)

    # ── Load prompts and labels ────────────────────────────────────────────
    task_labels: Optional[np.ndarray] = None
    vividness_labels: Optional[np.ndarray] = None

    if args.data_dir:
        from src.data.narrative_loader import load_narrative_records
        prompts, raw_vividness = load_narrative_records(args.data_dir)
        _log.info(f"Loaded {len(prompts)} narrative prompts from {args.data_dir}")

        vivid = np.array(raw_vividness, dtype=np.float32)
        if np.isfinite(vivid).sum() >= 10:
            vividness_labels = vivid
            _log.info(
                f"Vividness labels: {np.isfinite(vivid).sum()} valid, "
                f"mean={np.nanmean(vivid):.2f}"
            )

        # Attempt to load condition labels from CSVs directly if pandas available
        try:
            import pandas as pd
            from pathlib import Path as _Path
            csv_files = sorted(_Path(args.data_dir).rglob("*_trialinfo*.csv"))
            cond_list = []
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                if "order" in df.columns:
                    df = df.sort_values("order").reset_index(drop=True)
                if "condition" in df.columns and "prompt_with_condition" in df.columns:
                    for _, row in df.iterrows():
                        p = str(row["prompt_with_condition"]).strip()
                        if p and p.lower() not in ("nan", "none", ""):
                            if np.isfinite(float(row.get("vividness_rating", float("nan")))):
                                cond_list.append(str(row.get("condition", "")))
            if cond_list and len(cond_list) == len(prompts):
                unique = sorted(set(c for c in cond_list if c))
                cmap = {c: i for i, c in enumerate(unique)}
                task_labels = np.array(
                    [cmap.get(c, 0) for c in cond_list], dtype=np.int64
                )
                _log.info(f"Task labels: {len(unique)} classes from condition column")
        except Exception as e:
            _log.debug(f"Could not load condition labels: {e}")
    else:
        prompts = create_default_prompts(args.num_prompts)
        _log.info(f"Using {len(prompts)} synthetic prompts")

    # ── W&B ────────────────────────────────────────────────────────────────
    wandb_run = None
    _in_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))
    if not args.wandb_project and _in_sweep:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "eegscrfp")
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name="sensor_count",
                config=vars(args),
            )
            # Sync sweep-injected config back to args so downstream code uses
            # the sweep's parameter values (wandb.agent overwrites wandb.config).
            for k, v in wandb_run.config.items():
                arg_key = k.replace("-", "_")
                if hasattr(args, arg_key):
                    setattr(args, arg_key, v)
            _log.info(f"W&B config synced: {dict(wandb_run.config)}")
        except Exception as e:
            _log.warning(f"W&B init failed: {e}")

    # ── Run experiment ─────────────────────────────────────────────────────
    results = run_sensor_count_experiment(
        model=model,
        prompts=prompts,
        tokenizer=tokenizer,
        device=device,
        task_labels=task_labels,
        vividness_labels=vividness_labels,
        patch_depth=args.patch_depth,
        heads_per_patch=args.heads_per_patch,
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
        alpha=args.alpha,
        wandb_run=wandb_run,
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    _log.info(f"Results saved to {json_path}")

    plot_degradation_curves(results, out_dir / "degradation_curves.png")

    # ── Print summary ──────────────────────────────────────────────────────
    _log.info("")
    _log.info("=" * 60)
    _log.info("SENSOR COUNT EXPERIMENT — SUMMARY")
    _log.info(f"  Full coverage:  M = {results['full_coverage_count']} patches")
    _log.info(f"  {'M':>4}  {'AUROC':>8}  {'R²':>8}")
    _log.info(f"  {'-'*4}  {'-'*8}  {'-'*8}")
    for m, a, r in zip(
        results["m_values"],
        results["auroc_mean"],
        results["r2_mean"],
    ):
        _log.info(f"  {m:>4}  {a:>8.3f}  {r:>8.3f}")
    _log.info("=" * 60)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
