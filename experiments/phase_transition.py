"""Phase transition analysis for attention sparsity experiments.

Core question
-------------
Does the frozen LLM's computational routing undergo a qualitative change
(phase transition) as the topk-sparsity parameter alpha is swept from
dense (1.0 = keep all weights) to sparse (0.05 = keep top-5%)?

Everything here is inference-only.  No LLM parameters are updated.
The only variables are:
    alpha  — topk percentage fed to SparseAttentionWrapper
    prompts — fixed text inputs (real narrative prompts or synthetic)

The sweep measures how each pathway metric responds to alpha, whether that
response is nonlinear, and (when vividness/condition labels are available)
whether the LLM's circuitry retains task-relevant information as sparsity
is cranked up.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scipy.stats import spearmanr, mannwhitneyu
from scipy.optimize import curve_fit

from src.model.sparse_attention import SparseAttentionWrapper
from src.metrics.pathway_metrics import compute_pathway_features
from src.projection.eeg_projector import EEGProjector
from src.data.tokenizer import TextTokenizer

log = logging.getLogger(__name__)

# ── Metric index → name (must match compute_pathway_features output order) ──

METRIC_NAMES = [
    "routing_sparsity",
    "path_competition_index",
    "path_efficiency",
    "routing_entropy",
    "inter_head_divergence",
    "layer_stability",
]
METRIC_DISPLAY = [
    "Routing Sparsity",
    "Path Competition Index (PCI)",
    "Path Efficiency",
    "Routing Entropy",
    "Inter-head Divergence",
    "Layer Stability",
]
N_METRICS = len(METRIC_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Alpha sweep — collect raw metric data
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_alpha_sweep(
    model: SparseAttentionWrapper,
    prompts: List[str],
    alpha_grid: np.ndarray,
    tokenizer: TextTokenizer,
    eeg_projector: Optional[EEGProjector] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Run frozen LLM over *prompts* at each alpha in *alpha_grid*.

    Args:
        model:         Frozen SparseAttentionWrapper (already on device).
        prompts:       Fixed list of N text strings evaluated at every alpha.
        alpha_grid:    1-D array of A alpha values to sweep (e.g. linspace).
        tokenizer:     TextTokenizer instance.
        eeg_projector: Optional EEGProjector; when provided, EEG stats are
                       also collected.  Must be in eval mode.
        batch_size:    Tokenize/forward this many prompts at a time to fit in
                       memory.  All batches at the same alpha are aggregated.
        device:        Torch device (inferred from model if None).

    Returns:
        Dict with keys:
            ``alpha``          [A]
            ``features_mean``  [A, 6]   — mean over prompts at each alpha
            ``features_std``   [A, 6]
            ``features_all``   [A, N, 6] — every prompt's features (for
                               correlation analysis)
            ``eeg_mean``       [A, C]   — only if eeg_projector given
            ``eeg_std``        [A, C]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if eeg_projector is not None:
        eeg_projector.eval()

    N = len(prompts)
    A = len(alpha_grid)
    features_all = np.zeros((A, N, N_METRICS), dtype=np.float32)
    eeg_all: Optional[np.ndarray] = None

    for ai, alpha in enumerate(alpha_grid):
        model.set_sparsity_level(float(alpha))
        batch_features: List[np.ndarray] = []
        batch_eeg: List[np.ndarray] = []

        for start in range(0, N, batch_size):
            batch_prompts = prompts[start: start + batch_size]
            tokens = tokenizer.tokenize(batch_prompts)
            input_ids = tokens["input_ids"].to(device)
            attn_mask = tokens["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                return_attention_maps=True,
                return_hidden_states=False,
            )
            feats = compute_pathway_features(out["attention_maps"])  # [B, 6]
            batch_features.append(feats.cpu().float().numpy())

            if eeg_projector is not None:
                eeg = eeg_projector(feats)
                batch_eeg.append(eeg.cpu().float().numpy())

        features_all[ai] = np.concatenate(batch_features, axis=0)   # [N, 6]

        if eeg_projector is not None and batch_eeg:
            eeg_concat = np.concatenate(batch_eeg, axis=0)          # [N, C]
            if eeg_all is None:
                eeg_all = np.zeros((A, N, eeg_concat.shape[1]), dtype=np.float32)
            eeg_all[ai] = eeg_concat

        if (ai + 1) % 10 == 0 or ai == A - 1:
            log.info(f"  alpha sweep: {ai+1}/{A} done (alpha={alpha:.3f})")

    result: Dict[str, np.ndarray] = {
        "alpha": np.array(alpha_grid, dtype=np.float32),
        "features_mean": features_all.mean(axis=1),   # [A, 6]
        "features_std": features_all.std(axis=1),     # [A, 6]
        "features_all": features_all,                  # [A, N, 6]
    }
    if eeg_all is not None:
        result["eeg_mean"] = eeg_all.mean(axis=1)     # [A, C]
        result["eeg_std"] = eeg_all.std(axis=1)       # [A, C]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vividness / task correlation sweep
# ─────────────────────────────────────────────────────────────────────────────

def vividness_correlation_sweep(
    features_all: np.ndarray,      # [A, N, 6]
    vividness: np.ndarray,         # [N]
) -> np.ndarray:
    """Spearman r between each metric and vividness at every alpha.

    Returns:
        [A, 6] Spearman r values (NaN when metric is constant).
    """
    A, N, M = features_all.shape
    rs = np.full((A, M), np.nan, dtype=np.float32)
    for ai in range(A):
        for mi in range(M):
            col = features_all[ai, :, mi]
            if col.std() < 1e-8:
                continue
            r, _ = spearmanr(col, vividness)
            rs[ai, mi] = float(r)
    return rs  # [A, 6]


def task_auroc_sweep(
    features_all: np.ndarray,      # [A, N, 6]
    labels: np.ndarray,            # [N] integer task/condition labels
    cv_folds: int = 5,
) -> np.ndarray:
    """Logistic-regression AUROC (cross-validated) at every alpha.

    Falls back gracefully if sklearn is unavailable or only one class present.

    Returns:
        [A] AUROC values (NaN where classification is not possible).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from sklearn.multiclass import OneVsRestClassifier
        _sklearn = True
    except ImportError:
        log.warning("sklearn not available — skipping AUROC sweep")
        return np.full(len(features_all), np.nan, dtype=np.float32)

    A = len(features_all)
    aurocs = np.full(A, np.nan, dtype=np.float32)
    unique_labels = np.unique(labels)
    multi = len(unique_labels) > 2
    n_splits = min(cv_folds, int(np.min(np.bincount(labels))))
    if n_splits < 2:
        log.warning("Too few samples per class for CV — skipping AUROC sweep")
        return aurocs

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for ai in range(A):
        X = features_all[ai]                                # [N, 6]
        fold_scores = []
        for train_idx, test_idx in skf.split(X, labels):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=200, random_state=42)
            clf.fit(X_tr, labels[train_idx])
            if multi:
                proba = clf.predict_proba(X_te)
                try:
                    score = roc_auc_score(
                        labels[test_idx], proba, multi_class="ovr",
                        average="macro", labels=unique_labels,
                    )
                except ValueError:
                    score = np.nan
            else:
                proba = clf.predict_proba(X_te)[:, 1]
                try:
                    score = roc_auc_score(labels[test_idx], proba)
                except ValueError:
                    score = np.nan
            fold_scores.append(score)
        valid = [s for s in fold_scores if not np.isnan(s)]
        if valid:
            aurocs[ai] = float(np.mean(valid))
    return aurocs  # [A]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Phase transition detection
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalise a 1-D array to [0, 1]; returns zeros if range is zero."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def compute_derivatives(
    alpha: np.ndarray,
    features_mean: np.ndarray,  # [A, 6]
) -> Tuple[np.ndarray, np.ndarray]:
    """Numerical first and second derivatives of each metric w.r.t. alpha.

    Uses np.gradient which handles uneven spacing correctly.
    Derivatives are computed on *normalised* metrics so magnitudes are
    comparable across metrics.

    Returns:
        d1: [A, 6]  first derivatives
        d2: [A, 6]  second derivatives
    """
    d1 = np.zeros_like(features_mean)
    d2 = np.zeros_like(features_mean)
    for mi in range(N_METRICS):
        y = _normalise_to_01(features_mean[:, mi])
        d1[:, mi] = np.gradient(y, alpha)
        d2[:, mi] = np.gradient(d1[:, mi], alpha)
    return d1, d2


def detect_phase_transition(
    alpha: np.ndarray,
    features_mean: np.ndarray,  # [A, 6]
) -> Dict:
    """Estimate alpha* using three complementary methods.

    Method 1 — Max-gradient:
        alpha* = argmax |d(PCI)/d(alpha)|

    Method 2 — Inflection point:
        alpha* = argmax |d²(PCI)/d(alpha²)|

    Method 3 — Multi-metric consensus:
        For each alpha, count metrics whose |d1| is in the top-25% of their
        own range.  Alpha* is where the most metrics are simultaneously
        changing rapidly.

    Returns:
        Dict with keys: alpha_star, confidence, method1, method2, method3,
        d1 [A,6], d2 [A,6].
    """
    d1, d2 = compute_derivatives(alpha, features_mean)

    # Method 1: max |d1| for PCI (index 1)
    pci_d1 = d1[:, 1]
    m1_idx = int(np.argmax(np.abs(pci_d1)))
    m1_alpha = float(alpha[m1_idx])

    # Method 2: max |d2| for PCI
    pci_d2 = d2[:, 1]
    m2_idx = int(np.argmax(np.abs(pci_d2)))
    m2_alpha = float(alpha[m2_idx])

    # Method 3: multi-metric consensus
    # score[a] = number of metrics whose |d1[a]| ≥ 75th percentile of |d1[:,m]|
    consensus = np.zeros(len(alpha), dtype=np.float32)
    for mi in range(N_METRICS):
        abs_d1 = np.abs(d1[:, mi])
        threshold = np.percentile(abs_d1, 75)
        consensus += (abs_d1 >= threshold).astype(np.float32)
    m3_idx = int(np.argmax(consensus))
    m3_alpha = float(alpha[m3_idx])

    # Aggregate: median of the three estimates
    estimates = [m1_alpha, m2_alpha, m3_alpha]
    alpha_star = float(np.median(estimates))

    # Confidence: how much do the three methods agree?
    spread = float(np.std(estimates))
    alpha_range = float(alpha.max() - alpha.min())
    confidence = max(0.0, 1.0 - (spread / (alpha_range + 1e-8)))

    return {
        "alpha_star": alpha_star,
        "confidence": confidence,
        "method1": {"alpha": m1_alpha, "metric": "PCI", "idx": m1_idx},
        "method2": {"alpha": m2_alpha, "metric": "PCI_inflection", "idx": m2_idx},
        "method3": {"alpha": m3_alpha, "consensus_score": float(consensus[m3_idx])},
        "consensus": consensus,
        "d1": d1,
        "d2": d2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_tests(
    alpha: np.ndarray,
    features_mean: np.ndarray,   # [A, 6]
    alpha_star: float,
    vividness: Optional[np.ndarray] = None,  # [N]
    features_all: Optional[np.ndarray] = None,  # [A, N, 6]
) -> Dict:
    """Run the four statistical tests described in the spec.

    1. Spearman monotonic correlation (alpha vs metric mean)
    2. Linear vs quadratic R² — nonlinearity test
    3. Pre/post transition Mann-Whitney U test
    4. EEG recoverability (if vividness provided): Spearman at each metric

    Returns a flat dict of scalar results.
    """
    results: Dict = {}

    # ── 1. Monotonic test ──────────────────────────────────────────────────
    for mi, name in enumerate(METRIC_NAMES):
        y = features_mean[:, mi]
        r, p = spearmanr(alpha, y)
        results[f"spearman_r_{name}"] = float(r)
        results[f"spearman_p_{name}"] = float(p)

    # ── 2. Nonlinearity: linear vs quadratic R² ───────────────────────────
    for mi, name in enumerate(METRIC_NAMES):
        y = features_mean[:, mi]
        # Linear fit
        lin_coeffs = np.polyfit(alpha, y, deg=1)
        y_lin = np.polyval(lin_coeffs, alpha)
        ss_res_lin = np.sum((y - y_lin) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / (ss_tot + 1e-12)

        # Quadratic fit
        quad_coeffs = np.polyfit(alpha, y, deg=2)
        y_quad = np.polyval(quad_coeffs, alpha)
        ss_res_quad = np.sum((y - y_quad) ** 2)
        r2_quad = 1 - ss_res_quad / (ss_tot + 1e-12)

        results[f"r2_linear_{name}"] = float(r2_lin)
        results[f"r2_quadratic_{name}"] = float(r2_quad)
        results[f"nonlinearity_{name}"] = float(r2_quad - r2_lin)

    # ── 3. Pre/post transition Mann-Whitney ────────────────────────────────
    pre_mask = alpha < alpha_star
    post_mask = alpha >= alpha_star
    if pre_mask.sum() >= 2 and post_mask.sum() >= 2:
        for mi, name in enumerate(METRIC_NAMES):
            y = features_mean[:, mi]
            stat, p = mannwhitneyu(y[pre_mask], y[post_mask], alternative="two-sided")
            results[f"mannwhitney_stat_{name}"] = float(stat)
            results[f"mannwhitney_p_{name}"] = float(p)

    # ── 4. Vividness correlation sweep (already computed, just summarise) ──
    if vividness is not None and features_all is not None:
        viv_rs = vividness_correlation_sweep(features_all, vividness)  # [A, 6]
        for mi, name in enumerate(METRIC_NAMES):
            valid = viv_rs[:, mi][~np.isnan(viv_rs[:, mi])]
            results[f"vividness_spearman_max_{name}"] = float(np.max(np.abs(valid))) if len(valid) else np.nan
            results[f"vividness_spearman_mean_{name}"] = float(np.mean(valid)) if len(valid) else np.nan

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. W&B logging
# ─────────────────────────────────────────────────────────────────────────────

def _make_metric_curves_figure(
    alpha: np.ndarray,
    features_mean: np.ndarray,   # [A, 6]
    features_std: np.ndarray,    # [A, 6]
    alpha_star: Optional[float] = None,
):
    """2×3 grid of metric-vs-alpha curves with ±std shading."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Pathway Metrics vs Attention Sparsity (alpha)", fontsize=13)
    for mi, (ax, name, display) in enumerate(
        zip(axes.flat, METRIC_NAMES, METRIC_DISPLAY)
    ):
        mean = features_mean[:, mi]
        std = features_std[:, mi]
        ax.plot(alpha, mean, lw=2, color=f"C{mi}")
        ax.fill_between(alpha, mean - std, mean + std, alpha=0.2, color=f"C{mi}")
        if alpha_star is not None:
            ax.axvline(alpha_star, color="red", linestyle="--", lw=1.2, label=f"α*={alpha_star:.2f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("alpha (topk fraction)")
        ax.set_ylabel(display, fontsize=9)
        ax.set_title(display, fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _make_derivative_figure(
    alpha: np.ndarray,
    d1: np.ndarray,  # [A, 6]
    d2: np.ndarray,  # [A, 6]
    alpha_star: Optional[float] = None,
):
    """First and second derivative plots for each metric."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, N_METRICS, figsize=(18, 6))
    fig.suptitle("Derivatives of Normalised Pathway Metrics w.r.t. Alpha", fontsize=12)
    for mi, (name, display) in enumerate(zip(METRIC_NAMES, METRIC_DISPLAY)):
        # First derivative
        ax1 = axes[0, mi]
        ax1.plot(alpha, d1[:, mi], color=f"C{mi}", lw=2)
        ax1.axhline(0, color="gray", lw=0.8)
        if alpha_star is not None:
            ax1.axvline(alpha_star, color="red", linestyle="--", lw=1)
        ax1.set_title(f"d({name[:8]})/dα", fontsize=8)
        ax1.set_xlabel("alpha")
        ax1.grid(alpha=0.3)

        # Second derivative
        ax2 = axes[1, mi]
        ax2.plot(alpha, d2[:, mi], color=f"C{mi}", lw=2, linestyle="--")
        ax2.axhline(0, color="gray", lw=0.8)
        if alpha_star is not None:
            ax2.axvline(alpha_star, color="red", linestyle="--", lw=1)
        ax2.set_title(f"d²({name[:8]})/dα²", fontsize=8)
        ax2.set_xlabel("alpha")
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _make_correlation_heatmap(features_all: np.ndarray, alpha_idx: int):
    """Pearson correlation matrix between metrics at a given alpha index."""
    import matplotlib.pyplot as plt
    X = features_all[alpha_idx]                  # [N, 6]
    # Need multiple samples for correlation
    if X.shape[0] < 3:
        return None
    corr = np.corrcoef(X.T)                      # [6, 6]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(N_METRICS))
    ax.set_yticks(range(N_METRICS))
    short = [n[:10] for n in METRIC_NAMES]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(N_METRICS):
        for j in range(N_METRICS):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Metric Correlation at alpha_idx={alpha_idx}")
    fig.tight_layout()
    return fig


def _make_vividness_figure(
    alpha: np.ndarray,
    viv_rs: np.ndarray,  # [A, 6]
    alpha_star: Optional[float] = None,
):
    """Spearman r (metric vs vividness) as a function of alpha."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, (name, display) in enumerate(zip(METRIC_NAMES, METRIC_DISPLAY)):
        rs = viv_rs[:, mi]
        valid = ~np.isnan(rs)
        ax.plot(alpha[valid], rs[valid], lw=2, label=display[:20], color=f"C{mi}")
    if alpha_star is not None:
        ax.axvline(alpha_star, color="red", linestyle="--", lw=1.5, label=f"α*={alpha_star:.2f}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("alpha (topk fraction)")
    ax.set_ylabel("Spearman r with vividness")
    ax.set_title("Vividness Predictability vs Alpha")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _make_auroc_figure(
    alpha: np.ndarray,
    auroc: np.ndarray,  # [A]
    alpha_star: Optional[float] = None,
):
    """Task classification AUROC as a function of alpha."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    valid = ~np.isnan(auroc)
    ax.plot(alpha[valid], auroc[valid], lw=2, color="steelblue", marker="o", ms=4)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="chance")
    if alpha_star is not None:
        ax.axvline(alpha_star, color="red", linestyle="--", lw=1.5, label=f"α*={alpha_star:.2f}")
    ax.set_xlabel("alpha (topk fraction)")
    ax.set_ylabel("Task AUROC (5-fold CV)")
    ax.set_title("Task Classification Accuracy vs Alpha")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _make_consensus_figure(
    alpha: np.ndarray,
    consensus: np.ndarray,
    alpha_star: float,
):
    """Multi-metric consensus score used for Method 3 detection."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(alpha, consensus, width=(alpha[1] - alpha[0]) * 0.8 if len(alpha) > 1 else 0.05,
           color="steelblue", alpha=0.7)
    ax.axvline(alpha_star, color="red", linestyle="--", lw=1.5, label=f"α*={alpha_star:.2f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("# metrics changing rapidly")
    ax.set_title("Multi-metric Consensus Score (Phase Transition Detection)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def log_to_wandb(
    sweep_data: Dict[str, np.ndarray],
    transition: Dict,
    stats: Dict,
    viv_rs: Optional[np.ndarray] = None,    # [A, 6]
    auroc: Optional[np.ndarray] = None,     # [A]
    run=None,                               # wandb run (or None → auto-init)
    project: str = "eeg-phase-transition",
    run_name: str = "phase_sweep",
):
    """Create all figures and log them plus scalar stats to W&B.

    When *run* is None a new run is initialised.  Pass an existing run to
    log into an already-open experiment.
    """
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed — skipping W&B logging")
        return None

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend

    alpha = sweep_data["alpha"]
    features_mean = sweep_data["features_mean"]
    features_std = sweep_data["features_std"]
    features_all = sweep_data.get("features_all")
    alpha_star = transition["alpha_star"]
    d1, d2 = transition["d1"], transition["d2"]

    if run is None:
        run = wandb.init(project=project, name=run_name)

    # ── Scalar summary ────────────────────────────────────────────────────
    run.log({
        "phase_transition/alpha_star": alpha_star,
        "phase_transition/confidence": transition["confidence"],
        "phase_transition/method1_alpha": transition["method1"]["alpha"],
        "phase_transition/method2_alpha": transition["method2"]["alpha"],
        "phase_transition/method3_alpha": transition["method3"]["alpha"],
    })
    for k, v in stats.items():
        if not np.isnan(float(v) if not isinstance(v, float) else v):
            run.log({f"stats/{k}": v})

    # ── Figure 1: Metric curves ───────────────────────────────────────────
    fig = _make_metric_curves_figure(alpha, features_mean, features_std, alpha_star)
    run.log({"plots/metric_curves": wandb.Image(fig)})
    import matplotlib.pyplot as plt
    plt.close(fig)

    # ── Figure 2: Derivatives ─────────────────────────────────────────────
    fig = _make_derivative_figure(alpha, d1, d2, alpha_star)
    run.log({"plots/derivatives": wandb.Image(fig)})
    plt.close(fig)

    # ── Figure 3: Consensus ───────────────────────────────────────────────
    fig = _make_consensus_figure(alpha, transition["consensus"], alpha_star)
    run.log({"plots/consensus": wandb.Image(fig)})
    plt.close(fig)

    # ── Figure 4: Correlation heatmap at alpha_star ───────────────────────
    if features_all is not None:
        star_idx = int(np.argmin(np.abs(alpha - alpha_star)))
        fig = _make_correlation_heatmap(features_all, star_idx)
        if fig is not None:
            run.log({"plots/correlation_heatmap": wandb.Image(fig)})
            plt.close(fig)

    # ── Figure 5: Vividness Spearman sweep ───────────────────────────────
    if viv_rs is not None:
        fig = _make_vividness_figure(alpha, viv_rs, alpha_star)
        run.log({"plots/vividness_spearman": wandb.Image(fig)})
        plt.close(fig)

    # ── Figure 6: Task AUROC sweep ────────────────────────────────────────
    if auroc is not None:
        fig = _make_auroc_figure(alpha, auroc, alpha_star)
        run.log({"plots/task_auroc": wandb.Image(fig)})
        plt.close(fig)

    return run


# ─────────────────────────────────────────────────────────────────────────────
# 6. Top-level entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_transition_analysis(
    model: SparseAttentionWrapper,
    prompts: List[str],
    tokenizer: TextTokenizer,
    alpha_steps: int = 40,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    eeg_projector: Optional[EEGProjector] = None,
    vividness: Optional[List[float]] = None,
    conditions: Optional[List] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: str = "phase_sweep",
) -> Dict:
    """Run the full phase transition analysis pipeline.

    Args:
        model:           Frozen SparseAttentionWrapper.
        prompts:         Fixed list of prompts (all evaluated at every alpha).
        tokenizer:       TextTokenizer.
        alpha_steps:     Number of alpha values in the grid.
        alpha_min/max:   Grid bounds.
        eeg_projector:   Optional; enables EEG stats collection.
        vividness:       Optional per-prompt vividness ratings [N].
        conditions:      Optional per-prompt task/condition labels [N].
        batch_size:      Batch size for LLM forward passes.
        device:          Torch device (inferred from model if None).
        wandb_project:   W&B project name (None → no W&B logging).
        wandb_run_name:  W&B run name.

    Returns:
        Dict with keys: sweep_data, transition, stats, viv_rs, auroc.
    """
    alpha_grid = np.linspace(alpha_min, alpha_max, alpha_steps)
    log.info(f"Phase transition sweep: {alpha_steps} alpha values [{alpha_min:.2f}, {alpha_max:.2f}] "
             f"over {len(prompts)} prompts")

    # ── Sweep ──────────────────────────────────────────────────────────────
    sweep_data = collect_alpha_sweep(
        model, prompts, alpha_grid, tokenizer,
        eeg_projector=eeg_projector, batch_size=batch_size, device=device,
    )

    # ── Phase transition detection ─────────────────────────────────────────
    transition = detect_phase_transition(sweep_data["alpha"], sweep_data["features_mean"])
    log.info(f"Phase transition detected at alpha_star={transition['alpha_star']:.3f} "
             f"(confidence={transition['confidence']:.2f})")

    # ── Vividness correlation sweep ────────────────────────────────────────
    viv_rs: Optional[np.ndarray] = None
    if vividness is not None:
        viv_arr = np.array(vividness, dtype=np.float32)
        viv_rs = vividness_correlation_sweep(sweep_data["features_all"], viv_arr)
        log.info("Vividness Spearman sweep complete.")

    # ── Task AUROC sweep ───────────────────────────────────────────────────
    auroc: Optional[np.ndarray] = None
    if conditions is not None:
        label_arr = np.array(conditions)
        # Convert string labels to integers if needed
        if label_arr.dtype.kind in ("U", "S", "O"):
            unique = sorted(set(label_arr.tolist()))
            label_map = {v: i for i, v in enumerate(unique)}
            label_arr = np.array([label_map[v] for v in label_arr], dtype=np.int32)
        auroc = task_auroc_sweep(sweep_data["features_all"], label_arr)
        log.info(f"Task AUROC sweep complete. Range: [{np.nanmin(auroc):.3f}, {np.nanmax(auroc):.3f}]")

    # ── Statistical tests ──────────────────────────────────────────────────
    stats = run_statistical_tests(
        sweep_data["alpha"], sweep_data["features_mean"],
        alpha_star=transition["alpha_star"],
        vividness=np.array(vividness) if vividness is not None else None,
        features_all=sweep_data.get("features_all"),
    )

    # ── W&B logging ────────────────────────────────────────────────────────
    wb_run = None
    if wandb_project is not None:
        wb_run = log_to_wandb(
            sweep_data, transition, stats,
            viv_rs=viv_rs, auroc=auroc,
            project=wandb_project, run_name=wandb_run_name,
        )
        if wb_run is not None:
            wb_run.finish()

    return {
        "sweep_data": sweep_data,
        "transition": transition,
        "stats": stats,
        "viv_rs": viv_rs,
        "auroc": auroc,
    }
