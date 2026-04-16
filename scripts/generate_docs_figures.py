#!/usr/bin/env python3
"""
Generate EEGSCRFP theory figures for Sphinx documentation.

Run from anywhere:
    python EEGSCRFP/scripts/generate_docs_figures.py

All output goes to docs/_static/eegscrfp/  (created if absent).
No model inference required — figures use illustrative synthetic data
that faithfully represents the mathematical structure of each concept.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
OUT = REPO_ROOT / "docs" / "_static" / "eegscrfp"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
PATCH_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac", "#d4a6c8", "#86bcb6",
]

RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f7f7",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.color": "#cccccc",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#aaaaaa",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


def _apply_style() -> None:
    plt.rcParams.update(RC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _topk_attn(logits: np.ndarray, alpha: float) -> np.ndarray:
    """Apply topk sparsity to a row of attention logits."""
    S = logits.shape[-1]
    k = max(1, int(alpha * S))
    attn = _softmax(logits)
    idx = np.argsort(attn, axis=-1)[..., -k:]
    mask = np.zeros_like(attn)
    np.put_along_axis(mask, idx, 1.0, axis=-1)
    out = attn * mask
    row_sums = out.sum(axis=-1, keepdims=True)
    return out / np.maximum(row_sums, 1e-8)


def _cka_matrix(L: int, alpha: float) -> np.ndarray:
    """
    Simulate a layer-pairwise CKA matrix Φ[i,j] = CKA(layer_i, layer_j).

    Dense routing (alpha≈1) → slow off-diagonal decay (representations stay similar).
    Sparse routing (alpha≈0.1) → fast decay (each layer transforms information).
    """
    gamma = 0.5 + 4.5 * (1.0 - alpha)   # sparse ↔ higher gamma ↔ faster decay
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    base = np.exp(-gamma * np.abs(i - j) / L)
    # Small random perturbation to look realistic
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((L, L)) * 0.03
    mat = np.clip(base + noise, 0.0, 1.0)
    np.fill_diagonal(mat, 1.0)
    return mat


def _sigmoid(x: np.ndarray, k: float = 1.0, x0: float = 0.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_ROOT)}")


# ===========================================================================
# Figure 1 — Unfolded Pathway Conceptual Diagram
# ===========================================================================

def fig1_unfolded_pathway() -> None:
    _apply_style()
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 0.6, 3], wspace=0.05)
    ax_bio = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1])
    ax_tr = fig.add_subplot(gs[2])

    for ax in (ax_bio, ax_mid, ax_tr):
        ax.set_facecolor("white")
        ax.axis("off")

    # ── Left: biological recurrent circuit ──────────────────────────────────
    ax_bio.set_xlim(0, 4)
    ax_bio.set_ylim(-0.5, 5.5)
    ax_bio.set_title("Biological Cortex", pad=14, fontsize=13, fontweight="bold")

    node_x = 2.0
    node_ys = [4.2, 3.0, 1.8, 0.6]
    node_labels = ["Output\nneurons", "Layer 3", "Layer 2", "Layer 1\n(input)"]
    node_col = "#4e79a7"

    for i, (y, lbl) in enumerate(zip(node_ys, node_labels)):
        circ = plt.Circle((node_x, y), 0.38, fc=node_col, ec="#2c5282",
                           lw=2, zorder=5)
        ax_bio.add_patch(circ)
        ax_bio.text(node_x + 0.65, y, lbl, va="center", ha="left",
                    fontsize=9, color="#333")
        if i < len(node_ys) - 1:
            ax_bio.annotate(
                "", xy=(node_x, node_ys[i + 1] + 0.4),
                xytext=(node_x, y - 0.4),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5),
            )

    # Recurrent feedback arrow (curved)
    ax_bio.annotate(
        "", xy=(node_x, node_ys[0] + 0.42),
        xytext=(node_x, node_ys[-1] - 0.42),
        arrowprops=dict(
            arrowstyle="->", color="#e15759", lw=2.5,
            connectionstyle="arc3,rad=-0.65",
        ),
    )
    ax_bio.text(0.25, 2.4, "recurrent\nfeedback", fontsize=9, color="#e15759",
                ha="center", style="italic", rotation=0)

    ax_bio.text(node_x, -0.3, "time is continuous & folded",
                ha="center", fontsize=9, color="#888", style="italic")

    # ── Centre: equivalence ──────────────────────────────────────────────────
    ax_mid.text(0.5, 0.55, "≅", fontsize=42, ha="center", va="center",
                transform=ax_mid.transAxes, color="#555")
    ax_mid.text(0.5, 0.38, "unfold\nrecurrence\ninto depth", fontsize=9,
                ha="center", va="center", transform=ax_mid.transAxes,
                color="#888", style="italic")

    # ── Right: transformer unfolded grid ────────────────────────────────────
    ax_tr.set_xlim(-0.3, 5.5)
    ax_tr.set_ylim(-0.7, 5.2)
    ax_tr.set_title("Transformer = Unfolded Pathway", pad=14,
                    fontsize=13, fontweight="bold")

    L, H = 4, 4          # simplified (real: 12 × 12)
    cell_w, cell_h = 0.9, 0.7
    x0, y0 = 0.8, 0.4    # grid origin (bottom-left)

    # Patch definitions: (layer_start, layer_end, head_start, head_end, colour)
    patches_def = [
        (0, 1, 0, 1, PATCH_PALETTE[0]),   # P0
        (1, 2, 1, 3, PATCH_PALETTE[1]),   # P1
        (2, 3, 0, 2, PATCH_PALETTE[2]),   # P2
    ]

    # Draw grid cells
    for l_idx in range(L):
        for h_idx in range(H):
            rx = x0 + h_idx * cell_w
            ry = y0 + l_idx * cell_h
            fc = "#dce8f5"
            rect = FancyBboxPatch(
                (rx, ry), cell_w * 0.88, cell_h * 0.82,
                boxstyle="round,pad=0.04", fc=fc, ec="#aec6e0", lw=0.8, zorder=2,
            )
            ax_tr.add_patch(rect)
            if h_idx == 0:
                ax_tr.text(-0.1, ry + cell_h * 0.4,
                           f"ℓ={l_idx}", fontsize=8, ha="right", va="center",
                           color="#555")
        if l_idx == 0:
            for h_idx in range(H):
                ax_tr.text(
                    x0 + h_idx * cell_w + cell_w * 0.44, -0.3,
                    f"h={h_idx}", fontsize=8, ha="center", va="top", color="#555",
                )

    # Highlight patches with coloured overlays
    for (l0, l1, h0, h1, col) in patches_def:
        px = x0 + h0 * cell_w - 0.04
        py = y0 + l0 * cell_h - 0.04
        pw = (h1 - h0 + 1) * cell_w - 0.04
        ph = (l1 - l0 + 1) * cell_h - 0.04
        rect = FancyBboxPatch(
            (px, py), pw, ph,
            boxstyle="round,pad=0.06",
            fc=col, ec=col, lw=2, alpha=0.35, zorder=3,
        )
        ax_tr.add_patch(rect)

    # Residual stream arrow on the left
    ax_tr.annotate(
        "", xy=(x0 - 0.18, y0 + L * cell_h - 0.1),
        xytext=(x0 - 0.18, y0 + 0.1),
        arrowprops=dict(arrowstyle="->", color="#4e79a7", lw=2.2),
    )
    ax_tr.text(x0 - 0.28, y0 + L * cell_h / 2,
               "residual\nstream\n(depth →)", fontsize=8, ha="right",
               va="center", color="#4e79a7", style="italic")

    # sparsity label
    ax_tr.text(x0 + H * cell_w / 2, y0 + L * cell_h + 0.25,
               r"topk$_\alpha$ masks each head's routing",
               ha="center", fontsize=9, color="#e15759", style="italic")

    # Patch legend
    handles = [
        mpatches.Patch(fc=PATCH_PALETTE[i], alpha=0.6,
                       label=f"Patch $P_{i}$ → virtual EEG sensor")
        for i in range(3)
    ]
    ax_tr.legend(handles=handles, loc="lower right", fontsize=8,
                 framealpha=0.85)

    ax_tr.text(
        x0 + H * cell_w / 2, -0.55,
        r"depth $\ell \in [0, L]$ replaces continuous time",
        ha="center", fontsize=9, color="#888", style="italic",
    )

    _save(fig, "fig1_unfolded_pathway.png")


# ===========================================================================
# Figure 2 — Alpha as Neuromodulation (attention heatmaps)
# ===========================================================================

def fig2_alpha_neuromodulation() -> None:
    _apply_style()
    rng = np.random.default_rng(7)

    alphas = [1.0, 0.5, 0.15]
    alpha_labels = [r"$\alpha = 1.0$  (dense)", r"$\alpha = 0.5$  (selective)",
                    r"$\alpha = 0.15$  (sparse)"]
    vivid_labels = ["Vivid inner speech", "Partial engagement", "Degraded / suppressed"]
    S = 10  # sequence length
    n_layers_show = 3

    fig, axes = plt.subplots(n_layers_show, len(alphas), figsize=(13, 7))
    fig.suptitle(
        r"Topk Sparsity $\alpha$ as Neuromodulatory Gating",
        fontsize=14, fontweight="bold", y=1.01,
    )

    cmap = plt.cm.YlOrRd

    for col, (alpha, alabel, vlabel) in enumerate(
        zip(alphas, alpha_labels, vivid_labels)
    ):
        for row in range(n_layers_show):
            ax = axes[row, col]
            logits = rng.standard_normal((S, S)) * 1.5
            attn = _topk_attn(logits, alpha)
            im = ax.imshow(attn, vmin=0, vmax=1, cmap=cmap, aspect="auto",
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if row == 0:
                ax.set_title(f"{alabel}\n{vlabel}", fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(f"Layer {row + 1}", fontsize=10)

    # Shared colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, cax=cbar_ax, label="Attention weight")

    # Active-routes annotation
    for col, alpha in enumerate(alphas):
        k = max(1, int(alpha * S))
        axes[-1, col].set_xlabel(
            f"{k}/{S} active routes per query", fontsize=9, color="#555"
        )

    fig.text(0.5, -0.01,
             r"Each cell $\tilde{A}^{(\ell,h)}_{ij}$: probability that token $i$ "
             r"attends to token $j$ after topk masking",
             ha="center", fontsize=10, color="#555")

    _save(fig, "fig2_alpha_neuromodulation.png")


# ===========================================================================
# Figure 3 — Network Patch ↔ EEG Sensor Mapping
# ===========================================================================

def fig3_patch_to_sensor() -> None:
    _apply_style()
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 2, wspace=0.06)
    ax_grid = fig.add_subplot(gs[0])
    ax_eeg = fig.add_subplot(gs[1])

    for ax in (ax_grid, ax_eeg):
        ax.set_facecolor("white")
        ax.axis("off")

    # ── Left: transformer (layer × head) grid with patches ─────────────────
    ax_grid.set_xlim(-0.5, 12.5)
    ax_grid.set_ylim(-1.0, 13.5)
    ax_grid.set_title("Transformer: layers × heads", fontsize=13,
                       fontweight="bold", pad=8)

    L, H = 12, 12  # GPT-2 actual dimensions
    cw, ch = 1.0, 1.0

    # 12 patches: 4 layer windows × 3 head groups
    # window_size=3, head_group_size=4
    patch_id = 0
    patch_centres = {}
    for lw in range(4):
        l0 = lw * 3
        for hg in range(3):
            h0 = hg * 4
            col = PATCH_PALETTE[patch_id % 12]
            rx, ry = h0 * cw, l0 * ch
            rw, rh = 4 * cw, 3 * ch
            rect = FancyBboxPatch(
                (rx, ry), rw - 0.12, rh - 0.12,
                boxstyle="round,pad=0.1",
                fc=col, ec=col, lw=1.8, alpha=0.45, zorder=3,
            )
            ax_grid.add_patch(rect)
            ax_grid.text(
                rx + rw / 2, ry + rh / 2,
                f"P{patch_id}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333", zorder=4,
            )
            patch_centres[patch_id] = (
                rx + rw / 2, ry + rh / 2,
            )
            patch_id += 1

    # Grid lines
    for l in range(L + 1):
        ax_grid.axhline(l * ch, color="#ccc", lw=0.4, zorder=1)
    for h in range(H + 1):
        ax_grid.axvline(h * cw, color="#ccc", lw=0.4, zorder=1)

    # Axis labels
    ax_grid.text(6, -0.75, "← Head index h →", ha="center", fontsize=10, color="#555")
    ax_grid.text(-0.45, 6, "← Layer ℓ →", ha="center", va="center",
                 fontsize=10, color="#555", rotation=90)

    # ── Right: EEG head map ─────────────────────────────────────────────────
    ax_eeg.set_xlim(-1.4, 1.4)
    ax_eeg.set_ylim(-1.4, 1.5)
    ax_eeg.set_title("Virtual EEG sensors (network patches)", fontsize=13,
                      fontweight="bold", pad=8)

    # Head outline
    head = plt.Circle((0, 0), 1.0, fc="#f0f4f8", ec="#aaa", lw=2, zorder=1)
    ax_eeg.add_patch(head)
    # Nose
    ax_eeg.plot([0], [1.08], "^", ms=10, color="#aaa", zorder=2)
    # Ears
    ax_eeg.add_patch(plt.Circle((-1.02, 0), 0.09, fc="#f0f4f8", ec="#aaa", lw=1.5))
    ax_eeg.add_patch(plt.Circle((1.02, 0), 0.09, fc="#f0f4f8", ec="#aaa", lw=1.5))

    # 12 electrode positions arranged over the scalp
    thetas = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12
    radii = [0.28, 0.55, 0.28, 0.72, 0.55, 0.82,
              0.28, 0.72, 0.55, 0.28, 0.55, 0.82]
    elec_positions = [
        (r * np.sin(t), r * np.cos(t))
        for r, t in zip(radii, thetas)
    ]

    for idx, (ex, ey) in enumerate(elec_positions):
        col = PATCH_PALETTE[idx % 12]
        circ = plt.Circle((ex, ey), 0.085, fc=col, ec="white",
                           lw=2, zorder=5, alpha=0.9)
        ax_eeg.add_patch(circ)
        ax_eeg.text(ex, ey, f"P{idx}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)

    ax_eeg.text(0, -1.3, "Each dot ↔ one network patch\n"
                "position encodes layer depth & head group",
                ha="center", fontsize=9, color="#555", style="italic")

    _save(fig, "fig3_patch_to_sensor.png")


# ===========================================================================
# Figure 4 — CKA Information Flow Matrices
# ===========================================================================

def fig4_cka_flow() -> None:
    _apply_style()
    L = 13  # 12 layers + input embedding at position 0
    alphas = [1.0, 0.5, 0.15]
    titles = [
        r"$\alpha = 1.0$  — dense routing",
        r"$\alpha = 0.5$  — selective routing",
        r"$\alpha = 0.15$  — sparse routing",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        r"CKA Information-Flow Matrix  $\Phi_{\ell\ell'} = \mathrm{CKA}(X^\ell, X^{\ell'})$",
        fontsize=13, fontweight="bold", y=1.03,
    )
    cmap = "plasma"

    for ax, alpha, title in zip(axes, alphas, titles):
        mat = _cka_matrix(L, alpha)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=11, pad=6)
        ticks = [0, 3, 6, 9, 12]
        tick_labels = ["emb", "ℓ3", "ℓ6", "ℓ9", "ℓ12"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Layer ℓ′", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Layer ℓ", fontsize=10)

    plt.colorbar(im, ax=axes, shrink=0.7, label="CKA ∈ [0, 1]",
                 orientation="vertical", pad=0.02)

    fig.text(
        0.5, -0.06,
        "High off-diagonal CKA = representations preserved across depth  |  "
        "Sparse α collapses off-diagonal → information lost between layers",
        ha="center", fontsize=10, color="#555",
    )
    _save(fig, "fig4_cka_flow.png")


# ===========================================================================
# Figure 5 — Virtual EEG Traces (floating scores)
# ===========================================================================

def fig5_virtual_eeg() -> None:
    _apply_style()
    rng = np.random.default_rng(42)

    S = 35          # tokens (→ "time")
    M = 6           # patches shown (subset of 12)
    t = np.arange(S)
    onset = 8       # stimulus onset token

    # Simulate floating scores: each patch has a characteristic response profile
    depths = np.linspace(0, 1, M)   # 0 = shallow, 1 = deep
    peak_tokens = [12, 14, 16, 18, 20, 22]
    widths = [4, 5, 4, 6, 5, 7]
    amps = [0.6, 0.8, 0.5, 0.9, 0.7, 1.0]

    traces = []
    for i in range(M):
        # Gaussian bump around peak token
        bump = amps[i] * np.exp(-0.5 * ((t - peak_tokens[i]) / widths[i]) ** 2)
        # Add low-freq baseline oscillation and noise
        base = 0.08 * np.sin(2 * np.pi * t / 12 + rng.uniform(0, 2 * np.pi))
        noise = rng.standard_normal(S) * 0.04
        traces.append(bump + base + noise)

    fig, ax = plt.subplots(figsize=(12, 6))
    offset_step = 1.2
    depth_cmap = plt.cm.coolwarm

    for i, tr in enumerate(traces):
        offset = (M - 1 - i) * offset_step
        col = depth_cmap(depths[i])
        ax.fill_between(t, offset, offset + tr, alpha=0.25, color=col)
        ax.plot(t, offset + tr, color=col, lw=2.0, label=f"P{i}  (depth={depths[i]:.1f})")
        ax.axhline(offset, color="#ddd", lw=0.8)
        ax.text(-1.5, offset + 0.3, f"P{i}", fontsize=9, ha="right",
                va="center", color=col, fontweight="bold")

    ax.axvline(onset, color="#e15759", lw=2, ls="--", alpha=0.8)
    ax.text(onset + 0.3, M * offset_step + 0.1, "stimulus\nonset",
            color="#e15759", fontsize=9, va="top")

    ax.set_xlabel("Token position  (proxy: time)", fontsize=11)
    ax.set_ylabel("Floating score  $f_m(x)$  (+ offset per patch)", fontsize=11)
    ax.set_title(
        "Virtual EEG: Network Patch Floating Scores Over Sequence\n"
        r"$f_m(x) = \|\tilde{A}_m(x) - R_m\|_F^2 \;/\; \|R_m\|_F^2$",
        fontsize=12,
    )
    ax.set_yticks([])
    ax.set_xlim(-3, S + 1)

    # Colour legend
    handles = [
        mpatches.Patch(fc=depth_cmap(depths[i]), label=f"P{i}")
        for i in range(M)
    ]
    ax.legend(handles=handles, title="Patch (shallow→deep)",
              loc="upper right", ncol=2, fontsize=9)

    _save(fig, "fig5_virtual_eeg.png")


# ===========================================================================
# Figure 6 — Phase Transition Curve
# ===========================================================================

def fig6_phase_transition() -> None:
    _apply_style()
    rng = np.random.default_rng(3)

    alpha_star = 0.38
    k = 12.0
    alphas = np.linspace(0.05, 1.0, 200)
    auroc_mean = 0.5 + 0.38 * _sigmoid(alphas, k=k, x0=alpha_star)
    auroc_std = 0.04 * np.ones_like(alphas)

    # 1st and 2nd derivatives
    d1 = np.gradient(auroc_mean, alphas)
    d2 = np.gradient(d1, alphas)
    d2_abs = np.abs(d2)
    alpha_crit_idx = np.argmax(d2_abs)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7),
                                          gridspec_kw={"hspace": 0.08})

    # Top: AUROC curve
    ax_top.fill_between(
        alphas,
        auroc_mean - auroc_std,
        auroc_mean + auroc_std,
        alpha=0.25, color="#4e79a7",
    )
    ax_top.plot(alphas, auroc_mean, color="#4e79a7", lw=2.5, label="Task AUROC")
    ax_top.axhline(0.5, color="#888", lw=1.5, ls="--", label="Chance (0.5)")
    ax_top.axvline(alphas[alpha_crit_idx], color="#e15759", lw=2, ls=":",
                   label=rf"$\alpha^*={alphas[alpha_crit_idx]:.2f}$ (phase transition)")

    # Annotations
    ax_top.text(0.12, 0.56, "Degraded\ninner speech", ha="center",
                fontsize=10, color="#888", style="italic")
    ax_top.text(0.85, 0.87, "Vivid\ninner speech", ha="center",
                fontsize=10, color="#4e79a7", style="italic")

    ax_top.set_ylabel("Task AUROC", fontsize=11)
    ax_top.set_ylim(0.42, 0.95)
    ax_top.set_xlim(0.05, 1.0)
    ax_top.set_xticklabels([])
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.set_title(
        r"Phase Transition: AUROC vs Sparsity Level $\alpha$",
        fontsize=13,
    )

    # Bottom: 2nd derivative
    ax_bot.plot(alphas, d2_abs, color="#f28e2b", lw=2.5)
    ax_bot.fill_between(alphas, 0, d2_abs, alpha=0.2, color="#f28e2b")
    ax_bot.axvline(alphas[alpha_crit_idx], color="#e15759", lw=2, ls=":")
    ax_bot.text(
        alphas[alpha_crit_idx] + 0.02, d2_abs[alpha_crit_idx] * 0.85,
        rf"$\alpha^* = {alphas[alpha_crit_idx]:.2f}$",
        color="#e15759", fontsize=10,
    )
    ax_bot.set_xlabel(r"Sparsity level $\alpha$", fontsize=11)
    ax_bot.set_ylabel(r"$\left|d^2\,\mathrm{AUROC}/d\alpha^2\right|$", fontsize=11)
    ax_bot.set_xlim(0.05, 1.0)

    _save(fig, "fig6_phase_transition.png")


# ===========================================================================
# Figure 7 — Sensor Count Degradation Curves
# ===========================================================================

def fig7_sensor_degradation() -> None:
    _apply_style()
    rng = np.random.default_rng(9)

    M_vals = np.arange(1, 13)
    n_rep = 5

    def _auroc_curve(M: int) -> np.ndarray:
        base = 0.5 + 0.35 * _sigmoid(M, k=1.1, x0=4.5)
        return base + rng.standard_normal(n_rep) * 0.025

    def _r2_curve(M: int) -> np.ndarray:
        base = 0.1 + 0.75 * _sigmoid(M, k=1.2, x0=5.0)
        return np.clip(base + rng.standard_normal(n_rep) * 0.03, 0, 1)

    auroc = np.stack([_auroc_curve(m) for m in M_vals])
    r2 = np.stack([_r2_curve(m) for m in M_vals])

    auroc_mean, auroc_std = auroc.mean(1), auroc.std(1)
    r2_mean, r2_std = r2.mean(1), r2.std(1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mean, std, ylabel, chance, claim in [
        (ax_a, auroc_mean, auroc_std, "Task AUROC", 0.5, "Claim A"),
        (ax_b, r2_mean, r2_std, r"$R^2$ (sparsity prediction)", 0.0, "Control C"),
    ]:
        ax.fill_between(M_vals, mean - std, mean + std, alpha=0.22,
                        color="#4e79a7")
        ax.errorbar(M_vals, mean, yerr=std, fmt="o-", color="#4e79a7",
                    lw=2.5, ms=7, capsize=4, label=ylabel)
        ax.axhline(chance, color="#888", ls="--", lw=1.5, label="Chance")
        ax.axvline(12, color="#59a14f", ls=":", lw=2, label="Full coverage (M=12)")

        # Elbow annotation
        elbow = 5
        ax.annotate(
            "elbow", xy=(elbow, mean[elbow - 1]),
            xytext=(elbow + 1.5, mean[elbow - 1] - 0.06),
            arrowprops=dict(arrowstyle="->", color="#e15759", lw=1.5),
            fontsize=9, color="#e15759",
        )
        ax.set_xlabel("Patch count M", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Sensor Degradation — {claim}", fontsize=12)
        ax.set_xticks(M_vals)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Minimum Viable Sensor Count: AUROC and R² vs Patch Count M",
        fontsize=13, fontweight="bold", y=1.01,
    )
    _save(fig, "fig7_sensor_degradation.png")


# ===========================================================================
# Figure 8 — Full Pipeline Overview
# ===========================================================================

def fig8_pipeline_overview() -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 5.5)
    ax.set_title(
        "EEGSCRFP: Full Pipeline — Frozen LM as Unfolded Neural Pathway",
        fontsize=13, fontweight="bold", pad=12,
    )

    def box(ax, cx, cy, w, h, label, sublabel="", fc="#dce8f5", ec="#4e79a7",
            fontsize=10):
        rx, ry = cx - w / 2, cy - h / 2
        rect = FancyBboxPatch(
            (rx, ry), w, h,
            boxstyle="round,pad=0.12",
            fc=fc, ec=ec, lw=2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.12 if sublabel else 0), label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#222", zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.28, sublabel, ha="center", va="center",
                    fontsize=8, color="#555", style="italic", zorder=4)

    def arrow(ax, x0, y0, x1, y1, label="", col="#555", lw=2):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=col, lw=lw),
            zorder=2,
        )
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx + 0.1, my + 0.12, label, fontsize=8,
                    color="#666", ha="left", va="bottom", style="italic")

    # Input text
    box(ax, 1.1, 2.75, 1.8, 0.9,
        "Inner-speech\nprompt",
        r"$x \in \mathcal{D}$",
        fc="#fff8e8", ec="#edc948")

    # Frozen LM
    box(ax, 3.3, 2.75, 1.9, 1.1,
        "Frozen GPT-2",
        r"topk$_\alpha$ sparsity",
        fc="#e8f0fe", ec="#4e79a7")

    arrow(ax, 2.0, 2.75, 2.35, 2.75)

    # Routing graph
    box(ax, 5.4, 2.75, 1.9, 0.85,
        "Routing graph",
        r"$\mathcal{G}(x,\alpha)$  $\in \mathbb{R}^{L \times H \times S \times S}$",
        fc="#fce4ec", ec="#e15759")

    arrow(ax, 4.25, 2.75, 4.45, 2.75)

    # Two-branch split
    arrow(ax, 6.35, 3.1, 7.55, 4.0, col="#4e79a7")
    arrow(ax, 6.35, 2.4, 7.55, 1.5, col="#76b7b2")

    # Branch A: Patches → floating scores
    box(ax, 8.55, 4.3, 2.0, 0.75,
        "Patch extraction",
        r"$M$ windows $(\lambda_m, \eta_m)$",
        fc="#e8f5e9", ec="#59a14f")
    box(ax, 8.55, 3.3, 2.0, 0.75,
        "Floating scores",
        r"$\mathbf{f}(x) \in \mathbb{R}^M$",
        fc="#e8f5e9", ec="#59a14f")
    arrow(ax, 8.55, 3.92, 8.55, 3.67, col="#59a14f")

    # Branch B: Hidden states → CKA
    box(ax, 8.55, 1.7, 2.0, 0.75,
        "Hidden states",
        r"$\{X^\ell\}_{\ell=0}^{L}$",
        fc="#f3e5f5", ec="#b07aa1")
    box(ax, 8.55, 0.7, 2.0, 0.75,
        "CKA features",
        r"$\Phi \in \mathbb{R}^{L(L+1)/2}$",
        fc="#f3e5f5", ec="#b07aa1")
    arrow(ax, 8.55, 1.32, 8.55, 1.07, col="#b07aa1")

    # Concat → encoder
    arrow(ax, 9.55, 3.3, 10.2, 2.75, col="#59a14f")
    arrow(ax, 9.55, 0.7, 10.2, 2.25, col="#b07aa1")

    box(ax, 11.1, 2.5, 1.9, 0.9,
        "Patch encoder",
        "linear / MLP / transformer",
        fc="#fff3e0", ec="#f28e2b")

    # Three output heads
    arrow(ax, 12.05, 2.9, 12.75, 4.1, col="#e15759")
    arrow(ax, 12.05, 2.5, 12.75, 2.5, col="#4e79a7")
    arrow(ax, 12.05, 2.1, 12.75, 0.9, col="#76b7b2")

    box(ax, 13.35, 4.2, 1.2, 0.65,
        "Task AUROC", "Claim A", fc="#fce4ec", ec="#e15759", fontsize=9)
    box(ax, 13.35, 2.5, 1.2, 0.65,
        "Prompt sim.", "Claim B", fc="#e3f2fd", ec="#4e79a7", fontsize=9)
    box(ax, 13.35, 0.8, 1.2, 0.65,
        r"$\alpha$  R²", "Control C", fc="#e8f5e9", ec="#76b7b2", fontsize=9)

    _save(fig, "fig8_pipeline_overview.png")


# ===========================================================================
# Main
# ===========================================================================

FIGURES = [
    ("fig1_unfolded_pathway",   fig1_unfolded_pathway),
    ("fig2_alpha_neuromodulation", fig2_alpha_neuromodulation),
    ("fig3_patch_to_sensor",    fig3_patch_to_sensor),
    ("fig4_cka_flow",           fig4_cka_flow),
    ("fig5_virtual_eeg",        fig5_virtual_eeg),
    ("fig6_phase_transition",   fig6_phase_transition),
    ("fig7_sensor_degradation", fig7_sensor_degradation),
    ("fig8_pipeline_overview",  fig8_pipeline_overview),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate EEGSCRFP docs figures")
    parser.add_argument(
        "--only", nargs="+", metavar="FIG",
        help="Generate only these figures (e.g. --only fig1 fig4)",
    )
    args = parser.parse_args()

    selected = FIGURES
    if args.only:
        keep = {n for n in args.only}
        selected = [(n, f) for (n, f) in FIGURES if any(n.startswith(k) for k in keep)]
        if not selected:
            print(f"No figures matched {args.only}. Available: {[n for n,_ in FIGURES]}")
            raise SystemExit(1)

    print(f"Generating {len(selected)} figure(s) → {OUT.relative_to(REPO_ROOT)}/")
    for name, fn in selected:
        print(f"  [{name}]")
        fn()

    print("Done.")
