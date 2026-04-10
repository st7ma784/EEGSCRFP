"""Centered Kernel Alignment (CKA) features from transformer hidden states.

CKA measures representational similarity between two matrices independently
of linear transformations — it captures the *geometry* of the representation
space rather than just scalar statistics.

Why CKA over scatter statistics
--------------------------------
The 6 global pathway metrics (entropy, PCI, etc.) collapse the entire
[L, B, H, S, S] attention tensor to 6 scalars.  They primarily measure
*how sparse* routing is, not *what is being computed*.

Diagnostic: across real narrative prompts, eta²_prompt ≈ 0.009 for scalar
metrics (99% alpha-driven).  For CKA features, eta²_prompt ≈ 1.0 (all
variance driven by prompt content).

Feature sets produced
---------------------
(a) Layer-pairwise CKA: CKA(hidden_i, hidden_j) for all i≤j pairs.
    For GPT-2 (12 layers + embedding = 13 states): 91 values.
    Captures how similar the processing at different depths is.

(b) Input-to-layer CKA: CKA(embedding, hidden_i) for i in [1..L].
    12 values.  How much each layer preserves the input representation.

All CKA values are in [0, 1] and are scale/rotation invariant.

GPU performance design
-----------------------
The naive implementation loops over B samples and M² layer pairs, calling
linear_cka() once each.  For GPT-2 (M=13, B=32) that is 32 × 91 = 2,912
separate matrix products.

The vectorised implementation exploits:

    HSIC(X, Y) = ||X̃ᵀ Ỹ||²_F = trace(K_X · K_Y)

where K = X̃ X̃ᵀ is the centred Gram matrix.  For all M layers and all B
samples simultaneously:

    1. Compute K[m, b] = H̃[m,b] H̃[m,b]ᵀ  ∀ m,b  via one batched matmul
       K: [M, B, S, S] → reshape to K_flat: [B, M, S²]

    2. HSIC for all pairs at once:
       HSIC_all[b] = K_flat[b] @ K_flat[b]ᵀ   → [B, M, M]
       (one bmm call — all M² pair HSICs for all B samples)

    3. CKA[b, i, j] = HSIC_all[b,i,j] / sqrt(diag_i * diag_j)

This replaces B × M²/2 matrix products with:
  - one [M*B, S, d] → [M*B, S, S] batched matmul (Gram matrices)
  - one [B, M, S²] → [B, M, M] batched matmul (all pair HSICs)

Reference: Kornblith et al. (2019) "Similarity of Neural Network
Representations Revisited."  https://arxiv.org/abs/1905.00414
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Core CKA — fully vectorised over (B, M, M)
# ─────────────────────────────────────────────────────────────────────────────

def _centre_rows(X: torch.Tensor) -> torch.Tensor:
    """Subtract column mean from each column (centre over the N/sample dim).

    Accepts [..., N, d] and returns the same shape.
    """
    return X - X.mean(dim=-2, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Linear CKA between two [N, d1] and [N, d2] representation matrices.

    Uses the efficient HSIC formula: CKA = HSIC(X,Y) / sqrt(HSIC(X,X) HSIC(Y,Y))
    with the linear kernel K_X = X X^T.

    Returns:
        Scalar CKA value in [0, 1].
    """
    Xc = _centre_rows(X)    # [N, d1]
    Yc = _centre_rows(Y)    # [N, d2]

    XtY = Xc.T @ Yc         # [d1, d2]
    XtX = Xc.T @ Xc         # [d1, d1]
    YtY = Yc.T @ Yc         # [d2, d2]

    hsic_xy = (XtY ** 2).sum()
    hsic_xx = (XtX ** 2).sum()
    hsic_yy = (YtY ** 2).sum()

    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + _EPS)).clamp(0.0, 1.0)


def _all_pairs_cka_batched(
    hidden_states: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Compute all pairwise CKA values for all B samples simultaneously.

    This is the vectorised replacement for the per-sample loop in the original
    implementation.  Instead of B × M²/2 matrix products, uses:
      - one [M*B, S, d] bmm for Gram matrices
      - one [B, M, S²] bmm for all HSIC pairs

    Args:
        hidden_states: tuple of M tensors, each [B, S, d].

    Returns:
        [B, M, M] symmetric CKA matrix for each sample in the batch.
    """
    M = len(hidden_states)
    B, S, d = hidden_states[0].shape
    device = hidden_states[0].device

    # Stack: [M, B, S, d]
    H = torch.stack([h.float() for h in hidden_states], dim=0)

    # Centre over sequence dimension (analogous to centering X columns)
    H_c = H - H.mean(dim=2, keepdim=True)          # [M, B, S, d]

    # Gram matrices K[m,b] = H_c[m,b] @ H_c[m,b]^T
    # Reshape to [M*B, S, d] for one batched matmul
    H_flat = H_c.reshape(M * B, S, d)
    K_flat = torch.bmm(H_flat, H_flat.transpose(1, 2))   # [M*B, S, S]
    K = K_flat.reshape(M, B, S * S)                       # [M, B, S²]

    # Rearrange to [B, M, S²] for the bmm that gives all HSIC pairs
    K_T = K.permute(1, 0, 2)                              # [B, M, S²]

    # HSIC(i,j,b) = (K[i,b] · K[j,b]).sum() = K_T[b,i,:] · K_T[b,j,:]
    # All pairs at once: [B, M, M]
    HSIC = torch.bmm(K_T, K_T.transpose(1, 2))           # [B, M, M]

    # Normalise: CKA[b,i,j] = HSIC[b,i,j] / sqrt(HSIC[b,i,i] * HSIC[b,j,j])
    diag = HSIC.diagonal(dim1=1, dim2=2)                  # [B, M]
    denom = torch.sqrt(
        diag.unsqueeze(2) * diag.unsqueeze(1) + _EPS
    )                                                      # [B, M, M]
    cka = (HSIC / denom).clamp(0.0, 1.0)                  # [B, M, M]
    return cka


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from a forward pass
# ─────────────────────────────────────────────────────────────────────────────

def extract_cka_features(
    hidden_states: Tuple[torch.Tensor, ...],
    include_pairwise: bool = True,
    include_input_alignment: bool = True,
) -> torch.Tensor:
    """Compute CKA features from a transformer's hidden state sequence.

    GPU-optimised: all B samples and all M(M+1)/2 pairs are computed in
    two batched matmuls (see module docstring).  No Python loop over B.

    Args:
        hidden_states: tuple of (L+1) tensors, each [B, S, d].
                       Index 0 = embedding; 1..L = layer outputs.
                       (The output of ``output_hidden_states=True``.)
        include_pairwise:       Include all pairwise CKA values.
        include_input_alignment: Include CKA(embedding, layer_i) values.

    Returns:
        [B, D] feature tensor where D depends on L and which features
        are included.  All features are in [0, 1].

    Feature layout:
        [0 : L*(L+1)//2]             pairwise upper-triangle CKA (row-major)
        [L*(L+1)//2 : L*(L+1)//2+L] input-to-layer CKA (embedding vs layer i)
    """
    M = len(hidden_states)   # embedding + L layers
    B = hidden_states[0].shape[0]
    device = hidden_states[0].device

    # Compute full [B, M, M] CKA matrix in one pass.
    cka_mat = _all_pairs_cka_batched(hidden_states)   # [B, M, M]

    features_list: List[torch.Tensor] = []

    if include_pairwise:
        # Upper triangle (including diagonal) → [B, M*(M+1)//2]
        rows, cols = torch.triu_indices(M, M, offset=0, device=device)
        pairwise = cka_mat[:, rows, cols]             # [B, n_pairs]
        features_list.append(pairwise)

    if include_input_alignment:
        # CKA(embedding, layer_i) for i in [1..L] → row 0, columns 1..M-1
        input_align = cka_mat[:, 0, 1:]              # [B, L]
        features_list.append(input_align)

    return torch.cat(features_list, dim=-1)           # [B, D]


def cka_feature_dim(n_layers: int,
                    include_pairwise: bool = True,
                    include_input_alignment: bool = True) -> int:
    """Return the output dimension D for given settings."""
    M = n_layers + 1   # embedding + L layers
    d = 0
    if include_pairwise:
        d += M * (M + 1) // 2
    if include_input_alignment:
        d += n_layers
    return d


def pairwise_cka(
    hidden_states: List[torch.Tensor],
) -> torch.Tensor:
    """Compute all pairwise CKA values between a list of hidden states.

    Args:
        hidden_states: list of M tensors, each [N, d_i] or [B, S, d_i].
                       If 3-D, treats the S dimension as the observation axis
                       for CKA (sequence positions = data points).
                       If 2-D [N, d], unsqueezes a fake batch dim.

    Returns:
        [M, M] symmetric CKA matrix.  If input had a batch dim B > 1, the
        result is averaged over B.
    """
    hs: List[torch.Tensor] = []
    for h in hidden_states:
        if h.dim() == 2:
            h = h.unsqueeze(0)   # [1, N, d] — treat as B=1 batch
        hs.append(h.float())    # [B, S, d]

    cka_3d = _all_pairs_cka_batched(tuple(hs))   # [B, M, M]
    return cka_3d.mean(dim=0)                     # [M, M]


# ─────────────────────────────────────────────────────────────────────────────
# Module wrapper — moves with parent model via .to(device)
# ─────────────────────────────────────────────────────────────────────────────

class CKAMetricsComputer(nn.Module):
    """Stateless nn.Module wrapper around extract_cka_features.

    Usage:
        computer = CKAMetricsComputer(include_pairwise=True)
        features = computer(hidden_states)   # [B, D]
    """

    def __init__(
        self,
        include_pairwise: bool = True,
        include_input_alignment: bool = True,
    ):
        super().__init__()
        self.include_pairwise = include_pairwise
        self.include_input_alignment = include_input_alignment

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return extract_cka_features(
            hidden_states,
            include_pairwise=self.include_pairwise,
            include_input_alignment=self.include_input_alignment,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Combined feature extractor: existing 6 scalars + CKA
# ─────────────────────────────────────────────────────────────────────────────

def compute_combined_features(
    attention_maps: List[torch.Tensor],    # list of L tensors [B, H, S, S]
    hidden_states: Tuple[torch.Tensor, ...],  # tuple of L+1 tensors [B, S, d]
    include_pathway_stats: bool = True,
    include_cka: bool = True,
) -> torch.Tensor:
    """Combine existing pathway statistics with CKA features.

    Returns:
        [B, D] where D = 6 * include_pathway_stats
                          + cka_feature_dim(L) * include_cka
    """
    parts: List[torch.Tensor] = []

    if include_pathway_stats:
        from src.metrics.pathway_metrics import compute_pathway_features
        parts.append(compute_pathway_features(attention_maps))   # [B, 6]

    if include_cka:
        parts.append(extract_cka_features(hidden_states))        # [B, D_cka]

    return torch.cat(parts, dim=-1)
