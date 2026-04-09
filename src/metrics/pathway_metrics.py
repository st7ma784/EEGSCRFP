"""Compute pathway metrics from attention maps — fully vectorised, batch-aware.

All public functions accept a stacked attention tensor [L, B, H, S, S] and
return a per-sample vector [B], so the entire batch is processed in one CUDA
kernel dispatch.  No Python loops, no NumPy, no scipy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

_EPS = 1e-10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stack(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """List of L tensors [B, H, S, S] → stacked [L, B, H, S, S]."""
    return torch.stack(attention_maps, dim=0)


def _per_sample_flat(stacked: torch.Tensor) -> torch.Tensor:
    """[L, B, H, S, S] → [B, L*H*S*S] — all attention weights per sample."""
    L, B, H, S, _ = stacked.shape
    return stacked.permute(1, 0, 2, 3, 4).reshape(B, L * H * S * S)


# ---------------------------------------------------------------------------
# Per-metric functions — each returns [B]
# ---------------------------------------------------------------------------

def routing_sparsity(stacked: torch.Tensor) -> torch.Tensor:
    """Rényi-2 diversity of attention weights — lower = more sparse. [B]"""
    flat = _per_sample_flat(stacked).clamp(min=0)          # [B, N]
    flat = flat * (flat > 1e-5)                             # zero near-zero
    probs = flat / flat.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    renyi = -torch.log(probs.pow(2).sum(dim=-1) + _EPS)    # [B]
    return renyi.exp()


def path_competition_index(stacked: torch.Tensor) -> torch.Tensor:
    """max / mean of non-zero weights — higher = more winner-take-all. [B]"""
    flat = _per_sample_flat(stacked).clamp(min=0)           # [B, N]
    mask = (flat > 1e-5).float()
    max_vals = flat.max(dim=-1).values                      # [B]
    counts = mask.sum(dim=-1).clamp(min=1)
    mean_vals = (flat * mask).sum(dim=-1) / counts          # [B]
    return max_vals / (mean_vals + _EPS)


def path_efficiency(stacked: torch.Tensor, topk_percent: float = 0.2) -> torch.Tensor:
    """Energy fraction in top-k% weights — higher = more concentrated. [B]"""
    flat = _per_sample_flat(stacked).clamp(min=0)           # [B, N]
    k = max(1, int(flat.shape[-1] * topk_percent))
    topk_sum = torch.topk(flat, k=k, dim=-1).values.sum(dim=-1)   # [B]
    total = flat.sum(dim=-1).clamp(min=_EPS)                       # [B]
    return topk_sum / total


def routing_entropy(stacked: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy over all per-position attention rows. [B]"""
    L, B, H, S, _ = stacked.shape
    # [L*B*H*S, S] — each row is one attention distribution over S keys
    rows = stacked.permute(1, 0, 2, 3, 4).reshape(B * L * H * S, S)
    probs = rows / rows.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    h = -(probs * torch.log(probs + _EPS)).sum(dim=-1)      # [B*L*H*S]
    return h.reshape(B, L * H * S).mean(dim=-1)             # [B]


def inter_head_divergence(stacked: torch.Tensor) -> torch.Tensor:
    """Mean pairwise KL divergence between attention heads. [B]"""
    L, B, H, S, _ = stacked.shape
    if H < 2:
        return stacked.new_zeros(B)

    lb_h_ss = stacked.reshape(L * B, H, S * S)             # [LB, H, S*S]
    p = lb_h_ss / lb_h_ss.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    log_p = torch.log(p + _EPS)                            # [LB, H, S*S]

    # KL(p_i || p_j) = Σ_k p_ik (log p_ik - log p_jk)
    # = (p * log_p).sum(-1)_i  -  (p_i @ log_p_j^T).sum(-1)
    self_term = (p * log_p).sum(dim=-1)                     # [LB, H]
    cross = torch.bmm(p, log_p.transpose(1, 2))            # [LB, H, H]
    kl = self_term.unsqueeze(2) - cross                     # [LB, H, H]  KL[i,j]

    # Upper-triangle mask (exclude diagonal)
    mask = torch.triu(torch.ones(H, H, device=stacked.device, dtype=torch.bool), diagonal=1)
    kl_per_lb = kl[:, mask].mean(dim=-1)                    # [LB]
    return kl_per_lb.reshape(L, B).mean(dim=0)             # [B]


def layer_stability(stacked: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between consecutive layer attention maps. [B]"""
    L, B, H, S, _ = stacked.shape
    if L < 2:
        return stacked.new_ones(B)

    flat = stacked.reshape(L, B, H * S * S)                # [L, B, H*S*S]
    # Batch cosine similarity over L-1 consecutive pairs
    cos = F.cosine_similarity(flat[:-1], flat[1:], dim=-1)  # [L-1, B]
    return cos.mean(dim=0)                                  # [B]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PathwayMetricsComputer(nn.Module):
    """Stateless nn.Module wrapper — moves to GPU with the parent module."""

    def forward(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            attention_maps: list of L tensors, each [B, H, S, S]
        Returns:
            [B, 6] pathway feature matrix
        """
        return compute_pathway_features(attention_maps)


def compute_pathway_features(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """Compute all 6 pathway metrics for a full batch in one pass.

    Args:
        attention_maps: list of L tensors, each [B, H, S, S]

    Returns:
        [B, 6] float32 tensor on the same device as the inputs
    """
    stacked = _stack(attention_maps)   # [L, B, H, S, S]
    return torch.stack([
        routing_sparsity(stacked),
        path_competition_index(stacked),
        path_efficiency(stacked),
        routing_entropy(stacked),
        inter_head_divergence(stacked),
        layer_stability(stacked),
    ], dim=-1)                         # [B, 6]
