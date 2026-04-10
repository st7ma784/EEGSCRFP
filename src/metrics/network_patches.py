"""Network patch sampling — the EEG sensor model.

Motivation
----------
A real EEG electrode integrates activity from a *local spatial cluster* of
neurons.  It does not read out a mathematically precise global statistic.

We model this with "network patches": each patch covers a contiguous window
of adjacent layers and a random subset of attention heads.  Within the patch,
attention weights are pooled to produce a single scalar per input — the
**floating score**: how much does this input's routing through this patch
deviate from the population-mean routing?

    Fixed path   → low floating score  (patch routes identically for all inputs)
    Floating path → high floating score (patch responds to this input specifically)

Task-relevant computation concentrates in patches with high floating scores.

GPU performance design
-----------------------
All hot paths are branch-free and loop-free on the Python side:

1. ``PatchFeatureExtractor.__init__`` pre-computes integer index tensors for
   every patch's (layer, head) pairs and registers them as buffers, so they
   move with ``.to(device)`` and never require Python iteration in forward().

2. ``forward()`` stacks the attention maps once, then selects all patch means
   using advanced tensor indexing — no Python loop over patches.

3. All M floating scores are computed in a single fused kernel call by
   operating on the stacked [M, B, S²] tensor.

4. ``calibrate()`` similarly stacks all samples and all patches into one
   batched matrix operation.

Sensor count experiment
-----------------------
Full coverage: enough patches to sample every (layer, head) combination at least once.
For GPT-2 (12 layers, 12 heads) with depth=3, heads_per_patch=4:
    → 4 layer windows × 3 head groups = 12 patches

Scale M from 1 to 12 (or to full-overlap count).
Plot task-prediction accuracy vs M.
The elbow gives the minimum sensor count.

Usage
-----
    sampler = PatchSampler(n_layers=12, n_heads=12)
    patches = sampler.full_coverage_patches()         # 12 patches
    extractor = PatchFeatureExtractor(patches)
    extractor.calibrate(attention_maps_reference)     # set population baseline
    features = extractor(attention_maps)              # [B, M=12]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Patch definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkPatch:
    """A local sample of the transformer's routing structure.

    Covers a contiguous window of *layer_indices* and a random subset of
    *head_indices* within those layers.

    Attributes
    ----------
    layer_indices : list of int
        Indices of the consecutive layers included in this patch.
    head_indices : list of int
        Indices of the attention heads sampled from those layers.
    """
    layer_indices: List[int]
    head_indices: List[int]

    def __repr__(self):
        return (f"NetworkPatch(layers={self.layer_indices}, "
                f"heads={self.head_indices})")

    @property
    def n_layers(self) -> int:
        return len(self.layer_indices)

    @property
    def n_heads(self) -> int:
        return len(self.head_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised helpers (no Python loops)
# ─────────────────────────────────────────────────────────────────────────────

def _stack_attention(
    attention_maps: List[torch.Tensor],
) -> torch.Tensor:
    """List of L tensors [B, H, S, S] → stacked [L, B, H, S, S].

    Called once per forward pass; subsequent operations index into this tensor.
    """
    return torch.stack(attention_maps, dim=0)   # [L, B, H, S, S]


def _patch_mean_from_stacked(
    stacked: torch.Tensor,          # [L, B, H, S, S]
    layer_idx: torch.Tensor,        # [n_l]  long — pre-computed at init
    head_idx: torch.Tensor,         # [n_h]  long — pre-computed at init
) -> torch.Tensor:
    """Mean attention map for one patch, fully vectorised.  Returns [B, S, S].

    Uses advanced indexing to select the (n_l, n_h) slice in a single
    gather operation, then reduces with .mean().
    """
    # stacked[layer_idx]: [n_l, B, H, S, S]
    # [:, :, head_idx]:   [n_l, B, n_h, S, S]
    # .mean(0).mean(1):   [B, S, S]
    return stacked[layer_idx][:, :, head_idx].mean(dim=0).mean(dim=1)


def floating_score_batched(
    patch_attns: torch.Tensor,          # [M, B, S, S]
    references: Optional[torch.Tensor], # [M, S_ref, S_ref] or None
) -> torch.Tensor:
    """Compute floating scores for all M patches in one fused operation.

    Handles the common case where calibration and inference sequence lengths
    differ (e.g. calibrated on S=11, inference batch padded to S=10).
    When S != S_ref we use the shared prefix min(S, S_ref) of token positions.

    Returns [B, M].
    """
    M, B, S, _ = patch_attns.shape

    if references is not None and references.shape[-1] != S:
        S_use = min(S, references.shape[-1])
        flat = patch_attns[:, :, :S_use, :S_use].reshape(M, B, S_use * S_use)
        ref  = references[:, :S_use, :S_use].reshape(M, 1, S_use * S_use)
    else:
        flat = patch_attns.reshape(M, B, S * S)         # [M, B, S²]
        ref  = (references.reshape(M, 1, S * S)
                if references is not None
                else flat.mean(dim=1, keepdim=True))     # [M, 1, S²]

    # Per-position std over batch for normalisation.
    ref_std = flat.std(dim=1, keepdim=True).clamp(min=_EPS)   # [M, 1, S²]
    z = (flat - ref) / ref_std                                 # [M, B, S²]
    scores = z.pow(2).mean(dim=-1)                             # [M, B]
    return scores.T                                            # [B, M]


def floating_score(
    patch_attn: torch.Tensor,            # [B, S, S] mean attention in patch
    reference: Optional[torch.Tensor] = None,  # [S, S] population mean
) -> torch.Tensor:
    """Per-input floating score for a single patch.

    Convenience wrapper around floating_score_batched for the single-patch case.
    Returns [B].
    """
    scores = floating_score_batched(
        patch_attn.unsqueeze(0),
        reference.unsqueeze(0) if reference is not None else None,
    )
    return scores[:, 0]     # [B]


def floating_score_temporal_batched(
    patch_attns: torch.Tensor,          # [M, B, S, S]
    references: Optional[torch.Tensor], # [M, S_ref, S_ref] or None
) -> torch.Tensor:
    """Per-query-position floating scores — preserves the token time axis.

    Instead of collapsing S² to a scalar (like floating_score_batched), this
    collapses only the key axis, giving one score per query position t:

        score[m, b, t] = mean_k( ((attn[m,b,t,k] - ref[m,t,k]) / std[m,t,k])² )

    Returns [B, M, S] which maps directly to the EEG encoder interface [B, C, T]:
        C = M  (virtual electrodes = patches)
        T = S  (time = token query positions)

    Feed this output into PatchEEGEncoder / EEGProjector from EEGViewer
    by instantiating the encoder with n_channels=M and patch_size chosen to
    match the expected sequence length S.
    """
    M, B, S, _ = patch_attns.shape

    if references is not None and references.shape[-1] != S:
        S_use = min(S, references.shape[-1])
        attn = patch_attns[:, :, :S_use, :S_use]           # [M, B, S_use, S_use]
        ref  = references[:, :S_use, :S_use].unsqueeze(1)  # [M, 1, S_use, S_use]
    else:
        attn = patch_attns
        ref  = (references.unsqueeze(1)
                if references is not None
                else attn.mean(dim=1, keepdim=True))        # [M, 1, S, S]

    # Std over batch, per (query-position, key-position) — [M, 1, S, S]
    ref_std = attn.std(dim=1, keepdim=True).clamp(min=_EPS)
    z = (attn - ref) / ref_std                              # [M, B, S, S]
    scores = z.pow(2).mean(dim=-1)                          # [M, B, S] — mean over keys
    return scores.permute(1, 0, 2)                          # [B, M, S]


# ─────────────────────────────────────────────────────────────────────────────
# Full extractor: all patches → feature matrix
# ─────────────────────────────────────────────────────────────────────────────

class PatchFeatureExtractor(nn.Module):
    """Apply M patches to a forward pass and return [B, M] floating scores.

    GPU performance
    ---------------
    * Index tensors for every patch are pre-computed in __init__ and
      registered as buffers so they move with .to(device).
    * forward() contains no Python loops: all patches are indexed and
      reduced in one pass over the stacked attention tensor, then all
      floating scores are computed in a single batched kernel.
    * References (population baselines) are stored as a single [M, S, S]
      buffer rather than a Python list.

    Call ``calibrate()`` on a representative dataset before using in the
    sensor count experiment.  Without calibration the batch mean is used
    as a per-batch reference, which still works but is less stable.
    """

    def __init__(self, patches: List[NetworkPatch]):
        super().__init__()
        self.patches = patches
        self.M = len(patches)

        # Pre-build index tensors for each patch — registered as buffers so
        # they are moved by .to(device) and appear in state_dict.
        for i, p in enumerate(patches):
            self.register_buffer(
                f"_layer_idx_{i}",
                torch.tensor(p.layer_indices, dtype=torch.long),
            )
            self.register_buffer(
                f"_head_idx_{i}",
                torch.tensor(p.head_indices, dtype=torch.long),
            )

        # Reference baselines: [M, S, S] — set by calibrate().
        # Shape depends on S (sequence length), so we cannot allocate it here.
        # Use None sentinel; calibrated in calibrate().
        self.register_buffer("_references", None)
        self._calibrated: bool = False

    # ------------------------------------------------------------------
    # Internal helpers — called only during __init__ or calibrate
    # ------------------------------------------------------------------

    def _layer_idx(self, i: int) -> torch.Tensor:
        return getattr(self, f"_layer_idx_{i}")

    def _head_idx(self, i: int) -> torch.Tensor:
        return getattr(self, f"_head_idx_{i}")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        attention_maps_list: List[List[torch.Tensor]],
    ) -> None:
        """Estimate the population-mean routing for each patch.

        Uses batched patch extraction per sample.  Handles variable sequence
        lengths across calibration samples by zero-padding all attention maps
        to S_max before accumulating, so the reference has a consistent shape.

        Args:
            attention_maps_list: N-length list; each element is a list of L
                tensors [1, H, S_i, S_i] from a single forward pass.
                S_i may differ across samples (before batched padding).
        """
        device = attention_maps_list[0][0].device
        N = len(attention_maps_list)

        # Find the largest sequence length seen in the calibration set.
        S_max = max(am[0].shape[-1] for am in attention_maps_list)

        acc = torch.zeros(self.M, S_max, S_max, device=device)

        for attn_maps in attention_maps_list:
            stacked = _stack_attention(attn_maps)   # [L, 1, H, S_i, S_i]
            S_i = stacked.shape[-1]

            # Batch-extract all M patch means for this sample → [M, 1, S_i, S_i]
            patch_means = torch.stack([
                _patch_mean_from_stacked(stacked, self._layer_idx(i), self._head_idx(i))
                for i in range(self.M)
            ], dim=0)                               # [M, 1, S_i, S_i]
            pm = patch_means[:, 0]                  # [M, S_i, S_i]

            # Pad to S_max if needed (zero-pad the token-position axes).
            if S_i < S_max:
                pm = F.pad(pm, (0, S_max - S_i, 0, S_max - S_i))

            acc += pm                               # [M, S_max, S_max]

        refs = acc / N                              # [M, S_max, S_max]
        self.register_buffer("_references", refs)
        self._calibrated = True

    # ------------------------------------------------------------------
    # Forward — no Python loops, no branches
    # ------------------------------------------------------------------

    def forward(
        self,
        attention_maps: List[torch.Tensor],  # list of L tensors [B, H, S, S]
    ) -> torch.Tensor:
        """Return [B, M] floating scores for the current batch.

        All M patches are extracted and scored in two batched tensor operations:
        1. One index-gather per patch (no loop; gather is fused on GPU).
        2. One call to floating_score_batched operating on [M, B, S, S].
        """
        stacked = _stack_attention(attention_maps)  # [L, B, H, S, S]
        B = stacked.shape[1]

        # Collect all patch means → [M, B, S, S]
        patch_means = torch.stack([
            _patch_mean_from_stacked(stacked, self._layer_idx(i), self._head_idx(i))
            for i in range(self.M)
        ], dim=0)

        return floating_score_batched(patch_means, self._references)  # [B, M]

    def forward_temporal(
        self,
        attention_maps: List[torch.Tensor],  # list of L tensors [B, H, S, S]
    ) -> torch.Tensor:
        """Return [B, M, S] per-position floating scores.

        Preserves the token query axis as a time dimension so the output
        maps directly to the EEG encoder interface [B, C, T]:
            C = M  (patches = virtual electrodes)
            T = S  (token positions = time samples)

        Use this to feed patch features into PatchEEGEncoder / EEGProjector.
        Use forward() instead for sklearn probes (sensor_count.py).
        """
        stacked = _stack_attention(attention_maps)  # [L, B, H, S, S]

        patch_means = torch.stack([
            _patch_mean_from_stacked(stacked, self._layer_idx(i), self._head_idx(i))
            for i in range(self.M)
        ], dim=0)                                   # [M, B, S, S]

        return floating_score_temporal_batched(patch_means, self._references)  # [B, M, S]


# ─────────────────────────────────────────────────────────────────────────────
# Patch generation
# ─────────────────────────────────────────────────────────────────────────────

class PatchSampler:
    """Generates patch sets for the sensor count experiment.

    Two modes:
    ``full_coverage_patches`` — non-overlapping windows covering every
        (layer, head) exactly once.  This is the "full network readout"
        baseline.

    ``random_patches`` — randomly sampled contiguous layer windows + random
        head subsets.  Used to simulate having M < full_coverage sensors.

    For GPT-2 (n_layers=12, n_heads=12, depth=3, heads_per_patch=4):
        full_coverage → 4 layer windows × 3 head groups = 12 patches
        random M=6    → 6 randomly placed patches
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        patch_depth: int = 3,
        heads_per_patch: int = 4,
        seed: int = 42,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.patch_depth = patch_depth
        self.heads_per_patch = heads_per_patch
        self._rng = np.random.default_rng(seed)

    def full_coverage_patches(self) -> List[NetworkPatch]:
        """Non-overlapping patches that cover every (layer, head) exactly once.

        Layer windows: stride = patch_depth (no overlap)
        Head groups:   stride = heads_per_patch (no overlap)

        Any remainder (when n_layers or n_heads not divisible) is absorbed
        into the last window.
        """
        patches: List[NetworkPatch] = []
        layer_starts = list(range(0, self.n_layers, self.patch_depth))
        head_starts = list(range(0, self.n_heads, self.heads_per_patch))

        for ls in layer_starts:
            layer_ids = list(range(ls, min(ls + self.patch_depth, self.n_layers)))
            for hs in head_starts:
                head_ids = list(range(hs, min(hs + self.heads_per_patch, self.n_heads)))
                patches.append(NetworkPatch(layer_ids, head_ids))
        return patches

    def random_patches(self, n: int) -> List[NetworkPatch]:
        """Generate *n* random contiguous-layer patches with random head subsets.

        Each patch:
        - Layer start: uniform in [0, n_layers - patch_depth]
        - Heads: random sample of size heads_per_patch (without replacement)
        """
        patches: List[NetworkPatch] = []
        max_start = max(0, self.n_layers - self.patch_depth)
        for _ in range(n):
            ls = int(self._rng.integers(0, max_start + 1))
            layer_ids = list(range(ls, min(ls + self.patch_depth, self.n_layers)))
            head_ids = sorted(self._rng.choice(
                self.n_heads, self.heads_per_patch, replace=False
            ).tolist())
            patches.append(NetworkPatch(layer_ids, head_ids))
        return patches

    def overlapping_coverage_patches(self, layer_stride: int = 1) -> List[NetworkPatch]:
        """Full-overlap patches — every contiguous window, every head group.

        Produces more patches than full_coverage but gives a finer M grid
        for the degradation experiment.

        Total = ceil(n_layers / layer_stride) × ceil(n_heads / heads_per_patch)
        """
        patches: List[NetworkPatch] = []
        layer_starts = list(range(0, self.n_layers - self.patch_depth + 1, layer_stride))
        head_starts = list(range(0, self.n_heads, self.heads_per_patch))
        for ls in layer_starts:
            layer_ids = list(range(ls, ls + self.patch_depth))
            for hs in head_starts:
                head_ids = list(range(hs, min(hs + self.heads_per_patch, self.n_heads)))
                patches.append(NetworkPatch(layer_ids, head_ids))
        return patches

    @property
    def full_coverage_count(self) -> int:
        n_lw = math.ceil(self.n_layers / self.patch_depth)
        n_hw = math.ceil(self.n_heads / self.heads_per_patch)
        return n_lw * n_hw

    def m_values_for_experiment(self, n_repeats: int = 3) -> List[int]:
        """Return the M values to sweep in the sensor count experiment.

        Includes 1, powers-of-2 up to full_coverage, and full_coverage.
        """
        fc = self.full_coverage_count
        ms: List[int] = sorted(set(
            [1] + [2 ** i for i in range(int(math.log2(max(fc, 2))) + 1)] + [fc]
        ))
        return [m for m in ms if m <= fc]
