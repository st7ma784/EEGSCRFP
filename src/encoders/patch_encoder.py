"""Encoder architectures for [B, M, S] patch temporal features.

Input contract
--------------
All encoders accept a single tensor of shape [B, M, S]:
    B = batch size
    M = number of patches (virtual electrodes)
    S = number of token query positions (time axis)

All encoders return [B, embed_dim].

Architecture variants
---------------------
linear      — flatten(M×S) → Linear → embed_dim.  Fully interpretable;
              each output dimension is a fixed linear combination of all
              patch×position features.

mlp         — flatten(M×S) → MLP with configurable depth and hidden_dim.
              Non-linear but still operates on the flattened feature vector.

transformer — project each patch's S-dim profile to embed_dim, treat the M
              patches as a sequence, apply transformer self-attention, pool
              with a CLS token.  Directly analogous to ViT: the M patches
              are "spatial tokens" and S is the per-token feature dimension.
              Recommended: attends across patches, learns which combinations
              of virtual electrodes carry task-relevant signal.

eeg_viewer  — wraps PatchEEGEncoder from the parent EEGViewer repo.
              Treats [B, M, S] as raw EEG [B, channels=M, time=S] with
              sample_rate=1.0 (one sample per token position).  The filter
              bank reduces to 4 parallel Conv1d(kernel=3) at this rate —
              useful for ablation but functionally equivalent to the
              transformer variant with four short convolutional front-ends.

Use ``build_patch_encoder()`` to instantiate any variant by name.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

_EncoderType = Literal["linear", "mlp", "transformer", "eeg_viewer"]


@dataclass
class PatchEncoderOutput:
    """Dual output from PatchTokenEncoder.

    task_vec:    [B, embed_dim] — CLS token after transformer.
                 Trained toward task/alpha discrimination.

    content_vec: [B, embed_dim] — mean of the M patch output tokens.
                 Preserves per-patch content information independently of the
                 CLS token's task objective.  Used by the recon head so that
                 prompt-identity information isn't destroyed by task training.
    """
    task_vec:    torch.Tensor   # [B, embed_dim]
    content_vec: torch.Tensor   # [B, embed_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_hidden_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers += [nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer variant
# ─────────────────────────────────────────────────────────────────────────────

class PatchTokenEncoder(nn.Module):
    """Transformer that attends across M virtual electrodes.

    Each patch's S-dimensional token-position profile is independently
    projected to embed_dim (a per-patch linear layer, no weight sharing across
    patches).  The M projected patches form a sequence; a CLS token pools them.

    Analogy: ViT where each patch is a virtual electrode and S is the
    "pixel" dimension inside that patch.

    Args:
        M:          Number of patches (virtual electrodes).
        S:          Token sequence length (time dimension).
        embed_dim:  Transformer hidden dimension.
        n_layers:   Transformer encoder depth.
        n_heads:    Number of self-attention heads.
        dropout:    Dropout rate applied inside transformer layers.
    """

    def __init__(
        self,
        M: int,
        S: int,
        embed_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.M = M
        self.S = S
        self.embed_dim = embed_dim

        # Per-patch projection: maps each patch's S-dim profile to embed_dim.
        # Weight matrix is [M, S, embed_dim] — each patch has its own projection.
        self.patch_proj = nn.Linear(S, embed_dim)

        # Learnable patch positional embeddings (which electrode is this?)
        self.patch_pos = nn.Parameter(torch.randn(1, M, embed_dim) * 0.02)

        # CLS token: pooled output after transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> PatchEncoderOutput:
        """
        Args:
            x: [B, M, S] patch temporal features.
        Returns:
            PatchEncoderOutput with task_vec [B, embed_dim] (CLS) and
            content_vec [B, embed_dim] (mean of M patch outputs).
        """
        B = x.shape[0]
        patches = self.patch_proj(x) + self.patch_pos   # [B, M, embed_dim]
        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, embed_dim]
        seq = torch.cat([cls, patches], dim=1)           # [B, M+1, embed_dim]
        out = self.norm(self.transformer(seq))           # [B, M+1, embed_dim]
        return PatchEncoderOutput(
            task_vec=out[:, 0],          # CLS  → task / alpha heads
            content_vec=out[:, 1:].mean(dim=1),  # patch mean → recon head
        )


# ─────────────────────────────────────────────────────────────────────────────
# EEGViewer PatchEEGEncoder wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _EEGViewerEncoderWrapper(nn.Module):
    """Wraps PatchEEGEncoder from EEGViewer for patch temporal features.

    Treats [B, M, S] as raw EEG [B, channels=M, time=S] with sample_rate=1.0.
    At sample_rate=1.0 all four filter bank kernels reduce to size 3 (minimum),
    so this is functionally a four-branch Conv1d(3) front-end followed by a
    transformer.  Useful as an ablation against the dedicated PatchTokenEncoder.
    """

    def __init__(
        self,
        M: int,
        S: int,
        embed_dim: int,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Locate EEGViewer src — EEGSCRFP lives at EEGViewer/EEGSCRFP/
        _eegs_root = Path(__file__).resolve().parent.parent.parent
        _viewer_root = _eegs_root.parent
        if str(_viewer_root) not in sys.path:
            sys.path.insert(0, str(_viewer_root))

        from src.models.eeg_encoder import PatchEEGEncoder
        self.encoder = PatchEEGEncoder(
            num_channels=M,
            patch_size=1,       # one token = one "sample"
            embed_dim=embed_dim,
            num_layers=n_layers,
            num_heads=n_heads,
            dropout=dropout,
            sample_rate=1.0,    # 1 token/sec → all filter kernels → size 3
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> PatchEncoderOutput:
        # x: [B, M, S] — passed directly as [B, channels, time]
        pooled = self.encoder(x).pooled   # [B, embed_dim]
        return PatchEncoderOutput(task_vec=pooled, content_vec=pooled)


class _FlatEncoder(nn.Module):
    """Wraps a flat (linear/MLP) encoder to emit PatchEncoderOutput.

    For flat encoders there is no separate content pathway, so task_vec and
    content_vec are identical.  The recon head therefore reads from the same
    representation as the task head — expected to underperform the transformer
    variant, which is the point of the ablation.
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> PatchEncoderOutput:
        z = self.net(x)
        return PatchEncoderOutput(task_vec=z, content_vec=z)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_patch_encoder(
    encoder_type: _EncoderType,
    M: int,
    S: int,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    n_hidden_layers: int = 2,
    n_attn_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """Instantiate a patch encoder by name.

    Args:
        encoder_type:    One of "linear", "mlp", "transformer", "eeg_viewer".
        M:               Number of patches (virtual electrodes).
        S:               Sequence length (token positions, i.e. time).
        embed_dim:       Output embedding dimension (all variants).
        hidden_dim:      MLP hidden width (mlp only).
        n_hidden_layers: MLP depth or transformer encoder layers.
        n_attn_heads:    Transformer self-attention heads (transformer/eeg_viewer).
        dropout:         Dropout rate.

    Returns:
        nn.Module that accepts [B, M, S] and returns PatchEncoderOutput.
        For linear/mlp variants task_vec == content_vec (no dual pathway).
    """
    if encoder_type == "linear":
        return _FlatEncoder(nn.Sequential(
            _Flatten(),
            nn.Linear(M * S, embed_dim),
        ))

    if encoder_type == "mlp":
        return _FlatEncoder(nn.Sequential(
            _Flatten(),
            _make_mlp(
                in_dim=M * S,
                hidden_dim=hidden_dim,
                out_dim=embed_dim,
                n_hidden_layers=max(1, n_hidden_layers),
                dropout=dropout,
            ),
        ))

    if encoder_type == "transformer":
        # Ensure n_attn_heads divides embed_dim
        while embed_dim % n_attn_heads != 0:
            n_attn_heads //= 2
        return PatchTokenEncoder(
            M=M,
            S=S,
            embed_dim=embed_dim,
            n_layers=max(1, n_hidden_layers),
            n_heads=n_attn_heads,
            dropout=dropout,
        )

    if encoder_type == "eeg_viewer":
        while embed_dim % n_attn_heads != 0:
            n_attn_heads //= 2
        return _EEGViewerEncoderWrapper(
            M=M,
            S=S,
            embed_dim=embed_dim,
            n_layers=max(1, n_hidden_layers),
            n_heads=n_attn_heads,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown encoder_type '{encoder_type}'. "
        f"Choose from: linear, mlp, transformer, eeg_viewer"
    )
