"""Project pathway features to EEG-like signals.

GPU-friendly design notes
--------------------------
* The smoothing convolution kernel is a ``registered_buffer`` — allocated
  once at construction, never re-created in ``forward()``.
* Noise injection and smoothing are dispatched to bound methods at
  construction time so ``forward()`` contains no if-statements.
* The ``training`` boolean parameter has been removed; ``self.training``
  (set by ``model.train()`` / ``model.eval()``) is used instead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EEGProjector(nn.Module):

    def __init__(
        self,
        input_dim: int = 6,
        output_channels: int = 105,
        add_noise: bool = True,
        noise_std: float = 0.1,
        smoothing_window: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.noise_std = noise_std

        self.projection = nn.Linear(input_dim, output_channels, bias=True)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Register smoothing kernel as buffer (moves with .to(device) / .cuda())
        if smoothing_window is not None and smoothing_window > 1:
            kernel = torch.ones(output_channels, 1, smoothing_window) / smoothing_window
            self.register_buffer("smoothing_kernel", kernel)
            self._pad = smoothing_window // 2
        else:
            self.register_buffer("smoothing_kernel", None)
            self._pad = 0

        # Resolve noise and smoothing behaviour at construction time
        # so forward() is branch-free.
        self._noise_fn = self._add_noise if add_noise else self._identity
        self._smooth_fn = self._apply_smoothing if (smoothing_window and smoothing_window > 1) else self._identity

    # ------------------------------------------------------------------
    # Component methods — selected once at init, never branched in forward
    # ------------------------------------------------------------------

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise only during training (uses self.training)."""
        if self.training:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def _apply_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel moving-average using the pre-allocated buffer kernel."""
        # x: [B, C] — treat channels as a 1-D temporal axis of length 1
        # Reshape to [B, C, 1] for conv1d with groups=C
        x = x.unsqueeze(-1)                                    # [B, C, 1]
        x = F.pad(x, (self._pad, self._pad))                   # [B, C, 1+2*pad]
        x = F.conv1d(x, self.smoothing_kernel, groups=self.output_channels)
        return x.squeeze(-1)                                   # [B, C]

    # ------------------------------------------------------------------
    # Forward — no if-statements
    # ------------------------------------------------------------------

    def forward(self, pathway_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pathway_features: [B, input_dim]
        Returns:
            [B, output_channels]
        """
        x = self.projection(pathway_features)
        x = self._noise_fn(x)
        x = self._smooth_fn(x)
        return x


def project_to_eeg(
    pathway_features: torch.Tensor,
    projector: EEGProjector,
) -> torch.Tensor:
    return projector(pathway_features)
