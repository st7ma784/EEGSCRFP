"""Real-world EEG noise models for network patch floating scores.

Scientific motivation
---------------------
LLM network patches are the **ideal** analog of EEG electrodes: clean,
deterministic, float32.  Real EEG recordings are contaminated by six
physically distinct noise sources that have no equivalent in LLM activations.

Modelling these sources serves two purposes:

1. **Training augmentation** — inject realistic noise so the encoder learns
   representations robust to the degradation it will encounter in real EEG.

2. **Sensitivity analysis** — systematically degrade the clean LLM signal to
   predict how well the encoder will transfer to real EEG recordings, and
   identify which noise type is the primary obstacle.

The six noise types
-------------------
A. ELECTRODE-LEVEL (independent per channel)

   1. **Gaussian (thermal) noise** — Johnson–Nyquist thermal noise in the
      resistive electrode–scalp contact.  White spectrum; independent across
      channels and time.  Typical scale: σ ≈ 1–5 μV vs signal ≈ 10–100 μV.

   2. **1/f (pink) noise** — intrinsic neural background activity not related
      to the task.  Power spectrum ∝ 1/f^β with β ≈ 1 for cortical field
      potentials.  Autocorrelated along the token/time axis; present even in
      perfectly recorded data.

   3. **Drift** — slow per-channel offset change due to electrode impedance
      drift, perspiration, or temperature gradients.  Modelled as an
      independent linear trend per channel over the time axis.

B. COMMON-MODE (shared across all channels)

   4. **Reference electrode noise** — all EEG channels are measured relative
      to a single reference electrode.  The reference's own noise η(t) is
      subtracted from every channel:

          V_ch(t) = signal_ch(t) − η_ref(t)

      The η_ref term is *identical* for every M channels, so it shifts all
      floating scores by the same amount.  It does NOT cancel in per-channel
      normalisation.  However, **CKA features are invariant to this noise**
      because CKA centres each hidden-state matrix over the sequence/sample
      dimension, which removes the shared additive offset exactly.

C. SPATIAL (correlated across channels)

   5. **Volume conduction** — electrical currents from a neural source spread
      through the skull and scalp.  Each source contributes to multiple
      electrodes with amplitude ∝ 1/r² (dipole model).  Modelled as a 1D
      Gaussian blur over the channel (M) axis.  Reduces the effective rank
      of the [B, M] feature matrix — neighbouring patches become correlated.

D. ARTIFACTUAL (non-stationary, large amplitude)

   6. **Spike artifacts** — eye blinks (~500 μV), EMG bursts (broadband).
      Modelled as a Bernoulli(p_spike) mask selecting positions that receive
      an additive Gaussian impulse with scale >> typical signal amplitude.
      The sparse but extreme nature of spike artifacts is more damaging to
      the calibration baseline than equal-power Gaussian noise.

Shape conventions
-----------------
All functions accept x of shape [B, M] (scalar floating scores) or [B, M, S]
(temporal floating scores where S = token positions = virtual time axis).
Pink noise and drift operate on the last (S) dimension and require dim ≥ 3.
Common-mode and volume conduction operate on the M dimension in both cases.

References
----------
- Makeig et al. (2004) "Mining event-related brain dynamics." TICS.
- Nunez & Srinivasan (2006) "Electric Fields of the Brain." Oxford.
- Delorme & Makeig (2004) "EEGLAB." J Neurosci Methods.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gaussian (thermal) noise
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_noise(
    x: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add i.i.d. Gaussian noise N(0, sigma²) to every element of x.

    Works on any shape.  Noise is independent across channels and time.

    Args:
        x:         Input tensor, any shape.
        sigma:     Standard deviation of the noise.
        generator: Optional RNG for reproducibility.

    Returns:
        Tensor of the same shape and dtype as x.
    """
    if sigma == 0.0:
        return x.clone()
    noise = torch.empty_like(x).normal_(0.0, 1.0, generator=generator)
    return x + sigma * noise


# ─────────────────────────────────────────────────────────────────────────────
# 2. 1/f (pink) noise
# ─────────────────────────────────────────────────────────────────────────────

def pink_noise_1d(
    length: int,
    beta: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate unit-variance 1D pink noise of the given length.

    Uses the Fourier spectral shaping method:
      1. Draw white noise w ~ N(0, 1).
      2. FFT → W(f).
      3. Multiply amplitude by |f|^(-beta/2)  (sets PSD ∝ 1/f^beta).
      4. IFFT → coloured noise, normalised to unit variance.

    For typical EEG background noise use beta=1 (pink) or beta=2 (brown/red).
    For lengths < 4 the spectral method is unreliable — returns white noise.

    Args:
        length:    Number of timepoints.
        beta:      Spectral exponent (0=white, 1=pink, 2=brown).
        dtype:     Output dtype (defaults to float32).
        device:    Output device.
        generator: Optional RNG for reproducibility.

    Returns:
        [length] float tensor with unit variance.
    """
    if dtype is None:
        dtype = torch.float32

    white = torch.empty(length, dtype=dtype, device=device).normal_(
        0.0, 1.0, generator=generator
    )

    if length < 4 or beta == 0.0:
        std = white.std().clamp(min=_EPS)
        return white / std

    X = torch.fft.rfft(white.float())
    freqs = torch.fft.rfftfreq(length, device=device)

    # DC bin → zero (no offset).
    shaping = freqs.clone()
    shaping[0] = 1.0                      # temporary non-zero to avoid log(0)
    shaping = shaping.pow(-beta / 2.0)
    shaping[0] = 0.0                      # remove DC

    X = X * shaping
    pink = torch.fft.irfft(X, n=length).to(dtype)

    std = pink.std().clamp(min=_EPS)
    return pink / std


def pink_noise(
    x: torch.Tensor,
    beta: float = 1.0,
    sigma: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add pink (1/f^beta) noise along the last (time) axis of x.

    Requires x.dim() >= 2.  Each (batch, channel) combination receives an
    independent pink noise realisation scaled to standard deviation sigma.

    For [B, M] (scalar floating scores) use gaussian_noise instead — there
    is no time axis over which the pink correlation can be expressed.

    Args:
        x:         [B, M, S] tensor.  S is the time/token axis.
        beta:      Spectral exponent (0=white, 1=pink, 2=brown).
        sigma:     Noise amplitude.
        generator: Optional RNG.

    Returns:
        [B, M, S] tensor.
    """
    if sigma == 0.0:
        return x.clone()
    if x.dim() < 3:
        raise ValueError(
            "pink_noise requires at least 3 dimensions [B, M, S]; "
            "for [B, M] use gaussian_noise."
        )
    B, M, S = x.shape[0], x.shape[1], x.shape[-1]
    # Generate BM independent noise series, each of length S.
    noise = torch.stack(
        [pink_noise_1d(S, beta=beta, dtype=x.dtype, device=x.device,
                       generator=generator)
         for _ in range(B * M)],
        dim=0,
    ).view(B, M, S)
    return x + sigma * noise


# ─────────────────────────────────────────────────────────────────────────────
# 3. Drift (slow electrode impedance change)
# ─────────────────────────────────────────────────────────────────────────────

def drift_noise(
    x: torch.Tensor,
    max_slope: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add an independent linear drift to each channel.

    Each (batch, channel) pair receives a slope drawn uniformly from
    [-max_slope, +max_slope] multiplied by the normalised time axis [0, 1].
    This models the slow DC offset change seen in real EEG due to electrode
    impedance drift, sweat, or temperature variation.

    Requires x.dim() >= 3 ([B, M, S]).

    Args:
        x:          [B, M, S] tensor.
        max_slope:  Maximum absolute drift in units of signal amplitude.
        generator:  Optional RNG.

    Returns:
        [B, M, S] tensor.
    """
    if max_slope == 0.0:
        return x.clone()
    if x.dim() < 3:
        raise ValueError("drift_noise requires at least 3 dimensions [B, M, S].")
    B, M, S = x.shape[0], x.shape[1], x.shape[-1]
    # Per-channel slopes: [B, M, 1]
    slopes = (
        torch.empty(B, M, 1, dtype=x.dtype, device=x.device)
        .uniform_(-max_slope, max_slope, generator=generator)
    )
    # Normalised time axis: [1, 1, S]
    t = torch.linspace(0.0, 1.0, S, dtype=x.dtype, device=x.device).view(1, 1, S)
    return x + slopes * t


# ─────────────────────────────────────────────────────────────────────────────
# 4. Common-mode (reference electrode) noise
# ─────────────────────────────────────────────────────────────────────────────

def common_mode_noise(
    x: torch.Tensor,
    sigma_ref: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add a single noise realisation shared across ALL channels.

    Models the reference electrode noise in real EEG:
        V_ch(t) = signal_ch(t) − η_ref(t)
    where η_ref is drawn once and subtracted from every channel.

    For [B, M]: one noise scalar per sample b, added to all M channels.
    For [B, M, S]: one noise series [S] per sample b, added to all M channels.

    Key property: inter-channel differences are preserved exactly:
        x_noisy[b, i, t] − x_noisy[b, j, t] == x[b, i, t] − x[b, j, t]

    CKA features are invariant to this noise because the centering step
    (subtracting the mean over the sequence dimension) removes the shared
    additive offset before computing Gram matrices.

    Args:
        x:         [B, M] or [B, M, S] tensor.
        sigma_ref: Standard deviation of the reference noise.
        generator: Optional RNG.

    Returns:
        Same shape as x.
    """
    if sigma_ref == 0.0:
        return x.clone()
    B = x.shape[0]
    if x.dim() == 2:
        # One scalar per sample, broadcast over M.
        eta = torch.empty(B, 1, dtype=x.dtype, device=x.device).normal_(
            0.0, sigma_ref, generator=generator
        )
    else:
        # One time series [S] per sample, broadcast over M.
        S = x.shape[-1]
        eta = torch.empty(B, 1, S, dtype=x.dtype, device=x.device).normal_(
            0.0, sigma_ref, generator=generator
        )
    return x + eta


# ─────────────────────────────────────────────────────────────────────────────
# 5. Volume conduction (spatial blur over channels)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float, device: Optional[torch.device] = None) -> torch.Tensor:
    """1D symmetric Gaussian kernel, normalised to sum to 1."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device).float()
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def volume_conduction(
    x: torch.Tensor,
    sigma_spatial: float,
) -> torch.Tensor:
    """Apply a Gaussian spatial blur over the channel (M) dimension.

    Models electrical volume conduction through the skull: a single neural
    source is detected by multiple nearby electrodes, weighted by 1/distance².
    We approximate this as a 1D Gaussian blur over the ordered patch axis.

    For [B, M]:   blurs over M independently for each sample.
    For [B, M, S]: blurs over M independently for each (sample, time) pair.

    Args:
        x:              [B, M] or [B, M, S] tensor.
        sigma_spatial:  Width of the Gaussian kernel in "patch units".
                        sigma=0 → identity.  sigma→∞ → all channels equal.

    Returns:
        Same shape as x.
    """
    if sigma_spatial <= 0.0:
        return x.clone()

    M = x.shape[1] if x.dim() >= 2 else x.shape[0]
    if M <= 1:
        return x.clone()

    kernel = _gaussian_kernel_1d(sigma_spatial, device=x.device).to(x.dtype)
    pad = kernel.shape[0] // 2
    # Kernel as [out_channels=1, in_channels=1, k] for F.conv1d.
    k = kernel.view(1, 1, -1)

    if x.dim() == 2:
        # [B, M] → [B, 1, M] → conv → [B, 1, M] → [B, M]
        # Operate in float32; cast back to input dtype at the end.
        out = F.conv1d(x.float().unsqueeze(1), k.float(), padding=pad)
        return out.squeeze(1).to(x.dtype)
    else:
        # [B, M, S]: blur M axis for each (b, s) independently.
        B, M_, S = x.shape
        # Permute to [B, S, M] → reshape to [B*S, 1, M].
        xp = x.float().permute(0, 2, 1).reshape(B * S, 1, M_)
        out = F.conv1d(xp, k.float(), padding=pad)           # [B*S, 1, M]
        return out.squeeze(1).reshape(B, S, M_).permute(0, 2, 1).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Spike artifacts (eye blinks, EMG bursts)
# ─────────────────────────────────────────────────────────────────────────────

def spike_artifacts(
    x: torch.Tensor,
    p_spike: float = 0.05,
    spike_scale: float = 10.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add sparse, large-amplitude spike artifacts.

    Models eye-blink (~500 μV) and EMG burst artifacts.  Each (batch, channel,
    time) position is independently affected with probability p_spike.
    Affected positions receive an additive impulse ~ N(0, spike_scale²).

    For [B, M] (no time axis): spikes are per (batch, channel) position.
    For [B, M, S]: spikes are per (batch, channel, time) position.

    The extreme amplitude of spike artifacts is far more damaging to the
    PatchFeatureExtractor calibration baseline than equal-power Gaussian
    noise, because a few extreme outliers can dominate the population mean.

    Args:
        x:           Input tensor [B, M] or [B, M, S].
        p_spike:     Probability of a spike at each position.
        spike_scale: Amplitude (std dev) of spike impulses.
        generator:   Optional RNG.

    Returns:
        Same shape as x.
    """
    if p_spike == 0.0:
        return x.clone()
    mask = torch.empty_like(x).uniform_(generator=generator) < p_spike
    impulse = torch.empty_like(x).normal_(0.0, spike_scale, generator=generator)
    return x + mask.float() * impulse


# ─────────────────────────────────────────────────────────────────────────────
# Configurable augmenter (for training)
# ─────────────────────────────────────────────────────────────────────────────

class EEGNoiseAugmenter(nn.Module):
    """Composable EEG noise augmenter for training.

    Applies a configurable mix of the six noise types to floating-score
    tensors.  Each noise type is applied independently with its configured
    scale; set a scale to 0.0 to disable that type.

    All noise is applied deterministically given a fixed generator seed, so
    the augmenter is reproducible.  In practice, pass generator=None and
    rely on the global RNG to get different noise each forward pass.

    Accepted input shapes: [B, M] or [B, M, S].  Pink noise and drift
    require [B, M, S]; they are silently skipped for [B, M] inputs.

    Example::

        augmenter = EEGNoiseAugmenter(
            gaussian_sigma=0.05,
            pink_sigma=0.03,
            pink_beta=1.0,
            drift_max_slope=0.02,
            common_mode_sigma=0.04,
            volume_conduction_sigma=0.5,
            spike_p=0.01,
            spike_scale=5.0,
        )
        x_noisy = augmenter(x_clean)
    """

    def __init__(
        self,
        gaussian_sigma: float = 0.0,
        pink_sigma: float = 0.0,
        pink_beta: float = 1.0,
        drift_max_slope: float = 0.0,
        common_mode_sigma: float = 0.0,
        volume_conduction_sigma: float = 0.0,
        spike_p: float = 0.0,
        spike_scale: float = 10.0,
    ):
        super().__init__()
        self.gaussian_sigma = gaussian_sigma
        self.pink_sigma = pink_sigma
        self.pink_beta = pink_beta
        self.drift_max_slope = drift_max_slope
        self.common_mode_sigma = common_mode_sigma
        self.volume_conduction_sigma = volume_conduction_sigma
        self.spike_p = spike_p
        self.spike_scale = spike_scale

    def forward(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply all configured noise types in sequence.

        Noise types with scale=0 are no-ops (no allocation, no copy).

        Args:
            x:         [B, M] or [B, M, S] floating-score tensor.
            generator: Optional RNG for reproducibility.

        Returns:
            Noisy tensor with same shape and dtype as x.
        """
        out = x
        if self.gaussian_sigma > 0.0:
            out = gaussian_noise(out, self.gaussian_sigma, generator=generator)
        if self.pink_sigma > 0.0 and x.dim() >= 3:
            out = pink_noise(out, beta=self.pink_beta, sigma=self.pink_sigma,
                             generator=generator)
        if self.drift_max_slope > 0.0 and x.dim() >= 3:
            out = drift_noise(out, max_slope=self.drift_max_slope, generator=generator)
        if self.common_mode_sigma > 0.0:
            out = common_mode_noise(out, self.common_mode_sigma, generator=generator)
        if self.volume_conduction_sigma > 0.0:
            out = volume_conduction(out, self.volume_conduction_sigma)
        if self.spike_p > 0.0:
            out = spike_artifacts(out, self.spike_p, self.spike_scale, generator=generator)
        return out

    def noise_config(self) -> Dict[str, float]:
        """Return the current noise parameters as a flat dict."""
        return {
            "gaussian_sigma": self.gaussian_sigma,
            "pink_sigma": self.pink_sigma,
            "pink_beta": self.pink_beta,
            "drift_max_slope": self.drift_max_slope,
            "common_mode_sigma": self.common_mode_sigma,
            "volume_conduction_sigma": self.volume_conduction_sigma,
            "spike_p": self.spike_p,
            "spike_scale": self.spike_scale,
        }
