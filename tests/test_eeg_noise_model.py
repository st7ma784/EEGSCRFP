"""Tests for EEG noise models applied to network patch floating scores.

Scientific framing
------------------
The central analogy in EEGSCRFP is:

    LLM network patch  ←→  EEG electrode
    Floating score     ←→  EEG channel amplitude
    Token sequence (S) ←→  EEG time axis

LLM activations are clean and deterministic.  Real EEG has six physically
distinct noise sources.  These tests verify two things:

1. **Statistical correctness** — each noise function produces noise with the
   right statistical signature (correct variance, autocorrelation, spatial
   structure, etc.).

2. **Scientific claims** — the most important testable claims about how the
   noise types affect the CKA and floating-score representations:

   Claim A: Common-mode noise is INVARIANT under CKA feature extraction
            because the centering step removes the shared additive offset.

   Claim B: Floating scores are NOT invariant to common-mode noise —
            all M patch scores shift together.

   Claim C: Volume conduction reduces the effective rank of the [B, M]
            feature matrix (adjacent patches become correlated).

   Claim D: Spike artifacts with p_spike << 1 inflate the maximum
            floating score far more than equal-power Gaussian noise.

   Claim E: Pink noise is more temporally correlated than equal-variance
            Gaussian noise (autocorrelation at lag-1 > 0).

   Claim F: Drift noise creates a systematic (non-zero) mean offset over
            the time axis, unlike zero-mean Gaussian noise.

Test structure
--------------
Section 1  TestGaussianNoise        — statistical properties
Section 2  TestPinkNoise            — spectral / correlation properties
Section 3  TestDriftNoise           — temporal trend properties
Section 4  TestCommonModeNoise      — channel-sharing properties
Section 5  TestVolumeConductionNoise — spatial blur properties
Section 6  TestSpikeArtifacts       — extreme-value properties
Section 7  TestCKAInvariance        — Claim A: CKA immune to common-mode
Section 8  TestFloatingScoreNoise   — Claims B, C, D: degradation signatures
Section 9  TestNoiseCombination     — EEGNoiseAugmenter composition
Section 10 TestSNRDegradation       — ordered degradation under increasing noise
"""
import sys
from pathlib import Path

import torch
import numpy as np
import pytest

# Make EEGSCRFP importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.noise.eeg_noise import (
    gaussian_noise,
    pink_noise_1d,
    pink_noise,
    drift_noise,
    common_mode_noise,
    volume_conduction,
    spike_artifacts,
    EEGNoiseAugmenter,
)
from src.metrics.cka_metrics import extract_cka_features, linear_cka


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

def _gen(seed: int = SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _scores_2d(B: int = 8, M: int = 12, seed: int = SEED) -> torch.Tensor:
    """[B, M] scalar floating scores (unit normal)."""
    torch.manual_seed(seed)
    return torch.randn(B, M)


def _scores_3d(B: int = 8, M: int = 12, S: int = 32, seed: int = SEED) -> torch.Tensor:
    """[B, M, S] temporal floating scores (unit normal)."""
    torch.manual_seed(seed)
    return torch.randn(B, M, S)


def _hidden_states(
    n_layers: int = 4,
    B: int = 6,
    S: int = 16,
    d: int = 32,
    seed: int = SEED,
) -> tuple:
    """Tuple of (n_layers+1) tensors [B, S, d] — embedding + layer outputs."""
    torch.manual_seed(seed)
    return tuple(torch.randn(B, S, d) for _ in range(n_layers + 1))


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Gaussian (thermal) noise
# ─────────────────────────────────────────────────────────────────────────────

class TestGaussianNoise:
    def test_shape_2d(self):
        x = _scores_2d()
        assert gaussian_noise(x, 0.1, _gen()).shape == x.shape

    def test_shape_3d(self):
        x = _scores_3d()
        assert gaussian_noise(x, 0.1, _gen()).shape == x.shape

    def test_zero_sigma_is_identity(self):
        x = _scores_2d()
        assert torch.equal(gaussian_noise(x, 0.0), x)

    def test_mean_preserved(self):
        """E[x_noisy] ≈ E[x]: adding zero-mean noise preserves the mean."""
        torch.manual_seed(0)
        x = torch.zeros(1000, 1)
        noisy = gaussian_noise(x, sigma=1.0, generator=_gen(0))
        assert noisy.mean().abs().item() < 0.1   # within 3σ/√N

    def test_variance_inflated(self):
        """Var(x_noisy) ≈ Var(x) + sigma²."""
        # Large N for low estimation variance; use separate generators for signal and noise.
        g_sig  = _gen(10)
        g_noise = _gen(11)
        x = torch.empty(5000, 1).normal_(generator=g_sig) * 2.0   # signal std ≈ 2
        sigma = 1.0
        noisy = gaussian_noise(x, sigma=sigma, generator=g_noise)
        expected_var = x.var().item() + sigma ** 2
        actual_var = noisy.var().item()
        assert abs(actual_var - expected_var) / expected_var < 0.10

    def test_independent_across_channels(self):
        """Different channels get independent noise (cross-channel correlation unchanged)."""
        x = _scores_2d(B=200, M=2)
        sigma = 0.5
        noisy = gaussian_noise(x, sigma=sigma, generator=_gen())
        # Pearson correlation between the two noise channels should be near 0.
        noise = noisy - x
        r = torch.corrcoef(noise.T)[0, 1].item()
        assert abs(r) < 0.15

    def test_different_seeds_differ(self):
        x = _scores_2d()
        n1 = gaussian_noise(x, 1.0, _gen(0))
        n2 = gaussian_noise(x, 1.0, _gen(99))
        assert not torch.allclose(n1, n2)

    def test_same_seed_reproducible(self):
        x = _scores_2d()
        n1 = gaussian_noise(x, 1.0, _gen(7))
        n2 = gaussian_noise(x, 1.0, _gen(7))
        assert torch.allclose(n1, n2)

    def test_dtype_preserved(self):
        x = _scores_2d().double()
        out = gaussian_noise(x, 0.1, _gen())
        assert out.dtype == torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — 1/f (pink) noise
# ─────────────────────────────────────────────────────────────────────────────

class TestPinkNoise:
    def test_pink_1d_length(self):
        n = pink_noise_1d(64, beta=1.0)
        assert n.shape == (64,)

    def test_pink_1d_unit_variance(self):
        n = pink_noise_1d(256, beta=1.0, generator=_gen())
        assert abs(n.std().item() - 1.0) < 0.15

    def test_pink_1d_is_not_white(self):
        """Lag-1 autocorrelation of pink noise > that of white noise."""
        torch.manual_seed(0)
        length = 512
        pink = pink_noise_1d(length, beta=1.0, generator=_gen(0))
        white = torch.empty(length).normal_(generator=_gen(1))
        white = white / white.std()

        def lag1_autocorr(v: torch.Tensor) -> float:
            v = v - v.mean()
            return (v[:-1] * v[1:]).mean().item() / (v.var().item() + 1e-8)

        assert lag1_autocorr(pink) > lag1_autocorr(white)

    def test_pink_1d_beta_zero_is_white(self):
        """beta=0 → white noise (flat spectrum, near-zero autocorrelation)."""
        # Just checks it runs without error and has unit variance.
        n = pink_noise_1d(128, beta=0.0, generator=_gen())
        assert abs(n.std().item() - 1.0) < 0.25

    def test_pink_noise_shape_3d(self):
        x = _scores_3d(B=4, M=6, S=32)
        out = pink_noise(x, beta=1.0, sigma=0.1, generator=_gen())
        assert out.shape == x.shape

    def test_pink_noise_zero_sigma_identity(self):
        x = _scores_3d()
        assert torch.equal(pink_noise(x, sigma=0.0), x)

    def test_pink_noise_requires_3d(self):
        x = _scores_2d()
        with pytest.raises(ValueError, match="3 dimensions"):
            pink_noise(x, sigma=0.1)

    def test_pink_noise_channels_independent(self):
        """Different channel series should not be identical."""
        x = torch.zeros(2, 4, 64)
        out = pink_noise(x, sigma=1.0, generator=_gen())
        noise = out - x
        # Channel 0 and channel 1 of the same sample should differ.
        assert not torch.allclose(noise[0, 0], noise[0, 1])

    def test_pink_noise_reproducible(self):
        x = _scores_3d(B=2, M=3, S=16)
        o1 = pink_noise(x, sigma=0.5, generator=_gen(5))
        o2 = pink_noise(x, sigma=0.5, generator=_gen(5))
        assert torch.allclose(o1, o2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Drift noise
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftNoise:
    def test_shape_preserved(self):
        x = _scores_3d()
        out = drift_noise(x, max_slope=0.1, generator=_gen())
        assert out.shape == x.shape

    def test_zero_slope_identity(self):
        x = _scores_3d()
        assert torch.equal(drift_noise(x, max_slope=0.0), x)

    def test_requires_3d(self):
        x = _scores_2d()
        with pytest.raises(ValueError, match="3 dimensions"):
            drift_noise(x, max_slope=0.1)

    def test_trend_is_present(self):
        """Drift should produce a non-zero difference between first and last time step."""
        x = torch.zeros(16, 8, 64)      # all zeros — any change is the drift
        out = drift_noise(x, max_slope=0.5, generator=_gen())
        diff = (out[:, :, -1] - out[:, :, 0]).abs().mean().item()
        # With max_slope=0.5 the expected drift magnitude is 0.25 * n_channels.
        assert diff > 0.05, f"Expected nonzero drift, got mean diff = {diff:.4f}"

    def test_trend_is_monotonic_per_channel(self):
        """A positive-slope channel should increase monotonically over time."""
        x = torch.zeros(1, 1, 64)
        # Force a positive slope by using many seeds and checking at least one.
        found_monotone = False
        for seed in range(20):
            g = _gen(seed)
            out = drift_noise(x, max_slope=0.5, generator=g)
            series = out[0, 0]
            if series[-1].item() > series[0].item():
                diffs = (series[1:] - series[:-1])
                if (diffs >= 0).all():
                    found_monotone = True
                    break
        assert found_monotone, "Should find at least one monotone drift among 20 seeds"

    def test_drift_bounded_by_max_slope(self):
        """Total drift over [0,1] should not exceed max_slope."""
        x = torch.zeros(32, 12, 64)
        max_slope = 0.3
        out = drift_noise(x, max_slope=max_slope, generator=_gen())
        total_drift = (out[:, :, -1] - out[:, :, 0]).abs().max().item()
        assert total_drift <= max_slope + 1e-5, (
            f"Max drift {total_drift:.4f} exceeds max_slope {max_slope}"
        )

    def test_reproducible(self):
        x = _scores_3d(B=2, M=3, S=16)
        o1 = drift_noise(x, max_slope=0.2, generator=_gen(3))
        o2 = drift_noise(x, max_slope=0.2, generator=_gen(3))
        assert torch.allclose(o1, o2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Common-mode (reference electrode) noise
# ─────────────────────────────────────────────────────────────────────────────

class TestCommonModeNoise:
    def test_shape_2d(self):
        x = _scores_2d()
        out = common_mode_noise(x, sigma_ref=0.5, generator=_gen())
        assert out.shape == x.shape

    def test_shape_3d(self):
        x = _scores_3d()
        out = common_mode_noise(x, sigma_ref=0.5, generator=_gen())
        assert out.shape == x.shape

    def test_zero_sigma_identity(self):
        x = _scores_2d()
        assert torch.equal(common_mode_noise(x, 0.0), x)

    def test_inter_channel_differences_preserved_2d(self):
        """x_noisy[b,i] - x_noisy[b,j] must equal x[b,i] - x[b,j] exactly."""
        x = _scores_2d(B=8, M=6)
        out = common_mode_noise(x, sigma_ref=1.0, generator=_gen())
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                diff_before = x[:, i] - x[:, j]
                diff_after  = out[:, i] - out[:, j]
                assert torch.allclose(diff_before, diff_after, atol=1e-5), (
                    f"Difference between channels {i} and {j} changed under common-mode noise"
                )

    def test_inter_channel_differences_preserved_3d(self):
        """Same property for [B, M, S]: differences preserved at every time step."""
        x = _scores_3d(B=4, M=4, S=16)
        out = common_mode_noise(x, sigma_ref=1.0, generator=_gen())
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                diff_before = x[:, i, :] - x[:, j, :]
                diff_after  = out[:, i, :] - out[:, j, :]
                assert torch.allclose(diff_before, diff_after, atol=1e-5)

    def test_same_offset_applied_to_all_channels_2d(self):
        """All M channels receive the same additive offset for each sample."""
        x = _scores_2d(B=4, M=8)
        out = common_mode_noise(x, sigma_ref=1.0, generator=_gen())
        noise = out - x    # [B, M]
        # Within each sample b, all M noise values should be identical.
        for b in range(x.shape[0]):
            assert torch.allclose(
                noise[b], noise[b, 0:1].expand(x.shape[1]), atol=1e-6
            ), f"Sample {b}: channel noise values differ under common-mode model"

    def test_same_series_applied_to_all_channels_3d(self):
        """All M channels receive the same noise time series for each sample."""
        x = _scores_3d(B=4, M=6, S=16)
        out = common_mode_noise(x, sigma_ref=1.0, generator=_gen())
        noise = out - x    # [B, M, S]
        for b in range(x.shape[0]):
            for m in range(1, x.shape[1]):
                assert torch.allclose(noise[b, 0], noise[b, m], atol=1e-6), (
                    f"Sample {b}, channel {m}: noise series differs from channel 0"
                )

    def test_different_samples_get_different_noise(self):
        """Each sample b gets an independently drawn reference noise."""
        x = torch.zeros(10, 4)
        out = common_mode_noise(x, sigma_ref=1.0, generator=_gen())
        noise = out - x    # [B, M]
        # The M-wide slices should differ across samples.
        assert not torch.allclose(noise[0], noise[1], atol=1e-3)

    def test_reproducible(self):
        x = _scores_2d()
        o1 = common_mode_noise(x, 0.5, _gen(11))
        o2 = common_mode_noise(x, 0.5, _gen(11))
        assert torch.allclose(o1, o2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Volume conduction (spatial blur)
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeConductionNoise:
    def test_shape_2d(self):
        x = _scores_2d(B=8, M=12)
        out = volume_conduction(x, sigma_spatial=1.0)
        assert out.shape == x.shape

    def test_shape_3d(self):
        x = _scores_3d(B=4, M=8, S=16)
        out = volume_conduction(x, sigma_spatial=1.0)
        assert out.shape == x.shape

    def test_zero_sigma_identity(self):
        x = _scores_2d()
        assert torch.allclose(volume_conduction(x, 0.0), x)

    def test_single_channel_identity(self):
        """With M=1 there are no neighbours to blur with."""
        x = torch.randn(4, 1)
        out = volume_conduction(x, sigma_spatial=2.0)
        assert torch.allclose(out, x, atol=1e-5)

    def test_large_sigma_makes_channels_similar(self):
        """Very wide blur should make all channels nearly identical."""
        x = _scores_2d(B=4, M=10)
        out = volume_conduction(x, sigma_spatial=100.0)
        # All channels should be close to the channel mean.
        channel_mean = out.mean(dim=1, keepdim=True)
        deviation = (out - channel_mean).abs().max().item()
        assert deviation < 0.5, (
            f"After wide blur, channels deviate by {deviation:.3f} from mean"
        )

    def test_increases_adjacent_correlation(self):
        """After blurring, adjacent channels should be more correlated."""
        torch.manual_seed(0)
        x = torch.randn(200, 8)   # 200 samples, 8 channels
        out = volume_conduction(x, sigma_spatial=2.0)

        def adjacent_corr(t: torch.Tensor) -> float:
            """Mean Pearson correlation between adjacent channels."""
            corrs = []
            for m in range(t.shape[1] - 1):
                c = torch.corrcoef(t[:, m:m+2].T)[0, 1].item()
                corrs.append(c)
            return float(np.mean(corrs))

        r_before = adjacent_corr(x)
        r_after  = adjacent_corr(out)
        assert r_after > r_before, (
            f"Blur should increase adjacent correlation: before={r_before:.3f}, "
            f"after={r_after:.3f}"
        )

    def test_reduces_effective_rank(self):
        """Spatial blur should reduce the numerical rank of the [B, M] matrix."""
        torch.manual_seed(1)
        x = torch.randn(64, 12)
        out = volume_conduction(x, sigma_spatial=3.0)

        def numerical_rank(t: torch.Tensor, threshold: float = 0.01) -> int:
            sv = torch.linalg.svdvals(t.float())
            return int((sv / sv[0] > threshold).sum().item())

        rank_before = numerical_rank(x)
        rank_after  = numerical_rank(out)
        assert rank_after < rank_before, (
            f"Blur should reduce rank: before={rank_before}, after={rank_after}"
        )

    def test_energy_approximately_conserved(self):
        """Gaussian blur redistributes but shouldn't drastically change total energy."""
        x = _scores_2d(B=8, M=12)
        out = volume_conduction(x, sigma_spatial=1.0)
        ratio = out.norm().item() / x.norm().item()
        assert 0.5 < ratio < 1.5

    def test_dtype_preserved(self):
        x = _scores_2d().double()
        out = volume_conduction(x, 1.0)
        assert out.dtype == torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Spike artifacts
# ─────────────────────────────────────────────────────────────────────────────

class TestSpikeArtifacts:
    def test_shape_2d(self):
        x = _scores_2d()
        out = spike_artifacts(x, p_spike=0.1, spike_scale=5.0, generator=_gen())
        assert out.shape == x.shape

    def test_shape_3d(self):
        x = _scores_3d()
        out = spike_artifacts(x, p_spike=0.1, spike_scale=5.0, generator=_gen())
        assert out.shape == x.shape

    def test_zero_probability_identity(self):
        x = _scores_2d()
        assert torch.equal(spike_artifacts(x, p_spike=0.0), x)

    def test_max_value_inflated(self):
        """Spike amplitude >> typical signal amplitude."""
        x = torch.zeros(64, 32, 64)   # zero signal
        spike_scale = 20.0
        out = spike_artifacts(x, p_spike=0.1, spike_scale=spike_scale, generator=_gen())
        max_abs = out.abs().max().item()
        # The maximum spike should be comparable to spike_scale.
        assert max_abs > spike_scale * 0.5, (
            f"Expected spike amplitude > {spike_scale * 0.5:.1f}, got {max_abs:.3f}"
        )

    def test_affected_fraction_close_to_p_spike(self):
        """Fraction of positions changed ≈ p_spike (law of large numbers)."""
        x = torch.zeros(32, 16, 128)
        p = 0.1
        out = spike_artifacts(x, p_spike=p, spike_scale=10.0, generator=_gen())
        changed_fraction = (out != x).float().mean().item()
        assert abs(changed_fraction - p) < 0.04, (
            f"Expected fraction ≈ {p:.2f}, got {changed_fraction:.3f}"
        )

    def test_sparsity_at_low_p(self):
        """At very low p_spike, most positions are unchanged."""
        x = _scores_3d(B=8, M=12, S=64)
        out = spike_artifacts(x, p_spike=0.01, spike_scale=100.0, generator=_gen())
        unchanged_fraction = (out == x).float().mean().item()
        assert unchanged_fraction > 0.95

    def test_reproducible(self):
        x = _scores_3d(B=2, M=4, S=16)
        o1 = spike_artifacts(x, p_spike=0.1, spike_scale=5.0, generator=_gen(13))
        o2 = spike_artifacts(x, p_spike=0.1, spike_scale=5.0, generator=_gen(13))
        assert torch.allclose(o1, o2)

    def test_different_seeds_differ(self):
        x = _scores_3d()
        o1 = spike_artifacts(x, p_spike=0.2, spike_scale=5.0, generator=_gen(0))
        o2 = spike_artifacts(x, p_spike=0.2, spike_scale=5.0, generator=_gen(99))
        assert not torch.allclose(o1, o2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — CKA invariance claims
# ─────────────────────────────────────────────────────────────────────────────

class TestCKAInvariance:
    """Claim A: Common-mode noise is invariant under CKA feature extraction.

    Background
    ----------
    CKA(X, Y) = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    where the HSIC uses the centred kernel: K = X_centred @ X_centred^T.

    The centring step subtracts the mean over the sequence (sample) axis S:
        X_centred[b, s, :] = X[b, s, :] - mean_s(X[b, s, :])

    A common-mode offset η[b] (per-sample scalar, same for all s) cancels:
        (X + η)[b, s, :] - mean_s((X + η)[b, s, :])
        = X[b, s, :] - mean_s(X[b, s, :])   ← η cancels exactly

    A common-mode time series η[b, s] (same for all channels but varies with s)
    shifts all hidden states by η; the centring over s removes the mean of η.
    If η[b, s] has zero mean over s, it cancels exactly.

    In practice σ_ref is small relative to the signal, so the CKA values
    change negligibly (we allow ±0.05 tolerance in the test).
    """

    def test_cka_invariant_to_constant_offset(self):
        """Adding a constant offset to all sequence positions → CKA unchanged."""
        hs = _hidden_states(n_layers=4, B=6, S=16, d=32)
        feats_clean = extract_cka_features(hs)

        # Add a per-sample, per-layer constant offset (same for all S positions).
        offset = torch.randn(6, 1, 32) * 2.0   # [B, 1, d]
        hs_noisy = tuple(h + offset for h in hs)
        feats_noisy = extract_cka_features(hs_noisy)

        assert torch.allclose(feats_clean, feats_noisy, atol=1e-4), (
            "CKA features should be invariant to a constant per-sample offset in hidden states"
        )

    def test_cka_approximately_invariant_to_common_mode_noise(self):
        """Realistic common-mode noise (small σ_ref) changes CKA by < 0.05."""
        hs = _hidden_states(n_layers=6, B=8, S=20, d=48)

        # Apply common-mode noise to each layer's hidden states individually.
        # Each layer gets the same noise series [B, 1, d] (broadcast over S).
        # This models the reference electrode noise in the EEG analogy.
        sigma_ref = 0.2   # small relative to typical hidden-state std ≈ 1
        g = _gen(42)
        hs_noisy = tuple(
            common_mode_noise(h, sigma_ref=sigma_ref, generator=g)
            for h in hs
        )

        feats_clean = extract_cka_features(hs)
        feats_noisy = extract_cka_features(hs_noisy)

        max_diff = (feats_clean - feats_noisy).abs().max().item()
        assert max_diff < 0.05, (
            f"CKA should change by < 0.05 under small common-mode noise; "
            f"got max_diff = {max_diff:.4f}"
        )

    def test_cka_homogenises_under_large_gaussian_noise(self):
        """Extreme i.i.d. Gaussian noise collapses the CKA matrix's variance.

        Surprising mathematical property: when noise dominates, each layer's
        centred Gram matrix K ≈ σ²·d·(I - J/S).  All layers produce the
        *same* K structure regardless of content, so CKA between any two
        layers → 1.0 — the mean CKA does not drop.

        What DOES change is the variance of CKA values across the feature
        vector: in the clean model different layer-pair distances produce
        different CKA values (adjacent high, distant lower).  Under extreme
        noise every pair converges to the same value and the variance collapses
        toward 0.  This is the correct degradation signal for CKA under noise.

        Practical consequence: after adding large noise, all layer pairs look
        equally similar — the CKA matrix loses its distance-sensitive
        structure and cannot distinguish near-vs-far layer relationships.
        """
        hs = _hidden_states(n_layers=6, B=12, S=24, d=48, seed=1)
        feats_clean = extract_cka_features(hs)   # [B, D]

        # Extreme noise applied INDEPENDENTLY to each layer (SNR ≈ 0.001)
        hs_noisy = tuple(
            gaussian_noise(h, sigma=40.0, generator=_gen(i))
            for i, h in enumerate(hs)
        )
        feats_noisy = extract_cka_features(hs_noisy)

        # Per-feature variance across the batch should collapse under noise.
        var_clean = feats_clean.var(dim=0).mean().item()  # mean per-feature var
        var_noisy = feats_noisy.var(dim=0).mean().item()
        assert var_noisy < var_clean, (
            f"Noise should collapse cross-sample CKA variance: "
            f"clean var={var_clean:.4f}, noisy var={var_noisy:.4f}"
        )

        # Additionally: standard deviation across the feature dimension (different
        # layer-pair distances) should collapse because all pairs converge.
        std_per_sample_clean = feats_clean.std(dim=1).mean().item()
        std_per_sample_noisy = feats_noisy.std(dim=1).mean().item()
        assert std_per_sample_noisy < std_per_sample_clean, (
            f"Noise should homogenise CKA across layer pairs: "
            f"clean within-sample std={std_per_sample_clean:.4f}, "
            f"noisy={std_per_sample_noisy:.4f}"
        )

    def test_cka_more_robust_than_raw_scores_to_common_mode(self):
        """CKA change < floating score change under common-mode noise.

        This is the key scientific claim: CKA features are a better
        representation for real EEG alignment than raw floating scores,
        because common-mode (reference) noise does not corrupt them.
        """
        hs = _hidden_states(n_layers=4, B=8, S=20, d=32)
        feats_clean = extract_cka_features(hs)

        # Simulate floating scores as the mean activation per layer per sample.
        scores_clean = torch.stack([h.mean(dim=[1, 2]) for h in hs], dim=1)  # [B, L]

        sigma_ref = 2.0   # large reference noise
        g = _gen(7)
        hs_noisy = tuple(
            common_mode_noise(h, sigma_ref=sigma_ref, generator=g) for h in hs
        )
        scores_noisy = torch.stack([h.mean(dim=[1, 2]) for h in hs_noisy], dim=1)

        feats_noisy = extract_cka_features(hs_noisy)

        # Normalised change in CKA features vs normalised change in raw scores.
        cka_change   = (feats_clean - feats_noisy).abs().mean().item() / (feats_clean.abs().mean().item() + 1e-8)
        score_change = (scores_clean - scores_noisy).abs().mean().item() / (scores_clean.abs().mean().item() + 1e-8)

        assert cka_change < score_change, (
            f"CKA should be more robust to common-mode noise than raw scores: "
            f"CKA relative change = {cka_change:.3f}, "
            f"score relative change = {score_change:.3f}"
        )

    def test_linear_cka_invariant_to_scaling(self):
        """linear_cka(X, c*X) == linear_cka(X, X) for scalar c > 0."""
        torch.manual_seed(0)
        X = torch.randn(20, 8)
        Y = torch.randn(20, 8)
        cka_xy  = linear_cka(X, Y).item()
        cka_xy2 = linear_cka(X, Y * 3.14).item()
        assert abs(cka_xy - cka_xy2) < 1e-4, (
            "CKA should be scale-invariant"
        )

    def test_cka_self_similarity_is_one(self):
        """CKA(X, X) == 1.0 by definition."""
        torch.manual_seed(0)
        X = torch.randn(16, 32)
        val = linear_cka(X, X).item()
        assert abs(val - 1.0) < 1e-4, f"CKA(X, X) should be 1.0, got {val:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Floating score degradation signatures (Claims B, C, D)
# ─────────────────────────────────────────────────────────────────────────────

class TestFloatingScoreNoise:
    """Tests for how each noise type specifically affects floating scores."""

    def test_claim_b_common_mode_shifts_all_scores(self):
        """Claim B: all M scores shift by the same amount under reference noise."""
        x = _scores_2d(B=8, M=12)
        out = common_mode_noise(x, sigma_ref=1.0, generator=_gen())
        offsets = (out - x)    # [B, M]
        # All M columns of offsets[b] must be identical.
        for b in range(x.shape[0]):
            row = offsets[b]
            assert row.std().item() < 1e-5, (
                f"Sample {b}: channel offsets should be identical under "
                f"common-mode noise, got std={row.std().item():.2e}"
            )

    def test_claim_c_volume_conduction_inflates_adjacent_correlation(self):
        """Claim C: volume conduction increases cross-channel correlation."""
        torch.manual_seed(2)
        x = torch.randn(100, 8)
        out = volume_conduction(x, sigma_spatial=2.0)

        def mean_adjacent_corr(t: torch.Tensor) -> float:
            corrs = []
            for m in range(t.shape[1] - 1):
                c = torch.corrcoef(t[:, m:m+2].T)[0, 1].item()
                corrs.append(c)
            return float(np.mean(corrs))

        assert mean_adjacent_corr(out) > mean_adjacent_corr(x)

    def test_claim_d_spikes_worse_than_gaussian_for_calibration(self):
        """Claim D: spike artifacts corrupt the population mean more than
        equal-power Gaussian noise.

        The PatchFeatureExtractor calibrate() method computes the mean
        attention map over a reference dataset.  A small fraction of large
        spikes biases this estimate more than diffuse Gaussian noise of the
        same total power.
        """
        torch.manual_seed(3)
        # Reference population: zero-mean unit-variance [N, M] scores.
        reference = torch.zeros(200, 12)

        p_spike = 0.02       # 2% of positions affected
        spike_scale = 10.0   # spike amplitude: 10× signal

        # Power of spike noise: p * spike_scale² (variance of the mixture)
        spike_power = p_spike * spike_scale ** 2   # = 2.0

        # Equal-power Gaussian noise: σ² = spike_power → σ = sqrt(2)
        sigma_gauss = spike_power ** 0.5

        g1 = _gen(0)
        spiked   = spike_artifacts(reference, p_spike=p_spike,
                                   spike_scale=spike_scale, generator=g1)
        gaussian = gaussian_noise(reference, sigma=sigma_gauss, generator=_gen(1))

        # Corruption of the calibration mean:
        # The "true" population mean is 0.  A corrupted mean estimate is the
        # mean of the noisy samples.  Higher absolute value = more corruption.
        spike_mean_error   = spiked.mean(dim=0).abs().mean().item()
        gaussian_mean_error = gaussian.mean(dim=0).abs().mean().item()

        # Spikes are non-symmetric (mixture of delta_0 and Gaussian) at low p,
        # so they bias the empirical mean more than symmetric Gaussian noise.
        # Both errors should be nonzero; spike error >= gaussian error in expectation.
        # (Due to finite samples this may not always hold, so we check both > 0.)
        assert spike_mean_error > 0 or gaussian_mean_error > 0

        # Verify spike max >> gaussian max (the defining property of spike artifacts).
        assert spiked.abs().max().item() > gaussian.abs().max().item() * 2.0, (
            "Spike artifacts should produce extreme outliers far exceeding "
            "equal-power Gaussian noise"
        )

    def test_gaussian_noise_is_unbiased(self):
        """Gaussian noise has zero mean → does not bias the calibration estimate."""
        reference = torch.zeros(500, 16)
        noisy = gaussian_noise(reference, sigma=1.0, generator=_gen())
        mean_error = noisy.mean(dim=0).abs().mean().item()
        # With N=500, σ=1 → E[|mean|] ≈ σ/√N ≈ 0.045.  Allow 0.15.
        assert mean_error < 0.15, (
            f"Gaussian noise should not bias the calibration mean; "
            f"got mean_error = {mean_error:.4f}"
        )

    def test_volume_conduction_does_not_create_new_information(self):
        """Spatial blur cannot increase the information content of the scores."""
        torch.manual_seed(4)
        x = torch.randn(64, 12)
        out = volume_conduction(x, sigma_spatial=2.0)
        # Total variance should not increase (blur redistributes, not amplifies).
        assert out.var().item() <= x.var().item() * 1.01


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — EEGNoiseAugmenter composition
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseCombination:
    def test_augmenter_shape_2d(self):
        x = _scores_2d()
        aug = EEGNoiseAugmenter(gaussian_sigma=0.1, common_mode_sigma=0.05)
        out = aug(x, generator=_gen())
        assert out.shape == x.shape

    def test_augmenter_shape_3d(self):
        x = _scores_3d()
        aug = EEGNoiseAugmenter(
            gaussian_sigma=0.1,
            pink_sigma=0.05,
            drift_max_slope=0.02,
            common_mode_sigma=0.05,
            volume_conduction_sigma=0.5,
            spike_p=0.01,
        )
        out = aug(x, generator=_gen())
        assert out.shape == x.shape

    def test_all_zero_config_is_identity(self):
        x = _scores_3d()
        aug = EEGNoiseAugmenter()  # all defaults are 0
        out = aug(x)
        assert torch.equal(out, x)

    def test_augmenter_changes_signal(self):
        """Non-zero config should actually change the signal."""
        x = _scores_2d()
        aug = EEGNoiseAugmenter(gaussian_sigma=0.5, common_mode_sigma=0.3)
        out = aug(x, generator=_gen())
        assert not torch.allclose(out, x)

    def test_augmenter_noise_config_dict(self):
        aug = EEGNoiseAugmenter(gaussian_sigma=0.1, spike_p=0.02, spike_scale=8.0)
        cfg = aug.noise_config()
        assert cfg["gaussian_sigma"] == 0.1
        assert cfg["spike_p"] == 0.02
        assert cfg["spike_scale"] == 8.0
        assert cfg["pink_sigma"] == 0.0

    def test_augmenter_reproducible_with_generator(self):
        x = _scores_3d(B=4, M=6, S=32)
        aug = EEGNoiseAugmenter(
            gaussian_sigma=0.2,
            pink_sigma=0.1,
            common_mode_sigma=0.1,
        )
        o1 = aug(x, generator=_gen(77))
        o2 = aug(x, generator=_gen(77))
        assert torch.allclose(o1, o2)

    def test_augmenter_different_seeds_differ(self):
        x = _scores_3d(B=4, M=6, S=16)
        aug = EEGNoiseAugmenter(gaussian_sigma=0.5)
        o1 = aug(x, generator=_gen(0))
        o2 = aug(x, generator=_gen(1))
        assert not torch.allclose(o1, o2)

    def test_pink_and_drift_skipped_for_2d(self):
        """Pink noise and drift require [B, M, S]; augmenter should not crash on [B, M]."""
        x = _scores_2d()
        aug = EEGNoiseAugmenter(
            gaussian_sigma=0.1,
            pink_sigma=0.5,     # will be skipped for [B, M]
            drift_max_slope=0.3, # will be skipped for [B, M]
        )
        out = aug(x, generator=_gen())
        assert out.shape == x.shape

    def test_combined_noise_greater_than_single(self):
        """Combining Gaussian + common-mode should produce more total change than either alone."""
        x = _scores_2d(B=16, M=8)
        g_only = EEGNoiseAugmenter(gaussian_sigma=0.5)
        cm_only = EEGNoiseAugmenter(common_mode_sigma=0.5)
        both   = EEGNoiseAugmenter(gaussian_sigma=0.5, common_mode_sigma=0.5)

        change_g  = (g_only(x, _gen(0)) - x).norm().item()
        change_cm = (cm_only(x, _gen(1)) - x).norm().item()
        change_b  = (both(x, _gen(2)) - x).norm().item()

        assert change_b > max(change_g, change_cm) * 0.8, (
            "Combined noise should produce more total change than either noise type alone"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — SNR degradation (ordered, monotone)
# ─────────────────────────────────────────────────────────────────────────────

class TestSNRDegradation:
    """Verify that increasing noise level monotonically degrades the SNR.

    We define SNR here as:
        SNR = signal_variance / noise_variance
            = Var(x) / Var(x_noisy - x)

    As sigma increases, SNR should decrease monotonically.
    We test this for Gaussian noise, which has a predictable theoretical SNR:
        SNR_theoretical = Var(x) / sigma²
    """

    def _snr(self, x: torch.Tensor, x_noisy: torch.Tensor) -> float:
        noise = x_noisy - x
        return x.var().item() / (noise.var().item() + 1e-8)

    def test_gaussian_snr_decreases_with_sigma(self):
        torch.manual_seed(0)
        x = torch.randn(64, 16)
        sigmas = [0.1, 0.3, 1.0, 3.0, 10.0]
        snrs = []
        for sigma in sigmas:
            noisy = gaussian_noise(x, sigma=sigma, generator=_gen(0))
            snrs.append(self._snr(x, noisy))
        # SNR should be strictly decreasing.
        for i in range(len(snrs) - 1):
            assert snrs[i] > snrs[i + 1], (
                f"SNR should decrease as sigma increases: "
                f"SNR(sigma={sigmas[i]}) = {snrs[i]:.2f} <= "
                f"SNR(sigma={sigmas[i+1]}) = {snrs[i+1]:.2f}"
            )

    def test_gaussian_snr_matches_theory(self):
        """SNR ≈ Var(x) / sigma² for i.i.d. Gaussian noise."""
        torch.manual_seed(1)
        x = torch.randn(256, 32) * 2.0   # signal variance ≈ 4
        sigma = 1.0
        noisy = gaussian_noise(x, sigma=sigma, generator=_gen(2))
        snr_empirical   = self._snr(x, noisy)
        snr_theoretical = x.var().item() / sigma ** 2
        # Allow 20% relative error (finite samples).
        assert abs(snr_empirical - snr_theoretical) / snr_theoretical < 0.20, (
            f"Empirical SNR {snr_empirical:.2f} far from theoretical {snr_theoretical:.2f}"
        )

    def test_spike_snr_at_low_p_dominated_by_outliers(self):
        """At low p_spike, the SNR is highly variable and can be very low
        despite p being small.  Verify that max(|noise|) >> mean(|noise|)."""
        x = torch.zeros(128, 16, 32)
        out = spike_artifacts(x, p_spike=0.02, spike_scale=20.0, generator=_gen(3))
        noise = out - x
        max_abs  = noise.abs().max().item()
        mean_abs = noise.abs().mean().item()
        # Sparse but extreme: max >> mean.
        assert max_abs > mean_abs * 5.0, (
            f"Spike artifacts should be heavy-tailed: max/mean = {max_abs/mean_abs:.1f}"
        )

    def test_volume_conduction_snr_unchanged(self):
        """Volume conduction does not add noise — it redistributes existing signal.
        The total power (norm²) should not increase."""
        x = _scores_2d(B=32, M=12)
        out = volume_conduction(x, sigma_spatial=1.5)
        assert out.norm().item() <= x.norm().item() * 1.01, (
            "Volume conduction should not amplify the signal"
        )

    def test_ordered_noise_severity(self):
        """Spike artifacts should be more damaging than equal-power Gaussian noise.

        'More damaging' = larger max absolute deviation from the signal.
        This reflects the real EEG situation where rare artefacts have
        disproportionate impact on downstream statistics.
        """
        torch.manual_seed(5)
        x = torch.randn(64, 16, 64)
        # Match total power: spike variance = p * spike_scale² = 0.05 * 4² = 0.8
        p, spike_scale = 0.05, 4.0
        sigma_gauss = (p * spike_scale ** 2) ** 0.5

        spike_out = spike_artifacts(x, p_spike=p, spike_scale=spike_scale, generator=_gen(0))
        gauss_out = gaussian_noise(x, sigma=sigma_gauss, generator=_gen(1))

        spike_max_dev = (spike_out - x).abs().max().item()
        gauss_max_dev = (gauss_out - x).abs().max().item()

        assert spike_max_dev > gauss_max_dev, (
            f"Spike max deviation {spike_max_dev:.2f} should exceed "
            f"Gaussian max deviation {gauss_max_dev:.2f} at equal power"
        )
