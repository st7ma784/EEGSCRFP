"""Orientation-based neuron → EEG sensor forward model.

Each (layer, head) unit in the LLM is assigned a fixed, randomly-drawn unit
orientation on S² (analogous to a cortical dipole direction).  When a unit is
active at token position t, it contributes to every EEG sensor according to a
``cos(θ) / θ`` spatial kernel, where θ is the angular distance between the
unit's orientation focal point on the scalp and each sensor.  Activations from
deeper layers arrive with a layer-proportional propagation delay.

This replaces the arbitrary ``NetworkPatch`` approach.  Instead of grouping
neurons into patches by ``(layer_window, head_subset)``, every (layer, head)
unit is given its own spatial address on the scalp.  The EEG signal at sensor
*s* at time *t* is the weighted sum of all units pointing towards *s* that
were active at ``t − delay[layer]``.

Lead field design
-----------------
The lead field matrix ``G`` has shape ``(n_units, n_sensors)`` where
``n_units = n_layers × n_heads``.  Entry ``G[n, s]`` is the contribution of
unit *n* to sensor *s*:

    G[n, s] = cos(k · θ_{ns}) / max(k · θ_{ns}, ε)

where:
  * θ_{ns} = geodesic angle (radians) between orientation[n] and sensor[s]
  * k      = π / (2 · half_angle_bandwidth_rad) — controls spatial spread
  * ε      = 1e-6 — avoids division by zero at θ = 0 (limit = 1)

The kernel is a damped oscillation: value ≈ 1 when sensor is directly in line
with the dipole, oscillating toward zero as sensors become more distant.  The
``half_angle_bandwidth_rad`` parameter sets the angular distance at which the
kernel first reaches zero.

Delay model
-----------
``delay[l] = round(l / (n_layers − 1) × max_delay_samples)``

Layer 0 (first layer) contributes immediately (delay 0).  The deepest layer
contributes after ``max_delay_samples``.  This models the intuition that
higher-level representations take longer to surface as observable EEG signals.

Output interface
----------------
Identical to ``PatchFeatureExtractor.forward_temporal()``:

    input:  list of L tensors [B, H, S, S]  (per-layer attention maps)
    output: [B, n_sensors, S + max_delay]

The output plugs directly into ``build_patch_encoder()`` / ``PatchTokenEncoder``
with ``M = n_sensors``.

GPU performance
---------------
* All geometry (orientations, lead field, delays) is precomputed at ``__init__``
  and stored as ``nn.Module`` buffers — moves with ``.to(device)``.
* ``forward()`` has one Python loop over ``n_layers`` (~12 iterations for GPT-2)
  and two fused tensor operations per layer.  No per-sensor or per-neuron loops.
* Calibration (population baseline) is accumulated in one vectorised pass.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_unit_orientations(n: int, seed: int = 0) -> torch.Tensor:
    """Draw *n* unit vectors uniformly on S² (spherical uniform distribution).

    Uses the standard method: sample 3-D Gaussian, normalise.  Stored as
    float32.

    Args:
        n:    Number of orientation vectors.
        seed: RNG seed for reproducibility.

    Returns:
        ``(n, 3)`` float32 tensor, each row has unit L2 norm.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    raw = torch.randn(n, 3, generator=rng)
    return F.normalize(raw, dim=1)


def _cosine_sinc_kernel(
    angles: torch.Tensor,
    bandwidth_rad: float,
) -> torch.Tensor:
    """cos(x) / x spatial decay kernel.

    Evaluated at all (unit, sensor) angular separations simultaneously.

    ``w = cos(k · θ) / max(k · θ, ε)``

    At θ → 0 the limit is 1.  The first zero crossing is at
    ``θ = π / (2k) = bandwidth_rad``, giving intuitive control over how
    many sensors a dipole influences.

    Args:
        angles:        ``(n_units, n_sensors)`` angular separations in radians,
                       values in ``[0, π]``.
        bandwidth_rad: Half-angle bandwidth — distance (rad) at which the kernel
                       first reaches zero.  Use ``np.pi / 4`` (45°) for broad
                       spread, ``np.pi / 8`` (22.5°) for sharp.

    Returns:
        ``(n_units, n_sensors)`` float32 kernel weights.
    """
    k = math.pi / (2.0 * bandwidth_rad)
    x = k * angles                                      # (n_units, n_sensors)
    denom = x.clamp(min=_EPS)
    return torch.cos(x) / denom                         # limit at x=0 is 1


def _angular_distances(
    orientations: torch.Tensor,   # (n_units, 3)
    sensor_positions: torch.Tensor,  # (n_sensors, 3)
) -> torch.Tensor:
    """Geodesic angle (radians) between every unit–sensor pair.

    Both tensors are assumed to contain unit vectors.  The angle is
    ``arccos(dot(u, s))`` clamped to ``[0, π]``.

    Returns:
        ``(n_units, n_sensors)`` float32 tensor of angles in ``[0, π]``.
    """
    # dot product: (n_units, 3) × (3, n_sensors) → (n_units, n_sensors)
    dots = orientations @ sensor_positions.T
    dots = dots.clamp(-1.0 + _EPS, 1.0 - _EPS)         # numerical safety
    return torch.acos(dots)


def _layer_delays(
    n_layers: int,
    max_delay_samples: int,
) -> torch.Tensor:
    """Integer delay (samples) for each layer, proportional to depth.

    Layer 0 → delay 0.  Layer ``n_layers − 1`` → delay ``max_delay_samples``.
    Intermediate layers are linearly interpolated and rounded.

    Args:
        n_layers:          Number of transformer layers.
        max_delay_samples: Maximum delay applied to the deepest layer.

    Returns:
        ``(n_layers,)`` int64 tensor.
    """
    if n_layers == 1:
        return torch.zeros(1, dtype=torch.long)
    t = torch.linspace(0.0, float(max_delay_samples), n_layers)
    return t.round().long()


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class NeuronSensorMap(nn.Module):
    """Biophysical forward model: LLM (layer, head) activations → EEG.

    Each (layer, head) unit is assigned a fixed random orientation on S².
    Its activation at each token position contributes to every EEG sensor via
    a ``cos(θ)/θ`` spatial kernel and arrives with a layer-dependent delay.

    Typical usage::

        nmap = NeuronSensorMap(
            n_layers=12, n_heads=12,
            sensor_positions=NeuronSensorMap.sensor_positions_sphere(67),
        )
        nmap.calibrate(attention_maps_list)          # set population baseline
        eeg = nmap(attention_maps)                   # [B, 67, S + 10]

    Args:
        n_layers:              Number of transformer layers.
        n_heads:               Number of attention heads per layer.
        sensor_positions:      ``(n_sensors, 3)`` unit-sphere sensor positions.
                               Use :meth:`sensor_positions_sphere` for a
                               synthetic layout or :meth:`sensor_positions_from_mne`
                               for the standard 10-10 montage.
        bandwidth_rad:         Spatial decay bandwidth (radians).  Controls how
                               many sensors a single dipole influences.
                               Default ``π/4`` (45°) ≈ 12 sensors for 67-channel
                               layout.
        max_delay_samples:     Maximum propagation delay for the deepest layer.
                               At 100 Hz: 10 samples = 100 ms.
                               At 500 Hz: 10 samples = 20 ms.
        orientation_seed:      RNG seed for the fixed random orientations.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        sensor_positions: torch.Tensor,
        bandwidth_rad: float = math.pi / 4,
        max_delay_samples: int = 10,
        orientation_seed: int = 42,
    ) -> None:
        super().__init__()

        self._n_layers = n_layers
        self._n_heads = n_heads
        n_units = n_layers * n_heads
        n_sensors = sensor_positions.shape[0]
        self._n_sensors = n_sensors

        # ── Fixed random orientations ──────────────────────────────────────────
        # Each (layer, head) unit gets one unit vector on S².
        # Shape: (n_units, 3), registered as non-trainable buffer.
        orientations = _random_unit_orientations(n_units, seed=orientation_seed)
        self.register_buffer("orientations", orientations)       # (N, 3)

        # ── Sensor positions (unit sphere) ────────────────────────────────────
        sensor_pos = F.normalize(sensor_positions.float(), dim=1)
        self.register_buffer("sensor_positions", sensor_pos)     # (S, 3)

        # ── Lead field: (n_units, n_sensors) ─────────────────────────────────
        # Precomputed at init; never changes during training.
        angles = _angular_distances(orientations, sensor_pos)    # (N, S)
        lead_field = _cosine_sinc_kernel(angles, bandwidth_rad)  # (N, S)
        self.register_buffer("lead_field", lead_field)           # (N, S)

        # ── Layer delays ───────────────────────────────────────────────────────
        delays = _layer_delays(n_layers, max_delay_samples)
        self.register_buffer("layer_delays", delays)             # (n_layers,) int64
        self.max_delay = int(delays.max().item())

        # ── Calibration buffer (population baseline) ──────────────────────────
        # Set by calibrate(); shape (n_layers, n_heads, S_ref, S_ref).
        # None until calibrated; forward uses batch mean as fallback.
        self.register_buffer("_reference", None)
        self._calibrated: bool = False

    # ── Sensor position helpers ───────────────────────────────────────────────

    @staticmethod
    def sensor_positions_sphere(
        n_sensors: int,
        seed: int = 0,
    ) -> torch.Tensor:
        """Generate *n_sensors* quasi-uniform positions on the upper hemisphere.

        Positions are drawn from a uniform sphere, then the bottom hemisphere
        (z < 0) is reflected upward (``z → |z|``), approximating an EEG cap.

        Args:
            n_sensors: Number of sensor positions.
            seed:      RNG seed.

        Returns:
            ``(n_sensors, 3)`` float32 unit vectors, all with z ≥ 0.
        """
        pos = _random_unit_orientations(n_sensors, seed=seed)
        pos[:, 2] = pos[:, 2].abs()     # upper hemisphere
        return F.normalize(pos, dim=1)

    @staticmethod
    def sensor_positions_from_mne(
        ch_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Load standard 10-10 sensor positions from MNE.

        Requires ``mne`` to be installed.  Returns unit-sphere Cartesian
        coordinates from the ``standard_1005`` montage.

        Args:
            ch_names: Explicit list of channel names to extract.  If ``None``,
                      uses the 67-channel standard 10-10 set from
                      :data:`STANDARD_67_CHANNELS`.

        Returns:
            ``(n_sensors, 3)`` float32 unit vectors.

        Raises:
            ImportError: if MNE is not installed.
            KeyError:    if a requested channel is not in the montage.
        """
        import mne  # type: ignore[import]

        if ch_names is None:
            ch_names = STANDARD_67_CHANNELS

        montage = mne.channels.make_standard_montage("standard_1005")
        name_to_pos = {
            name: pos
            for name, pos in zip(montage.ch_names, montage.get_positions()["ch_pos"].values())
        }
        pos_list = []
        for ch in ch_names:
            if ch not in name_to_pos:
                raise KeyError(f"Channel '{ch}' not found in standard_1005 montage")
            pos_list.append(name_to_pos[ch])

        xyz = torch.tensor(np.stack(pos_list), dtype=torch.float32)
        return F.normalize(xyz, dim=1)

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(
        self,
        attention_maps_list: List[List[torch.Tensor]],
    ) -> None:
        """Estimate the population-mean attention pattern for each (layer, head).

        Accumulates the mean attention map across *N* samples.  The reference
        is used in :meth:`forward` to compute per-unit activation as deviation
        from the population baseline (same logic as ``PatchFeatureExtractor``).

        Handles variable sequence lengths by padding to ``S_max`` before
        accumulating.

        Args:
            attention_maps_list: N-length list; each element is a list of L
                tensors ``[1, H, S_i, S_i]`` from a single forward pass.
        """
        device = attention_maps_list[0][0].device
        N = len(attention_maps_list)
        L = len(attention_maps_list[0])
        H = attention_maps_list[0][0].shape[1]
        S_max = max(m[0].shape[-1] for m in attention_maps_list)

        acc = torch.zeros(L, H, S_max, S_max, device=device)

        for attn_maps in attention_maps_list:
            for l, a in enumerate(attn_maps):      # a: [1, H, S_i, S_i]
                a = a.squeeze(0)                   # [H, S_i, S_i]
                S_i = a.shape[-1]
                if S_i < S_max:
                    a = F.pad(a, (0, S_max - S_i, 0, S_max - S_i))
                acc[l] += a                        # [H, S_max, S_max]

        self.register_buffer("_reference", acc / N)   # [L, H, S_max, S_max]
        self._calibrated = True

    # ── Activation extraction ─────────────────────────────────────────────────

    def _compute_activations(
        self,
        stacked: torch.Tensor,       # [L, B, H, S, S]
    ) -> torch.Tensor:               # [B, n_units, S]
        """Per-unit, per-token-position activation from attention maps.

        Activation is the RMS deviation of each head's attention distribution
        from the population baseline, over the key axis:

            act[l, b, h, t] = sqrt( mean_k( ((attn - ref) / std)² ) )

        This is the same statistic as the "floating score" in
        ``PatchFeatureExtractor``, but computed per individual head rather
        than across an arbitrary layer+head group.

        Args:
            stacked: ``[L, B, H, S, S]`` stacked attention maps.

        Returns:
            ``[B, n_units, S]`` activation magnitudes,
            where ``n_units = L × H``.
        """
        L, B, H, S, _ = stacked.shape

        if self._reference is not None and self._reference.shape[-1] == S:
            ref = self._reference.unsqueeze(1)           # [L, 1, H, S, S]
        else:
            ref = stacked.mean(dim=1, keepdim=True)      # [L, 1, H, S, S]

        ref_std = stacked.std(dim=1, keepdim=True).clamp(min=_EPS)
        z = (stacked - ref) / ref_std                    # [L, B, H, S, S]

        # RMS over key positions → one activation value per (layer, head, query)
        act = z.pow(2).mean(dim=-1).sqrt()               # [L, B, H, S]

        # Reshape to [B, n_units=L×H, S]
        return act.permute(1, 0, 2, 3).reshape(B, L * H, S)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        attention_maps: List[torch.Tensor],   # L × [B, H, S, S]
    ) -> torch.Tensor:
        """Map LLM attention maps to a synthetic EEG signal.

        Each (layer, head) unit's activation at each token position is projected
        onto the sensor array via the precomputed lead field and shifted by the
        layer's propagation delay.

        Args:
            attention_maps: List of *L* tensors ``[B, H, S, S]``, one per layer
                (same format as ``SparseAttentionWrapper`` output and
                ``PatchFeatureExtractor`` input).

        Returns:
            ``[B, n_sensors, S + max_delay]`` float32 tensor — a synthetic EEG
            signal in the same ``[B, channels, time]`` format expected by
            ``build_patch_encoder()`` / ``PatchTokenEncoder``.
        """
        stacked = torch.stack(attention_maps, dim=0)     # [L, B, H, S, S]
        L, B, H, S, _ = stacked.shape

        act = self._compute_activations(stacked)         # [B, n_units, S]
        T_out = S + self.max_delay

        eeg = torch.zeros(
            B, self._n_sensors, T_out,
            device=stacked.device, dtype=stacked.dtype,
        )

        for l in range(L):
            delay_l = int(self.layer_delays[l].item())
            # act_l: [B, H, S] — activations for this layer's heads
            act_l = act[:, l * H:(l + 1) * H, :]        # [B, H, S]
            # lf_l:  [H, n_sensors] — lead field slice for this layer's heads
            lf_l = self.lead_field[l * H:(l + 1) * H, :]  # [H, n_sensors]

            # [B, S, H] @ [H, n_sensors] → [B, S, n_sensors] → [B, n_sensors, S]
            contrib = act_l.permute(0, 2, 1) @ lf_l     # [B, S, n_sensors]
            eeg[:, :, delay_l:delay_l + S].add_(contrib.permute(0, 2, 1))

        return eeg   # [B, n_sensors, T_out]

    # ── Diagnostic / visualisation ────────────────────────────────────────────

    def focal_points(self) -> torch.Tensor:
        """Unit sphere focal points for all (layer, head) units.

        The focal point of unit *n* is its orientation vector — the scalp
        location it "points at".  The lead field value ``G[n, :]`` is highest
        at the nearest sensor to this point.

        Returns:
            ``(n_units, 3)`` float32 orientations (same as ``self.orientations``).
        """
        return self.orientations

    def sensor_lead_field(self, unit_idx: int) -> torch.Tensor:
        """Lead field weights for a single unit across all sensors.

        Useful for plotting the spatial spread of one dipole on a scalp map.

        Args:
            unit_idx: Index into the ``n_units = n_layers × n_heads`` axis.
                      Unit ``l * n_heads + h`` corresponds to layer *l*, head *h*.

        Returns:
            ``(n_sensors,)`` float32 weights.
        """
        return self.lead_field[unit_idx]

    def unit_index(self, layer: int, head: int) -> int:
        """Flat index for a given (layer, head) pair."""
        return layer * self._n_heads + head


# ─────────────────────────────────────────────────────────────────────────────
# Standard 10-10 channel list (67 channels)
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_67_CHANNELS: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
    "AF7", "AF3", "AF4", "AF8",
    "F5", "F1", "F2", "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10",
    "C5", "C1", "C2", "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8",
    "P5", "P1", "P2", "P6",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "Fpz", "FCz", "Oz",
]
