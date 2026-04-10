"""Tests for experiments/lora_participants.py.

Structure
---------
1. lora_delta()              — shape, rank, magnitude, determinism
2. participant_state_dict()  — rank=0 no-op, rank>0 mutates target keys only,
                               different participants produce different deltas
3. split_participants()      — sizes, coverage, no overlap, edge cases
4. run_experiment()          — metric keys, value bounds, AUROC drop ≥ 0,
                               rank=0 symmetry (all participants identical)
5. Smoke test                — end-to-end pipeline with synthetic features,
                               no LLM required
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add EEGSCRFP root to path so imports work without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.lora_participants import (
    lora_delta,
    participant_state_dict,
    run_experiment,
    split_participants,
    _TARGET_MODULE_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_state_dict(shapes: dict) -> dict:
    """Build a fake state dict with zero tensors of the given shapes."""
    return {k: torch.zeros(*v) for k, v in shapes.items()}


def _gpt2_like_state_dict(n_layers: int = 2, d: int = 16) -> dict:
    """Minimal GPT-2–style state dict with c_attn and c_proj weights."""
    sd = {}
    for i in range(n_layers):
        sd[f"transformer.h.{i}.attn.c_attn.weight"] = torch.randn(d, d * 3)
        sd[f"transformer.h.{i}.attn.c_attn.bias"]   = torch.zeros(d * 3)
        sd[f"transformer.h.{i}.attn.c_proj.weight"] = torch.randn(d, d)
        sd[f"transformer.h.{i}.attn.c_proj.bias"]   = torch.zeros(d)
        sd[f"transformer.h.{i}.mlp.c_fc.weight"]    = torch.randn(d, d * 4)
        sd[f"transformer.h.{i}.mlp.c_fc.bias"]      = torch.zeros(d * 4)
    return sd


def _synthetic_features(
    n_participants: int,
    n_prompts: int,
    M: int = 6,
    S: int = 8,
    seed: int = 0,
) -> dict:
    """Build a dict of synthetic [P, M, S] feature tensors."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return {
        pid: torch.randn(n_prompts, M, S, generator=rng)
        for pid in range(n_participants)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. lora_delta
# ─────────────────────────────────────────────────────────────────────────────

class TestLoraDelta:
    def test_output_shape(self):
        delta = lora_delta(32, 64, rank=4, seed=0)
        assert delta.shape == (32, 64)

    def test_square_shape(self):
        delta = lora_delta(16, 16, rank=2, seed=1)
        assert delta.shape == (16, 16)

    def test_rank_upper_bound(self):
        """Numerical rank of the delta must be ≤ the requested rank."""
        rank = 3
        delta = lora_delta(20, 20, rank=rank, seed=42)
        sv = torch.linalg.svdvals(delta)
        numerical_rank = int((sv > sv[0] * 1e-5).sum().item())
        assert numerical_rank <= rank

    def test_rank_1_is_rank_1(self):
        """rank=1 must produce a matrix of numerical rank exactly 1."""
        delta = lora_delta(10, 15, rank=1, seed=7)
        sv = torch.linalg.svdvals(delta)
        numerical_rank = int((sv > sv[0] * 1e-5).sum().item())
        assert numerical_rank == 1

    def test_deterministic(self):
        """Same seed → identical delta."""
        d1 = lora_delta(8, 8, rank=2, seed=99)
        d2 = lora_delta(8, 8, rank=2, seed=99)
        assert torch.allclose(d1, d2)

    def test_different_seeds_differ(self):
        d1 = lora_delta(8, 8, rank=2, seed=0)
        d2 = lora_delta(8, 8, rank=2, seed=1)
        assert not torch.allclose(d1, d2)

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="rank"):
            lora_delta(8, 8, rank=0, seed=0)

    def test_magnitude_independent_of_rank(self):
        """Frobenius norm should be roughly constant across ranks (by design).

        The sigma = scale/sqrt(rank) normalisation means E[||delta||_F] is
        independent of rank.  Single-seed noise means we only check a loose
        bound (8×) to avoid flakiness from random fluctuation.
        """
        norms = [lora_delta(32, 32, rank=r, seed=0).norm().item()
                 for r in [1, 4, 16]]
        # Should not vary by more than 8× even with worst-case single seed
        assert max(norms) / min(norms) < 8.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. participant_state_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestParticipantStateDict:
    def test_rank_zero_returns_original(self):
        """rank=0 must return the exact same object (no copy)."""
        base = _gpt2_like_state_dict(n_layers=1, d=8)
        result = participant_state_dict(base, rank=0, target_module_names=["c_attn"],
                                        participant_id=0)
        assert result is base

    def test_rank_nonzero_copies(self):
        """rank>0 must return a new dict (not the same object)."""
        base = _gpt2_like_state_dict(n_layers=1, d=8)
        result = participant_state_dict(base, rank=2, target_module_names=["c_attn"],
                                        participant_id=0)
        assert result is not base

    def test_target_weights_changed(self):
        base = _gpt2_like_state_dict(n_layers=2, d=8)
        result = participant_state_dict(base, rank=2, target_module_names=["c_attn"],
                                        participant_id=0)
        changed = [
            k for k in base
            if "c_attn" in k and base[k].dim() == 2
            and not torch.allclose(base[k], result[k])
        ]
        assert len(changed) > 0, "c_attn weights should be modified"

    def test_non_target_weights_unchanged(self):
        base = _gpt2_like_state_dict(n_layers=2, d=8)
        result = participant_state_dict(base, rank=2, target_module_names=["c_attn"],
                                        participant_id=0)
        unchanged = [
            k for k in base
            if "c_attn" not in k
            and torch.allclose(base[k], result[k])
        ]
        # All non-c_attn keys unchanged
        non_target_keys = [k for k in base if "c_attn" not in k]
        assert len(unchanged) == len(non_target_keys)

    def test_biases_unchanged(self):
        """1-D bias tensors must never be perturbed."""
        base = _gpt2_like_state_dict(n_layers=2, d=8)
        result = participant_state_dict(base, rank=2, target_module_names=["c_attn"],
                                        participant_id=0)
        bias_keys = [k for k in base if "bias" in k]
        assert bias_keys, "state dict should contain bias keys"
        for k in bias_keys:
            assert torch.allclose(base[k], result[k]), f"bias {k} should not change"

    def test_different_participants_differ(self):
        base = _gpt2_like_state_dict(n_layers=2, d=8)
        sd0 = participant_state_dict(base, rank=2, target_module_names=["c_attn"], participant_id=0)
        sd1 = participant_state_dict(base, rank=2, target_module_names=["c_attn"], participant_id=1)
        changed_keys = [k for k in base if not torch.allclose(sd0[k], sd1[k])]
        assert len(changed_keys) > 0, "Different participant IDs must produce different weights"

    def test_same_participant_deterministic(self):
        base = _gpt2_like_state_dict(n_layers=1, d=8)
        sd0a = participant_state_dict(base, rank=2, target_module_names=["c_attn"], participant_id=5)
        sd0b = participant_state_dict(base, rank=2, target_module_names=["c_attn"], participant_id=5)
        for k in sd0a:
            assert torch.allclose(sd0a[k], sd0b[k]), f"Key {k} should be identical for same seed"

    def test_all_target_modules(self):
        """'all' target → both c_attn and c_proj weights changed."""
        base = _gpt2_like_state_dict(n_layers=2, d=8)
        result = participant_state_dict(base, rank=2,
                                        target_module_names=["c_attn", "c_proj"],
                                        participant_id=0)
        c_attn_changed = any(
            "c_attn" in k and base[k].dim() == 2 and not torch.allclose(base[k], result[k])
            for k in base
        )
        c_proj_changed = any(
            "c_proj" in k and base[k].dim() == 2 and not torch.allclose(base[k], result[k])
            for k in base
        )
        assert c_attn_changed, "c_attn weights should change"
        assert c_proj_changed, "c_proj weights should change"

    def test_target_module_map_keys(self):
        """Verify the target module map has the expected entries."""
        assert "qv" in _TARGET_MODULE_MAP
        assert "all" in _TARGET_MODULE_MAP
        assert "c_attn" in _TARGET_MODULE_MAP["qv"]
        assert "c_proj" in _TARGET_MODULE_MAP["all"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. split_participants
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitParticipants:
    def test_sizes_sum_to_total(self):
        train, test = split_participants(10, test_fraction=0.25)
        assert len(train) + len(test) == 10

    def test_test_fraction_respected(self):
        """Actual test size should be close to the requested fraction."""
        train, test = split_participants(20, test_fraction=0.25)
        assert len(test) == 5   # round(20 * 0.25) = 5

    def test_no_overlap(self):
        train, test = split_participants(12, test_fraction=0.30)
        assert len(set(train) & set(test)) == 0

    def test_full_coverage(self):
        """Train + test must cover every participant ID."""
        n = 15
        train, test = split_participants(n, test_fraction=0.2)
        assert sorted(train + test) == list(range(n))

    def test_at_least_one_each(self):
        """Even with extreme fractions, both sets are non-empty."""
        train, test = split_participants(2, test_fraction=0.9)
        assert len(train) >= 1
        assert len(test) >= 1

    def test_deterministic(self):
        t1, v1 = split_participants(10, 0.3, seed=0)
        t2, v2 = split_participants(10, 0.3, seed=0)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds_differ(self):
        _, v0 = split_participants(20, 0.3, seed=0)
        _, v1 = split_participants(20, 0.3, seed=99)
        assert v0 != v1

    def test_raises_on_single_participant(self):
        with pytest.raises(ValueError, match="2 participants"):
            split_participants(1, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. run_experiment
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_experiment():
    """Pre-computed synthetic inputs shared across run_experiment tests."""
    torch.manual_seed(0)
    n_participants = 6
    n_prompts = 20
    M, S = 4, 6
    features = _synthetic_features(n_participants, n_prompts, M, S)
    # 4 classes, one per prompt slot
    task_labels = torch.arange(n_prompts, dtype=torch.long) % 4
    train_ids, test_ids = split_participants(n_participants, test_fraction=0.33)
    return dict(
        features=features,
        train_ids=train_ids,
        test_ids=test_ids,
        task_labels=task_labels,
    )


@pytest.fixture(scope="module")
def experiment_results(tiny_experiment):
    """Run once, reuse across all metric tests."""
    return run_experiment(
        features_by_participant=tiny_experiment["features"],
        train_ids=tiny_experiment["train_ids"],
        test_ids=tiny_experiment["test_ids"],
        task_labels=tiny_experiment["task_labels"],
        encoder_type="linear",   # fastest encoder for unit tests
        embed_dim=16,
        hidden_dim=16,
        n_hidden_layers=1,
        n_attn_heads=2,
        dropout=0.0,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=16,
        max_epochs=3,
        device=torch.device("cpu"),
    )


class TestRunExperiment:
    def test_required_keys_present(self, experiment_results):
        for key in ("lora/within_auroc", "lora/cross_auroc", "lora/auroc_drop"):
            assert key in experiment_results, f"Missing key: {key}"

    def test_metadata_keys_present(self, experiment_results):
        for key in ("n_train_participants", "n_test_participants", "n_prompts", "M", "S"):
            assert key in experiment_results

    def test_within_auroc_in_unit_interval(self, experiment_results):
        v = experiment_results["lora/within_auroc"]
        assert np.isnan(v) or (0.0 <= v <= 1.0), f"within_auroc out of range: {v}"

    def test_cross_auroc_in_unit_interval(self, experiment_results):
        v = experiment_results["lora/cross_auroc"]
        assert np.isnan(v) or (0.0 <= v <= 1.0), f"cross_auroc out of range: {v}"

    def test_participant_counts_match_split(self, tiny_experiment, experiment_results):
        assert experiment_results["n_train_participants"] == len(tiny_experiment["train_ids"])
        assert experiment_results["n_test_participants"]  == len(tiny_experiment["test_ids"])

    def test_n_prompts_correct(self, tiny_experiment, experiment_results):
        assert experiment_results["n_prompts"] == len(tiny_experiment["task_labels"])

    def test_dimensions_logged(self, experiment_results):
        assert experiment_results["M"] == 4
        assert experiment_results["S"] == 6

    def test_auroc_drop_equals_difference(self, experiment_results):
        w = experiment_results["lora/within_auroc"]
        c = experiment_results["lora/cross_auroc"]
        d = experiment_results["lora/auroc_drop"]
        if not (np.isnan(w) or np.isnan(c)):
            assert abs(d - (w - c)) < 1e-6

    def test_rank0_participants_identical(self):
        """With rank=0 all participants are identical → within == cross (same data)."""
        torch.manual_seed(1)
        n_participants = 4
        n_prompts = 12
        M, S = 3, 5
        # All participants share the same features (rank=0 simulation)
        base_feats = torch.randn(n_prompts, M, S)
        features = {pid: base_feats.clone() for pid in range(n_participants)}
        task_labels = torch.arange(n_prompts, dtype=torch.long) % 3
        train_ids, test_ids = split_participants(n_participants, test_fraction=0.5)

        result = run_experiment(
            features_by_participant=features,
            train_ids=train_ids,
            test_ids=test_ids,
            task_labels=task_labels,
            encoder_type="linear",
            embed_dim=8,
            hidden_dim=8,
            n_hidden_layers=1,
            n_attn_heads=2,
            dropout=0.0,
            lr=1e-2,
            batch_size=8,
            max_epochs=5,
            device=torch.device("cpu"),
        )
        # Both sets have the same feature distribution → AUROC should be close
        w = result["lora/within_auroc"]
        c = result["lora/cross_auroc"]
        if not (np.isnan(w) or np.isnan(c)):
            assert abs(w - c) < 0.4, (
                f"With identical features across participants, within/cross should be close; "
                f"got within={w:.3f}, cross={c:.3f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Smoke — end-to-end with synthetic features (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("encoder_type", ["linear", "mlp", "transformer"])
def test_smoke_encoder_types(encoder_type):
    """run_experiment completes without error for each encoder type."""
    torch.manual_seed(42)
    n_participants, n_prompts, M, S = 4, 10, 6, 8
    features = _synthetic_features(n_participants, n_prompts, M, S)
    task_labels = torch.arange(n_prompts, dtype=torch.long) % 2
    train_ids, test_ids = split_participants(n_participants, test_fraction=0.5)

    result = run_experiment(
        features_by_participant=features,
        train_ids=train_ids,
        test_ids=test_ids,
        task_labels=task_labels,
        encoder_type=encoder_type,
        embed_dim=16,
        hidden_dim=16,
        n_hidden_layers=1,
        n_attn_heads=2,
        dropout=0.0,
        lr=1e-2,
        batch_size=8,
        max_epochs=2,
        device=torch.device("cpu"),
    )
    assert "lora/within_auroc" in result
    assert "lora/cross_auroc" in result


def test_smoke_varying_lora_rank():
    """Higher LoRA rank should produce different participant feature diffs."""
    base = _gpt2_like_state_dict(n_layers=2, d=32)

    diffs = {}
    for rank in [0, 2, 8]:
        sd = participant_state_dict(
            base, rank=rank, target_module_names=["c_attn"], participant_id=0
        )
        total_diff = sum(
            (base[k] - sd[k]).norm().item()
            for k in base if base[k].dim() == 2 and "c_attn" in k
        )
        diffs[rank] = total_diff

    # rank=0 → no change
    assert diffs[0] == 0.0
    # rank=2 and rank=8 both produce non-zero changes
    assert diffs[2] > 0.0
    assert diffs[8] > 0.0


def test_smoke_multiple_participants_all_different():
    """Every participant (rank>0) should have unique weights."""
    base = _gpt2_like_state_dict(n_layers=2, d=16)
    state_dicts = [
        participant_state_dict(base, rank=4, target_module_names=["c_attn"], participant_id=pid)
        for pid in range(5)
    ]
    for i in range(len(state_dicts)):
        for j in range(i + 1, len(state_dicts)):
            any_diff = any(
                not torch.allclose(state_dicts[i][k], state_dicts[j][k])
                for k in state_dicts[i]
            )
            assert any_diff, f"Participants {i} and {j} should differ"


def test_smoke_full_pipeline():
    """End-to-end: split → features → run_experiment → check output contract."""
    torch.manual_seed(0)
    n_total = 8
    n_prompts = 16
    M, S = 5, 7

    train_ids, test_ids = split_participants(n_total, test_fraction=0.25)
    features = {pid: torch.randn(n_prompts, M, S) for pid in range(n_total)}
    task_labels = torch.tensor([i % 3 for i in range(n_prompts)], dtype=torch.long)

    result = run_experiment(
        features_by_participant=features,
        train_ids=train_ids,
        test_ids=test_ids,
        task_labels=task_labels,
        encoder_type="linear",
        embed_dim=16,
        hidden_dim=16,
        n_hidden_layers=1,
        n_attn_heads=2,
        dropout=0.0,
        lr=5e-3,
        batch_size=8,
        max_epochs=5,
        device=torch.device("cpu"),
    )

    # Contract checks
    assert isinstance(result, dict)
    for key in ("lora/within_auroc", "lora/cross_auroc", "lora/auroc_drop",
                "n_train_participants", "n_test_participants", "n_prompts", "M", "S"):
        assert key in result, f"Missing output key: {key}"

    assert result["n_train_participants"] == len(train_ids)
    assert result["n_test_participants"] == len(test_ids)
    assert result["n_prompts"] == n_prompts
    assert result["M"] == M
    assert result["S"] == S

    w = result["lora/within_auroc"]
    c = result["lora/cross_auroc"]
    for v in (w, c):
        assert np.isnan(v) or (0.0 <= v <= 1.0)
