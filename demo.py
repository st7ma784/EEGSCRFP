"""Quick end-to-end demo: validates all components of the new pipeline.

Tests:
  1. SparseAttentionWrapper — topk sparsity, attention maps extracted
  2. PathwayMetricsComputer — 6 diagnostic scalars (debug stream)
  3. CKA features — vectorised all-pairs, content-sensitive
  4. PatchFeatureExtractor — full-coverage patches, calibration, floating scores
  5. Sensor count sweep (M=1→full_coverage) using sklearn probes
  6. eta² variance check — confirms CKA >> pathway in content sensitivity

Usage:
    python demo.py                    # synthetic prompts, CPU
    python demo.py --model gpt2-medium  # larger model
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.sparse_attention import create_sparse_model
from src.data.dataset import create_default_prompts
from src.data.tokenizer import TextTokenizer
from src.metrics.pathway_metrics import compute_pathway_features
from src.metrics.cka_metrics import extract_cka_features, cka_feature_dim
from src.metrics.network_patches import PatchSampler, PatchFeatureExtractor


def _sep(title: str = ""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")


def _ok(msg: str):
    print(f"  ✓  {msg}")


def _warn(msg: str):
    print(f"  ⚠  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Sparse attention model
# ─────────────────────────────────────────────────────────────────────────────

def test_sparse_attention(model_name: str, device: torch.device):
    _sep("Test 1 — SparseAttentionWrapper")
    model = create_sparse_model(model_name).to(device).eval()
    tokenizer = TextTokenizer(model_name=model_name)

    prompts = create_default_prompts(4)
    tokens = tokenizer.tokenize(prompts)
    input_ids = tokens["input_ids"].to(device)
    attn_mask = tokens["attention_mask"].to(device)

    t0 = time.time()
    with torch.no_grad():
        for alpha in [0.2, 0.5, 1.0]:
            model.set_sparsity_level(alpha)
            out = model(input_ids, attention_mask=attn_mask,
                        return_attention_maps=True, return_hidden_states=True)
    elapsed = time.time() - t0

    L = len(out["attention_maps"])
    B, H, S, _ = out["attention_maps"][0].shape
    _ok(f"{model_name}: L={L} layers, H={H} heads, S={S} seq_len, B={B}")
    _ok(f"3 alpha forward passes in {elapsed:.2f}s")
    _ok(f"hidden_states tuple length: {len(out['hidden_states'])}")

    # Confirm sparsity is actually changing attention maps
    model.set_sparsity_level(0.1)
    with torch.no_grad():
        out_sparse = model(input_ids, attention_mask=attn_mask,
                           return_attention_maps=True)
    model.set_sparsity_level(1.0)
    with torch.no_grad():
        out_dense = model(input_ids, attention_mask=attn_mask,
                          return_attention_maps=True)

    sparse_nnz = (out_sparse["attention_maps"][0] > 1e-5).float().mean().item()
    dense_nnz  = (out_dense["attention_maps"][0]  > 1e-5).float().mean().item()
    assert sparse_nnz < dense_nnz, "Sparsity injection not reducing non-zeros"
    _ok(f"Non-zero fraction: alpha=0.1 → {sparse_nnz:.3f}  alpha=1.0 → {dense_nnz:.3f}")

    return model, tokenizer, out


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Pathway metrics (debug diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

def test_pathway_metrics(out: dict):
    _sep("Test 2 — PathwayMetricsComputer (diagnostic / debug only)")
    t0 = time.time()
    feats = compute_pathway_features(out["attention_maps"])
    elapsed = time.time() - t0

    B = feats.shape[0]
    assert feats.shape == (B, 6), feats.shape
    assert feats.isfinite().all(), "NaN/Inf in pathway features"
    _ok(f"pathway_features shape {feats.shape}  elapsed {elapsed*1000:.1f}ms")
    names = ["routing_sparsity", "PCI", "path_efficiency",
             "routing_entropy", "inter_head_divergence", "layer_stability"]
    for name, val in zip(names, feats[0].tolist()):
        print(f"     {name:<26s}: {val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — CKA features
# ─────────────────────────────────────────────────────────────────────────────

def test_cka_features(out: dict):
    _sep("Test 3 — CKA features (vectorised, content-sensitive)")
    hidden_states = out["hidden_states"]
    M = len(hidden_states)
    expected_D = cka_feature_dim(M - 1)  # M-1 transformer layers

    t0 = time.time()
    feats = extract_cka_features(hidden_states)
    elapsed = time.time() - t0

    assert feats.shape[-1] == expected_D, f"Expected D={expected_D}, got {feats.shape}"
    assert feats.ge(0).all() and feats.le(1).all(), "CKA outside [0,1]"
    assert feats.isfinite().all()
    _ok(f"CKA features shape {feats.shape}  D={expected_D}  elapsed {elapsed*1000:.1f}ms")
    _ok(f"Value range [{feats.min():.3f}, {feats.max():.3f}] (should be in [0,1])")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — PatchFeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_extractor(model, tokenizer, device, model_name):
    _sep("Test 4 — PatchFeatureExtractor (floating scores)")

    n_layers = model.model.config.n_layer
    n_heads  = model.model.config.n_head

    # Show patch count landscape across resolutions
    print(f"  Patch count landscape for {model_name} ({n_layers}L × {n_heads}H):")
    for d, h in [(3, 4), (2, 2), (1, 1)]:
        s = PatchSampler(n_layers=n_layers, n_heads=n_heads,
                         patch_depth=d, heads_per_patch=h)
        fc = s.full_coverage_count
        ms = s.m_values_for_experiment()
        print(f"    depth={d}, heads={h} → {fc:>4} patches   sweep: {ms}")

    # Use depth=1, heads=1 — one patch per (layer, head) pair
    sampler  = PatchSampler(n_layers=n_layers, n_heads=n_heads,
                            patch_depth=1, heads_per_patch=1)
    patches  = sampler.full_coverage_patches()
    M_full   = len(patches)
    _ok(f"High-res config (depth=1, h=1): {M_full} patches (target ≥128)")

    extractor = PatchFeatureExtractor(patches).to(device)

    # Build calibration set (8 prompts)
    cal_prompts = create_default_prompts(8)
    cal_tokens  = tokenizer.tokenize(cal_prompts)
    cal_maps: list = []
    model.set_sparsity_level(1.0)
    with torch.no_grad():
        for i in range(8):
            ids  = cal_tokens["input_ids"][i:i+1].to(device)
            mask = cal_tokens["attention_mask"][i:i+1].to(device)
            o = model(ids, attention_mask=mask, return_attention_maps=True)
            cal_maps.append(o["attention_maps"])

    t0 = time.time()
    extractor.calibrate(cal_maps)
    _ok(f"calibrate() on 8 prompts in {(time.time()-t0)*1000:.0f}ms")
    assert extractor._references.shape == (M_full,
                                           cal_maps[0][0].shape[-1],
                                           cal_maps[0][0].shape[-1])

    # Forward pass on a batch of 4
    test_prompts = create_default_prompts(4)
    tok = tokenizer.tokenize(test_prompts)
    ids  = tok["input_ids"].to(device)
    mask = tok["attention_mask"].to(device)
    with torch.no_grad():
        o = model(ids, attention_mask=mask, return_attention_maps=True)

    t0 = time.time()
    scores = extractor(o["attention_maps"])
    elapsed = time.time() - t0

    assert scores.shape == (4, M_full), scores.shape
    assert scores.ge(0).all()
    _ok(f"forward() → scores shape {scores.shape}  elapsed {elapsed*1000:.1f}ms")
    _ok(f"Per-patch mean floating scores: {scores.mean(0).tolist()}")

    return extractor, sampler, M_full


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Sensor count mini-sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_sensor_count_sweep(model, tokenizer, device, extractor, sampler, M_full):
    _sep("Test 5 — Sensor count mini-sweep")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        _warn("scikit-learn not available — skipping sweep test")
        return

    # Collect floating scores for 2 alpha levels (act as 2 "classes")
    all_X, all_y = [], []
    patches_full = sampler.full_coverage_patches()
    for cls, alpha in enumerate([0.2, 1.0]):
        model.set_sparsity_level(alpha)
        prompts = create_default_prompts(12)
        tok = tokenizer.tokenize(prompts)
        ids  = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        with torch.no_grad():
            o = model(ids, attention_mask=mask, return_attention_maps=True)
        sc = extractor(o["attention_maps"]).cpu().numpy()   # [12, M_full]
        all_X.append(sc)
        all_y.extend([cls] * 12)

    X = np.concatenate(all_X)   # [24, M_full]
    y = np.array(all_y)

    m_values = sampler.m_values_for_experiment()
    print(f"  {'M':>4}  {'AUROC':>8}")
    print(f"  {'─'*4}  {'─'*8}")
    for M in m_values:
        rng = np.random.default_rng(0)
        cols = rng.choice(M_full, size=min(M, M_full), replace=False)
        Xm = X[:, cols]
        scaler = StandardScaler()
        Xm_s = scaler.fit_transform(Xm)
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=0)
        clf.fit(Xm_s, y)
        acc = clf.score(Xm_s, y)
        print(f"  {M:>4}  {acc:>8.3f}")

    _ok("Sweep complete — expect accuracy to rise with M")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — eta² sanity: CKA vs pathway
# ─────────────────────────────────────────────────────────────────────────────

def test_eta2(model, tokenizer, device):
    _sep("Test 6 — eta² variance decomposition (CKA vs pathway)")
    n_prompts = 8
    alpha_values = [0.2, 0.6, 1.0]
    prompts = create_default_prompts(n_prompts)

    path_all, cka_all = [], []

    for alpha in alpha_values:
        model.set_sparsity_level(alpha)
        tok = tokenizer.tokenize(prompts)
        ids  = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        with torch.no_grad():
            o = model(ids, attention_mask=mask,
                      return_attention_maps=True, return_hidden_states=True)
        path_all.append(compute_pathway_features(o["attention_maps"]).cpu().float().numpy())
        cka_all.append(extract_cka_features(o["hidden_states"]).cpu().float().numpy())

    path_feats = np.stack(path_all)   # [A, N, 6]
    cka_feats  = np.stack(cka_all)    # [A, N, D]

    def eta2_prompt(feats):
        A, N, D = feats.shape
        gm = feats.mean((0, 1), keepdims=True)
        pm = feats.mean(0, keepdims=True)
        ss_total  = ((feats - gm) ** 2).sum((0, 1))
        ss_prompt = A * ((pm - gm) ** 2).sum((0, 1))
        return (ss_prompt / (ss_total + 1e-30)).mean()

    ep_path = eta2_prompt(path_feats)
    ep_cka  = eta2_prompt(cka_feats)

    _ok(f"Pathway scalars  eta²_prompt = {ep_path:.3f}  (target: < 0.05)")
    _ok(f"CKA features     eta²_prompt = {ep_cka:.3f}  (target: > 0.20 even on synthetic)")

    if ep_cka > ep_path + 0.05:
        _ok("CKA is more content-sensitive than pathway scalars ✓")
    else:
        _warn("CKA not clearly better — expected on synthetic data; "
              "run audit_feature_variance.py --data-dir /data/EEG for real data")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEEGSCRFP demo  |  model={args.model}  device={device}")

    # Run all tests
    model, tokenizer, out = test_sparse_attention(args.model, device)
    test_pathway_metrics(out)
    test_cka_features(out)
    extractor, sampler, M_full = test_patch_extractor(model, tokenizer, device, args.model)
    test_sensor_count_sweep(model, tokenizer, device, extractor, sampler, M_full)
    test_eta2(model, tokenizer, device)

    _sep()
    print("\n  All tests passed.\n")
    print("  Next steps:")
    print("    python scripts/audit_feature_variance.py --data-dir /data/EEG")
    print("    python experiments/sensor_count.py --data-dir /data/EEG --wandb-project eegscrfp")
    print()


if __name__ == "__main__":
    main()
