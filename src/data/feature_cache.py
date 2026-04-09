"""One-time pathway-feature extraction and disk cache.

Because the sparse attention model is frozen, the pathway features for a
given (text, sparsity_level) pair are fully deterministic.  Running the LLM
on every training epoch is therefore pure wasted compute.

Usage (two-phase training)
--------------------------
Phase 1 — extract once:

    from src.data.feature_cache import extract_and_cache
    extract_and_cache(sparse_model, dataset, collate_fn, cache_path, device)

Phase 2 — train on cached tensors (no LLM):

    from src.data.feature_cache import CachedFeaturesDataset
    dataset = CachedFeaturesDataset(cache_path)
    # items: {'features': [6], 'sparsity_level': scalar}

The CachedFeaturesDataset items are collatable by the default PyTorch
collator — no custom collate_fn needed, and no tokenization overhead.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from src.metrics.pathway_metrics import compute_pathway_features


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_and_cache(
    sparse_model,
    dataset: Dataset,
    collate_fn,
    cache_path: str | Path,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    """Run the frozen LLM once over *dataset* and save pathway features.

    Samples are sorted by sparsity level before batching so that each batch
    has a narrow sparsity range; the batch-mean sparsity is used as the
    injection level — the same approximation made during training.

    Args:
        sparse_model: SparseAttentionWrapper (should be in eval mode).
        dataset: SparsityDataset or NarrativeSparsityDataset — items must
            have ``text`` (str) and ``sparsity_level`` (float tensor) keys.
        collate_fn: tokenising collate function (from ``get_collate_fn``).
        cache_path: where to save the ``.pt`` cache file.
        device: target device (defaults to sparse_model's device).
        batch_size: extraction batch size (can be larger than training BS
            since we only need a forward pass, no gradients).
        num_workers: DataLoader workers for tokenisation.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(sparse_model.parameters()).device

    sparse_model.eval()

    # Sort dataset indices by sparsity so consecutive batches share a level.
    # This minimises the approximation error from using the batch-mean sparsity.
    sparsity_vals = [dataset[i]["sparsity_level"].item() for i in range(len(dataset))]
    sorted_indices = sorted(range(len(dataset)), key=lambda i: sparsity_vals[i])

    sorted_dataset = torch.utils.data.Subset(dataset, sorted_indices)
    loader = DataLoader(
        sorted_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    all_features: list[torch.Tensor] = []
    all_sparsity: list[torch.Tensor] = []

    total = math.ceil(len(dataset) / batch_size)
    print(f"[feature_cache] Extracting features: {len(dataset)} samples, "
          f"{total} batches → {cache_path}")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            sp_levels = batch["sparsity_level"].to(device)

            sparse_model.set_sparsity_level(sp_levels.mean().item())
            out = sparse_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attention_maps=True,
                return_hidden_states=False,
            )
            features = compute_pathway_features(out["attention_maps"])  # [B, 6]
            all_features.append(features.cpu())
            all_sparsity.append(sp_levels.cpu())

            if (i + 1) % max(1, total // 10) == 0:
                print(f"  {i+1}/{total} batches done")

    features_t = torch.cat(all_features, dim=0)    # [N, 6]
    sparsity_t = torch.cat(all_sparsity, dim=0)    # [N]

    torch.save({"features": features_t, "sparsity_levels": sparsity_t}, cache_path)
    print(f"[feature_cache] Saved {len(features_t)} samples to {cache_path}")


# ---------------------------------------------------------------------------
# Cached dataset — used during training instead of the original dataset
# ---------------------------------------------------------------------------

class CachedFeaturesDataset(Dataset):
    """Pre-extracted pathway features — no LLM forward during training.

    Items: ``{'features': float32 [6], 'sparsity_level': float32 scalar}``

    The standard PyTorch default collator handles these items; no custom
    collate_fn is required.  Features are stored as a memory-mapped tensor
    via ``torch.load(..., mmap=True)`` so the OS page cache handles I/O
    without loading the full file into RAM.
    """

    def __init__(self, cache_path: str | Path):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Feature cache not found at '{cache_path}'. "
                "Run extract_and_cache() first."
            )
        # mmap=True: tensors are memory-mapped — only pages that are actually
        # accessed get loaded, so start-up is instant even for large caches.
        data = torch.load(cache_path, weights_only=True, mmap=True)
        self.features: torch.Tensor = data["features"]          # [N, 6]
        self.sparsity_levels: torch.Tensor = data["sparsity_levels"]  # [N]
        print(f"[CachedFeaturesDataset] Loaded {len(self.features)} samples "
              f"from {cache_path}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "sparsity_level": self.sparsity_levels[idx],
        }
