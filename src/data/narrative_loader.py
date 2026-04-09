"""Load narrative EEG trial-info CSVs as a sparsity dataset.

Each trial's ``prompt_with_condition`` is used as the LLM input text and
the participant's ``vividness_rating`` is normalised to [0.05, 0.95] to
govern attention sparsity in the experiment.  This lets the LLM's sparse-
attention pathway metrics vary with the same variable (imagery vividness)
that is hypothesised to modulate real EEG structure.

CSV file naming convention (from NarrativeEEGDataset):
    <data_dir>/<subj>/<subj>_<task>_trialinfo_aligned.csv   (preferred)
    <data_dir>/<subj>/<subj>_<task>_trialinfo.csv           (fallback)

Required columns: ``prompt_with_condition``, ``vividness_rating``
Optional columns: ``condition``, ``prompt``, ``order``
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_trial_info_csvs(data_dir: str | Path) -> List[Path]:
    """Return all trialinfo CSV paths found under *data_dir* (recursive)."""
    data_dir = Path(data_dir)
    # Prefer aligned variant; glob both suffixes and deduplicate by stem
    patterns = ["**/*_trialinfo_aligned.csv", "**/*_trialinfo.csv"]
    seen_stems: set[str] = set()
    paths: List[Path] = []
    for pattern in patterns:
        for p in sorted(data_dir.glob(pattern)):
            # Use the base recording ID (strip _aligned suffix) as dedup key
            stem = p.stem.replace("_aligned", "")
            if stem not in seen_stems:
                seen_stems.add(stem)
                paths.append(p)
    return paths


def load_narrative_records(data_dir: str | Path) -> Tuple[List[str], List[float]]:
    """Scan *data_dir* for trial-info CSVs and return (prompts, vividness).

    Returns:
        prompts: list of ``prompt_with_condition`` strings (one per trial).
        raw_vividness: list of raw vividness float values (NaN entries dropped).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load narrative CSVs") from exc

    all_prompts: List[str] = []
    all_vividness: List[float] = []

    csv_paths = _find_trial_info_csvs(data_dir)
    if not csv_paths:
        raise FileNotFoundError(
            f"No trialinfo CSV files found under '{data_dir}'. "
            "Expected paths matching **/*_trialinfo[_aligned].csv"
        )

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # Sort by presentation order when available so rows match EEG trial order
        if "order" in df.columns:
            df = df.sort_values("order").reset_index(drop=True)

        if "prompt_with_condition" not in df.columns:
            continue  # skip malformed files

        for _, row in df.iterrows():
            prompt = str(row["prompt_with_condition"]).strip()
            if not prompt or prompt.lower() in ("nan", "none", ""):
                continue

            vividness = float(row.get("vividness_rating", float("nan")))
            if np.isnan(vividness):
                continue  # skip trials without a vividness rating

            all_prompts.append(prompt)
            all_vividness.append(vividness)

    if not all_prompts:
        raise ValueError(
            f"No valid (prompt, vividness) pairs found in CSVs under '{data_dir}'."
        )

    return all_prompts, all_vividness


def vividness_to_sparsity(
    vividness: List[float],
    low: float = 0.05,
    high: float = 0.95,
) -> List[float]:
    """Min-max normalise *vividness* values into [low, high].

    Higher vividness → higher sparsity (more focused attention routing).
    The mapping is monotone; the resulting values become the per-trial
    sparsity levels fed into SparseAttentionWrapper.
    """
    arr = np.array(vividness, dtype=float)
    v_min, v_max = arr.min(), arr.max()
    if v_max == v_min:
        # All ratings identical — centre at midpoint
        return [0.5] * len(vividness)
    normalised = (arr - v_min) / (v_max - v_min)  # [0, 1]
    scaled = low + normalised * (high - low)       # [low, high]
    return scaled.tolist()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NarrativeSparsityDataset(Dataset):
    """Dataset built from narrative trial-info CSVs.

    Each sample is one EEG trial record with:
        text          — ``prompt_with_condition`` string fed to the LLM
        sparsity_level — vividness rating normalised to [0.05, 0.95]
        prompt_idx     — index into the flat prompt list (for compatibility)
        vividness_raw  — raw (unnormalised) vividness rating
    """

    def __init__(self, data_dir: str | Path):
        prompts, raw_vividness = load_narrative_records(data_dir)
        sparsity = vividness_to_sparsity(raw_vividness)

        self.prompts = prompts
        self.raw_vividness = raw_vividness
        self.sparsity_levels = sparsity

        print(
            f"NarrativeSparsityDataset: {len(prompts)} trials loaded from '{data_dir}'. "
            f"Sparsity range: [{min(sparsity):.3f}, {max(sparsity):.3f}]"
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.prompts[idx],
            "sparsity_level": torch.tensor(self.sparsity_levels[idx], dtype=torch.float32),
            "prompt_idx": idx,
            "vividness_raw": self.raw_vividness[idx],
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_narrative_dataloader(
    data_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader from narrative trial-info CSVs.

    Args:
        data_dir: Root directory containing per-subject subdirectories with
            ``*_trialinfo[_aligned].csv`` files.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle the dataset.
        num_workers: DataLoader worker count.
        collate_fn: Optional collate function (e.g. from ``get_collate_fn``).
            When *None* the default PyTorch collation is used; you will
            typically want to pass ``get_collate_fn(model_name=...)`` so that
            text strings are tokenised inside the DataLoader.

    Returns:
        DataLoader instance.
    """
    dataset = NarrativeSparsityDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
