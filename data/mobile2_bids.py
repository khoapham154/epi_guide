"""
Mobile-2 BIDS-native Dataset Loader.

Loads HD-EEG epochs directly from external_data/HD-EEG/ BIDS structure.
Parses stimulation metadata from JSON sidecars, looks up SEEG MNI coordinates
from TSV files, and derives labels for 3 downstream tasks:

  1. source_localization — predict 3D MNI coordinate (mm) of stimulation site
  2. ez_region           — classify brain region (Temporal/Frontal/Parieto-Occipital)
  3. stim_intensity      — classify current level (Low ≤ 0.3mA / High ≥ 0.5mA)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Label derivation helpers
# ---------------------------------------------------------------------------

def mni_to_region(x_mm: float, y_mm: float, z_mm: float) -> int:
    """Map MNI coordinate (mm) to brain region index.

    Heuristic based on observed electrode placements across 7 Mobile-2 subjects:
      0 = Temporal           — lateral/inferior regions (z < 0, or posterior-lateral low)
      1 = Frontal            — anterior/superior regions (y > -20, z > 0)
      2 = Parieto-Occipital  — posterior regions (y < -50, or posterior-superior)

    Rules applied in priority order:
      1. Very posterior (y < -50) → Parieto-Occipital
      2. Posterior-superior (y < -30, z > 30) → Parieto-Occipital
      3. Inferior (z < 0) → Temporal
      4. Posterior-lateral-low (y < -20, z < 25) → Temporal
      5. Otherwise → Frontal
    """
    if y_mm < -50:
        return 2  # Parieto-Occipital (occipital)
    if y_mm < -30 and z_mm > 30:
        return 2  # Parieto-Occipital (parietal)
    if z_mm < 0:
        return 0  # Temporal (ventral/inferior)
    if y_mm < -20 and z_mm < 25:
        return 0  # Temporal (posterior-lateral, low z)
    return 1  # Frontal (anterior/superior)


REGION_NAMES = {0: "Temporal", 1: "Frontal", 2: "Parieto-Occipital"}


def current_to_class(current_mA: float) -> int:
    """Map stimulation current to binary class.
    0 = Low (≤ 0.3 mA), 1 = High (≥ 0.5 mA).
    """
    return 0 if current_mA <= 0.3 else 1


INTENSITY_NAMES = {0: "Low", 1: "High"}


# ---------------------------------------------------------------------------
# BIDS parsing
# ---------------------------------------------------------------------------

def parse_stim_description(desc: str) -> Tuple[str, str, float]:
    """Parse 'Stimulation of channel K13-14 1mA' → (electrode1, electrode2, current_mA).

    Handles various formats:
      - 'K13-14 1mA'
      - 'X\\'1-2 0.5mA'
      - 'N10-11 5mA'
    """
    # Extract current (e.g., "1mA", "0.5mA", "5mA")
    current_match = re.search(r"([\d.]+)\s*mA", desc, re.IGNORECASE)
    if current_match:
        current_mA = float(current_match.group(1))
    else:
        current_mA = 1.0  # fallback

    # Extract channel pair (e.g., "K13-14", "X'1-2")
    # Pattern: optional prefix with apostrophe, then number-number
    chan_match = re.search(r"channel\s+([A-Za-z]+'?\d+)-(\d+)", desc)
    if chan_match:
        prefix_num = chan_match.group(1)  # e.g., "K13" or "X'1"
        second_num = chan_match.group(2)  # e.g., "14" or "2"

        # Extract the letter prefix from the first electrode
        prefix_match = re.match(r"([A-Za-z]+'?)", prefix_num)
        prefix = prefix_match.group(1) if prefix_match else ""

        electrode1 = prefix_num  # e.g., "K13"
        electrode2 = prefix + second_num  # e.g., "K14"
    else:
        electrode1 = "unknown"
        electrode2 = "unknown"

    return electrode1, electrode2, current_mA


def load_seeg_coords(tsv_path: str) -> Dict[str, np.ndarray]:
    """Load SEEG electrode MNI coordinates from TSV.

    Returns dict: electrode_name -> np.array([x_mm, y_mm, z_mm]).
    Coordinates in the TSV are in METERS — converted to mm here.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    coords = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        # Convert meters to mm
        coords[name] = np.array([
            row["x"] * 1000.0,
            row["y"] * 1000.0,
            row["z"] * 1000.0,
        ], dtype=np.float32)
    return coords


def load_hdeeg_electrode_info(tsv_path: str) -> Tuple[List[str], np.ndarray]:
    """Load HD-EEG electrode names and positions from TSV.

    Returns:
        names: list of 256 electrode names (e.g., ['e1', 'e2', ...])
        positions: (256, 3) array of (x, y, z) in meters
    """
    df = pd.read_csv(tsv_path, sep="\t")
    names = [str(n).strip() for n in df["name"].tolist()]
    positions = df[["x", "y", "z"]].values.astype(np.float32)
    return names, positions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Mobile2BIDSDataset(Dataset):
    """BIDS-native Mobile-2 dataset for 3 downstream tasks.

    Each sample is an HD-EEG epoch with task-specific labels derived from
    the known stimulation site (SEEG electrode MNI coordinates).
    """

    def __init__(
        self,
        bids_root: str = "external_data/HD-EEG",
        task: str = "source_localization",
        subjects: Optional[List[str]] = None,
        target_sr: int = 200,
        use_run_average: bool = False,
        pad_to_samples: int = 200,
    ):
        """
        Args:
            bids_root: path to Mobile-2 BIDS root
            task: one of 'source_localization', 'ez_region', 'stim_intensity'
            subjects: list of subject IDs (e.g., ['sub-01', 'sub-03']).
                      None = all 7 subjects.
            target_sr: target sample rate after downsampling (Hz)
            use_run_average: if True, average all trials per run (1 sample/run)
            pad_to_samples: zero-pad to this many samples at target_sr
        """
        self.bids_root = bids_root
        self.task = task
        self.target_sr = target_sr
        self.use_run_average = use_run_average
        self.pad_to_samples = pad_to_samples
        self.original_sr = 8000

        epochs_root = os.path.join(bids_root, "derivatives", "epochs")

        # Discover subjects
        if subjects is None:
            subjects = sorted([
                d for d in os.listdir(epochs_root)
                if d.startswith("sub-") and os.path.isdir(os.path.join(epochs_root, d))
            ])
        self.subjects = subjects

        # Parse all runs
        self.samples: List[Dict] = []
        self.electrode_names: Optional[List[str]] = None
        self.electrode_positions: Optional[np.ndarray] = None

        for sub_id in subjects:
            sub_eeg_dir = os.path.join(epochs_root, sub_id, "eeg")
            sub_ieeg_dir = os.path.join(epochs_root, sub_id, "ieeg")

            if not os.path.isdir(sub_eeg_dir):
                continue

            # Load SEEG MNI coordinates for this subject
            seeg_tsv = os.path.join(
                sub_ieeg_dir,
                f"{sub_id}_task-seegstim_space-MNI152NLin2009aSym_electrodes.tsv",
            )
            if not os.path.exists(seeg_tsv):
                print(f"  Warning: No SEEG MNI coords for {sub_id}, skipping.")
                continue
            seeg_coords = load_seeg_coords(seeg_tsv)

            # Load HD-EEG electrode info (same for all runs within a subject)
            hdeeg_tsv = os.path.join(
                sub_eeg_dir,
                f"{sub_id}_task-seegstim_electrodes.tsv",
            )
            if self.electrode_names is None and os.path.exists(hdeeg_tsv):
                self.electrode_names, self.electrode_positions = (
                    load_hdeeg_electrode_info(hdeeg_tsv)
                )

            # Find all epoch files for this subject
            epoch_files = sorted([
                f for f in os.listdir(sub_eeg_dir)
                if f.endswith("_epochs.npy")
            ])

            for epoch_file in epoch_files:
                # Derive run ID from filename
                run_match = re.search(r"run-(\d+)", epoch_file)
                run_id = f"run-{run_match.group(1)}" if run_match else "run-01"

                # Load metadata JSON
                json_file = epoch_file.replace(".npy", ".json")
                json_path = os.path.join(sub_eeg_dir, json_file)
                if not os.path.exists(json_path):
                    continue

                with open(json_path) as f:
                    meta = json.load(f)

                desc = meta.get("Description", "")
                e1_name, e2_name, current_mA = parse_stim_description(desc)

                # Look up SEEG coordinates
                if e1_name not in seeg_coords or e2_name not in seeg_coords:
                    print(f"  Warning: Electrodes {e1_name}/{e2_name} not in "
                          f"SEEG coords for {sub_id}, skipping {run_id}.")
                    continue

                midpoint_mm = (seeg_coords[e1_name] + seeg_coords[e2_name]) / 2.0
                region_label = mni_to_region(*midpoint_mm)
                intensity_label = current_to_class(current_mA)

                # Store sample info (lazy-load epochs later)
                npy_path = os.path.join(sub_eeg_dir, epoch_file)
                self.samples.append({
                    "npy_path": npy_path,
                    "subject_id": sub_id,
                    "run_id": run_id,
                    "description": desc,
                    "electrode1": e1_name,
                    "electrode2": e2_name,
                    "current_mA": current_mA,
                    "midpoint_mm": midpoint_mm,
                    "region_label": region_label,
                    "intensity_label": intensity_label,
                })

        # Expand to trial-level or keep run-level
        if use_run_average:
            # One sample per run (average ERP)
            self._trial_index = None
        else:
            # Expand: each trial within each run is a separate sample
            self._build_trial_index()

        self._print_summary()

    def _build_trial_index(self):
        """Build trial-level index by probing each npy file for n_trials."""
        self._trial_index = []
        for run_idx, s in enumerate(self.samples):
            # Peek at shape without loading full array
            epoch_shape = np.load(s["npy_path"], mmap_mode="r").shape
            n_trials = epoch_shape[0]
            for trial_idx in range(n_trials):
                self._trial_index.append((run_idx, trial_idx))

    def _print_summary(self):
        n_runs = len(self.samples)
        n_samples = len(self)
        n_subjects = len(set(s["subject_id"] for s in self.samples))

        # Label distributions
        regions = [s["region_label"] for s in self.samples]
        intensities = [s["intensity_label"] for s in self.samples]

        print(f"Mobile-2 BIDS: {n_samples} samples ({n_runs} runs) from "
              f"{n_subjects} subjects | task={self.task}")
        print(f"  Regions: {dict(zip(REGION_NAMES.values(), [regions.count(i) for i in range(3)]))}")
        print(f"  Intensity: {dict(zip(INTENSITY_NAMES.values(), [intensities.count(i) for i in range(2)]))}")

    def __len__(self) -> int:
        if self._trial_index is not None:
            return len(self._trial_index)
        return len(self.samples)

    def _load_and_process_epoch(self, npy_path: str, trial_idx: Optional[int] = None) -> np.ndarray:
        """Load epoch, optionally select trial, downsample, and pad.

        Returns: (C, pad_to_samples) float32 array at target_sr.
        """
        epochs = np.load(npy_path)  # (n_trials, 256, 2081) at 8kHz

        if trial_idx is not None:
            epoch = epochs[trial_idx]  # (256, 2081)
        else:
            # Run average
            epoch = epochs.mean(axis=0)  # (256, 2081)

        # Downsample: 8000 Hz → target_sr Hz
        factor = self.original_sr // self.target_sr
        # decimate along time axis with anti-aliasing filter
        downsampled = decimate(epoch, factor, axis=-1, zero_phase=True)
        # Result: (256, ~52)

        # Zero-pad to target length
        C, T = downsampled.shape
        if T < self.pad_to_samples:
            padded = np.zeros((C, self.pad_to_samples), dtype=np.float32)
            padded[:, :T] = downsampled
        else:
            padded = downsampled[:, :self.pad_to_samples].astype(np.float32)

        return padded

    def __getitem__(self, idx: int) -> Dict:
        if self._trial_index is not None:
            run_idx, trial_idx = self._trial_index[idx]
        else:
            run_idx = idx
            trial_idx = None

        s = self.samples[run_idx]
        eeg = self._load_and_process_epoch(s["npy_path"], trial_idx)

        # Task-specific target
        if self.task == "source_localization":
            target = torch.tensor(s["midpoint_mm"], dtype=torch.float32)
        elif self.task == "ez_region":
            target = torch.tensor(s["region_label"], dtype=torch.long)
        elif self.task == "stim_intensity":
            target = torch.tensor(s["intensity_label"], dtype=torch.long)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return {
            "raw_eeg": torch.tensor(eeg, dtype=torch.float32),
            "target": target,
            "subject_id": s["subject_id"],
            "run_id": s["run_id"],
            "current_mA": s["current_mA"],
        }


# ---------------------------------------------------------------------------
# Collate and DataLoader utilities
# ---------------------------------------------------------------------------

def mobile2_bids_collate_fn(batch: List[Dict]) -> Dict:
    """Collate Mobile-2 BIDS samples."""
    return {
        "raw_eeg": torch.stack([s["raw_eeg"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
        "subject_id": [s["subject_id"] for s in batch],
        "run_id": [s["run_id"] for s in batch],
        "current_mA": [s["current_mA"] for s in batch],
    }


def get_loso_splits(
    bids_root: str = "external_data/HD-EEG",
) -> List[Tuple[List[str], List[str]]]:
    """Generate leave-one-subject-out CV splits.

    Returns list of (train_subjects, test_subjects) tuples.
    """
    epochs_root = os.path.join(bids_root, "derivatives", "epochs")
    all_subjects = sorted([
        d for d in os.listdir(epochs_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(epochs_root, d))
    ])
    splits = []
    for test_sub in all_subjects:
        train_subs = [s for s in all_subjects if s != test_sub]
        splits.append((train_subs, [test_sub]))
    return splits


def create_mobile2_bids_loaders(
    bids_root: str,
    task: str,
    train_subjects: List[str],
    test_subjects: List[str],
    batch_size: int = 32,
    target_sr: int = 200,
    use_run_average: bool = False,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders for a single LOSO fold."""
    train_ds = Mobile2BIDSDataset(
        bids_root=bids_root,
        task=task,
        subjects=train_subjects,
        target_sr=target_sr,
        use_run_average=use_run_average,
    )
    test_ds = Mobile2BIDSDataset(
        bids_root=bids_root,
        task=task,
        subjects=test_subjects,
        target_sr=target_sr,
        use_run_average=use_run_average,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mobile2_bids_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_ds) > batch_size,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mobile2_bids_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
