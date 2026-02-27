"""
Mobile-2 HD-EEG Source Localization Dataset.

Loads HD-EEG epochs (256 channels, 8kHz), downsamples to 200Hz for LaBraM
compatibility, and provides ground truth SEEG electrode coordinates in MNI space.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Mobile2Dataset(Dataset):
    """
    Dataset for Mobile-2 HD-EEG source localization.

    Each sample is an HD-EEG epoch with:
      - raw_eeg: (C, T) at target_sample_rate (200Hz)
      - channel_ids: (C,) channel indices
      - seeg_coords: (3,) MNI x, y, z of stimulated SEEG electrode
      - subject_id: int subject index
    """

    def __init__(
        self,
        data_dir: str,
        original_sample_rate: int = 8000,
        target_sample_rate: int = 200,
        num_channels: int = 256,
        epoch_samples: int = 2081,
        subjects: Optional[List[int]] = None,
    ):
        self.data_dir = data_dir
        self.original_sr = original_sample_rate
        self.target_sr = target_sample_rate
        self.num_channels = num_channels
        self.downsample_factor = original_sample_rate // target_sample_rate

        self.epochs = []
        self.coords = []
        self.subject_ids = []

        # Load data per subject
        subject_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]) if os.path.exists(data_dir) else []

        for subj_idx, subj_dir in enumerate(subject_dirs):
            if subjects is not None and subj_idx not in subjects:
                continue

            subj_path = os.path.join(data_dir, subj_dir)

            # Look for epoch files (numpy format)
            epoch_file = os.path.join(subj_path, "epochs.npy")
            coord_file = os.path.join(subj_path, "seeg_coords.npy")

            if os.path.exists(epoch_file) and os.path.exists(coord_file):
                epochs = np.load(epoch_file)  # (N, C, T) at original_sr
                coords = np.load(coord_file)  # (N, 3) MNI coordinates

                for i in range(len(epochs)):
                    self.epochs.append(epochs[i])
                    self.coords.append(coords[i])
                    self.subject_ids.append(subj_idx)

        print(f"Mobile-2 dataset: {len(self.epochs)} epochs from {len(set(self.subject_ids))} subjects")

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx: int) -> Dict:
        epoch = self.epochs[idx]  # (C, T) at original_sr
        coords = self.coords[idx]  # (3,)

        # Downsample: simple decimation
        if self.downsample_factor > 1:
            epoch = epoch[:, ::self.downsample_factor]

        # Convert to tensors
        raw_eeg = torch.tensor(epoch, dtype=torch.float32)
        seeg_coords = torch.tensor(coords, dtype=torch.float32)
        channel_ids = torch.arange(raw_eeg.shape[0], dtype=torch.long)

        return {
            "raw_eeg": raw_eeg,
            "channel_ids": channel_ids,
            "seeg_coords": seeg_coords,
            "subject_id": self.subject_ids[idx],
        }


def mobile2_collate_fn(batch: List[Dict]) -> Dict:
    """Collate Mobile-2 samples."""
    return {
        "raw_eeg": torch.stack([s["raw_eeg"] for s in batch]),
        "channel_ids": torch.stack([s["channel_ids"] for s in batch]),
        "seeg_coords": torch.stack([s["seeg_coords"] for s in batch]),
        "subject_id": [s["subject_id"] for s in batch],
    }


def create_mobile2_loaders(
    data_dir: str,
    n_subjects: int = 38,
    leave_out_subject: int = 0,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/test loaders with leave-one-subject-out split.

    Args:
        data_dir: path to Mobile-2 data
        n_subjects: total number of subjects
        leave_out_subject: subject index to hold out for testing
        batch_size: batch size
        num_workers: data loading workers

    Returns:
        (train_loader, test_loader)
    """
    all_subjects = list(range(n_subjects))
    train_subjects = [s for s in all_subjects if s != leave_out_subject]
    test_subjects = [leave_out_subject]

    train_dataset = Mobile2Dataset(data_dir, subjects=train_subjects, **kwargs)
    test_dataset = Mobile2Dataset(data_dir, subjects=test_subjects, **kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mobile2_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=mobile2_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
