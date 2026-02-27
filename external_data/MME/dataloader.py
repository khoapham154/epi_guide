#!/usr/bin/env python3
"""
EpilepsyMultimodalDataset - PyTorch-compatible dataset loader.

Usage:
    from dataloader import EpilepsyMultimodalDataset, get_dataloaders

    # Load all tiers
    dataset = EpilepsyMultimodalDataset(tier='all')

    # Load only gold tier
    gold = EpilepsyMultimodalDataset(tier='gold')

    # Load for a specific task
    ez_dataset = EpilepsyMultimodalDataset(tier='all', task='ez_localization')

    # Get train/val/test splits
    train_loader, val_loader, test_loader = get_dataloaders(
        tier='gold', task='epilepsy_type', batch_size=8
    )

Each sample is a dict:
{
    'patient_id': str,
    'text': {
        'semiology': str or None,
        'mri_report': str or None,
        'eeg_report': str or None,
        'demographics': str or None,
        'raw_facts': str or None,
    },
    'mri_images': [{'path': str, 'modality': str, 'subcaption': str}, ...],
    'eeg_images': [{'path': str, 'modality': str, 'subcaption': str}, ...],
    'linked_images': [{'path': str, 'caption': str}, ...],
    'labels': {
        'epilepsy_type': str or None,
        'seizure_type': str or None,
        'ez_localization': str or None,
        'aed_response': str or None,
        'surgery_outcome': str or None,
    },
    'label_ids': {
        'epilepsy_type': int,   # 0-2 or -1
        'seizure_type': int,
        ...
    },
    'redacted_inputs': {
        'epilepsy_type_redacted': str or None,
        ...
    },
    'metadata': {
        'pmc_id': str,
        'quality_tier': str,
        'keyword': str,
        'age': float or None,
        'sex': str or None,
    }
}
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


DATASET_DIR = Path(__file__).parent
DOWNSTREAM_TASKS = [
    'epilepsy_type', 'seizure_type', 'ez_localization',
    'aed_response', 'surgery_outcome',
]


class EpilepsyMultimodalDataset:
    """
    Multimodal epilepsy dataset with MRI images, EEG images, and clinical text.

    Args:
        tier: Quality tier filter - 'gold', 'silver', 'bronze', or 'all'
        task: Optional downstream task filter - only include patients with this label
        load_images: Whether to load image pixels (requires PIL)
        image_transform: Optional transform to apply to loaded images
        max_images_per_modality: Cap number of images per patient per modality
    """

    def __init__(
        self,
        tier: str = 'all',
        task: Optional[str] = None,
        load_images: bool = False,
        image_transform=None,
        max_images_per_modality: int = 5,
    ):
        self.tier = tier.upper() if tier != 'all' else 'all'
        self.task = task
        self.load_images = load_images
        self.image_transform = image_transform
        self.max_images_per_modality = max_images_per_modality

        # Load data
        if self.tier == 'all':
            self.df = pd.read_csv(DATASET_DIR / 'multimodal_dataset.csv')
        else:
            self.df = pd.read_csv(DATASET_DIR / f'tier_{self.tier.lower()}.csv')

        # Filter by task if specified
        if self.task:
            truth_col = f'{self.task}_truth'
            self.df = self.df[self.df[truth_col].notna()].reset_index(drop=True)

        print(f"Loaded {len(self.df)} patients (tier={tier}, task={task})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self._build_sample(row)

    def _parse_images(self, json_str):
        """Parse JSON image list, cap at max_images_per_modality."""
        if pd.isna(json_str):
            return []
        imgs = json.loads(json_str)
        return imgs[:self.max_images_per_modality]

    def _load_image(self, path):
        """Load an image from disk."""
        if not HAS_PIL:
            raise ImportError("PIL required for image loading: pip install Pillow")
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert('RGB')
            if self.image_transform:
                img = self.image_transform(img)
            return img
        except Exception:
            return None

    def _build_sample(self, row):
        """Build a single sample dict from a dataframe row."""
        # Parse images
        mri_imgs = self._parse_images(row.get('mri_images'))
        eeg_imgs = self._parse_images(row.get('eeg_images'))
        linked_imgs = self._parse_images(row.get('linked_images'))

        # Optionally load image pixels
        if self.load_images:
            for img in mri_imgs:
                img['pixel_data'] = self._load_image(img['path'])
            for img in eeg_imgs:
                img['pixel_data'] = self._load_image(img['path'])
            for img in linked_imgs:
                img['pixel_data'] = self._load_image(img['path'])

        # Detect column format: classification (_label/_label_id) vs legacy (_truth)
        use_clf = f'{DOWNSTREAM_TASKS[0]}_label' in row.index

        if use_clf:
            labels = {
                task: row.get(f'{task}_label') if pd.notna(row.get(f'{task}_label')) else None
                for task in DOWNSTREAM_TASKS
            }
            label_ids = {
                task: int(row.get(f'{task}_label_id', -1))
                for task in DOWNSTREAM_TASKS
            }
        else:
            labels = {
                task: row.get(f'{task}_truth') if pd.notna(row.get(f'{task}_truth')) else None
                for task in DOWNSTREAM_TASKS
            }
            label_ids = {task: -1 for task in DOWNSTREAM_TASKS}

        sample = {
            'patient_id': row['patient_id'],
            'text': {
                'semiology': row.get('semiology_text') if pd.notna(row.get('semiology_text')) else None,
                'mri_report': row.get('mri_report_text') if pd.notna(row.get('mri_report_text')) else None,
                'eeg_report': row.get('eeg_report_text') if pd.notna(row.get('eeg_report_text')) else None,
                'demographics': row.get('demographics_notes') if pd.notna(row.get('demographics_notes')) else None,
                'raw_facts': row.get('raw_facts') if pd.notna(row.get('raw_facts')) else None,
            },
            'mri_images': mri_imgs,
            'eeg_images': eeg_imgs,
            'linked_images': linked_imgs,
            'labels': labels,
            'label_ids': label_ids,
            'redacted_inputs': {
                f'{task}_redacted': row.get(f'{task}_redacted') if pd.notna(row.get(f'{task}_redacted')) else None
                for task in DOWNSTREAM_TASKS
            },
            'metadata': {
                'pmc_id': row['pmc_id'],
                'quality_tier': row['quality_tier'],
                'keyword': row['keyword'],
                'age': row['age'] if pd.notna(row.get('age')) else None,
                'sex': row.get('sex') if pd.notna(row.get('sex')) else None,
            }
        }
        return sample

    def get_task_labels(self, task: str):
        """Get all unique labels for a downstream task."""
        col = f'{task}_truth'
        return self.df[col].dropna().unique().tolist()

    def get_statistics(self):
        """Return dataset statistics."""
        stats = {
            'total_patients': len(self.df),
            'tier_distribution': self.df['quality_tier'].value_counts().to_dict(),
        }
        for task in DOWNSTREAM_TASKS:
            stats[f'{task}_count'] = int(self.df[f'{task}_truth'].notna().sum())
        stats['patients_with_mri_images'] = int((self.df['num_mri_subfigures'] > 0).sum())
        stats['patients_with_eeg_images'] = int((self.df['num_eeg_subfigures'] > 0).sum())
        stats['patients_with_mri_text'] = int(self.df['has_mri_text'].sum())
        stats['patients_with_eeg_text'] = int(self.df['has_eeg_text'].sum())
        return stats

    def to_huggingface(self):
        """Convert to HuggingFace Dataset format."""
        try:
            from datasets import Dataset as HFDataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        records = [self._build_sample(self.df.iloc[i]) for i in range(len(self))]
        return HFDataset.from_list(records)


def get_dataloaders(
    tier: str = 'all',
    task: Optional[str] = None,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    load_images: bool = False,
    image_transform=None,
    num_workers: int = 0,
) -> Tuple:
    """
    Get train/val/test DataLoaders with stratified PMC-level splits.

    Splits are done at PMC level so no paper appears in multiple splits.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for DataLoaders: pip install torch")

    dataset = EpilepsyMultimodalDataset(
        tier=tier, task=task, load_images=load_images,
        image_transform=image_transform
    )

    # Split by PMC ID to avoid data leakage
    pmc_ids = dataset.df['pmc_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(pmc_ids)

    n_train = int(len(pmc_ids) * train_ratio)
    n_val = int(len(pmc_ids) * val_ratio)

    train_pmcs = set(pmc_ids[:n_train])
    val_pmcs = set(pmc_ids[n_train:n_train + n_val])
    test_pmcs = set(pmc_ids[n_train + n_val:])

    train_idx = dataset.df[dataset.df['pmc_id'].isin(train_pmcs)].index.tolist()
    val_idx = dataset.df[dataset.df['pmc_id'].isin(val_pmcs)].index.tolist()
    test_idx = dataset.df[dataset.df['pmc_id'].isin(test_pmcs)].index.tolist()

    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    def collate_fn(batch):
        """Custom collate that keeps dicts as-is (no tensor stacking)."""
        return batch

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)

    print(f"Splits (PMC-level, seed={seed}):")
    print(f"  Train: {len(train_idx)} patients ({len(train_pmcs)} papers)")
    print(f"  Val:   {len(val_idx)} patients ({len(val_pmcs)} papers)")
    print(f"  Test:  {len(test_idx)} patients ({len(test_pmcs)} papers)")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test
    ds = EpilepsyMultimodalDataset(tier='all')
    print(f"\nDataset statistics:")
    stats = ds.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nSample patient:")
    sample = ds[0]
    print(f"  ID: {sample['patient_id']}")
    print(f"  Tier: {sample['metadata']['quality_tier']}")
    print(f"  MRI images: {len(sample['mri_images'])}")
    print(f"  EEG images: {len(sample['eeg_images'])}")
    print(f"  Labels: { {k:v for k,v in sample['labels'].items() if v} }")
    print(f"  Text keys with data: {[k for k,v in sample['text'].items() if v]}")
