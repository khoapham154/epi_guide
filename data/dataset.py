"""
Dataset utilities for the MEAF epilepsy system.

Uses pre-consolidated classification CSVs (classification_gold.csv etc.)
with simplified 3-class label maps from label_maps.json.
No consolidation step needed — labels are already clean.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

DOWNSTREAM_TASKS = [
    "epilepsy_type", "seizure_type", "ez_localization",
    "aed_response", "surgery_outcome",
]


def load_label_maps(dataset_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Load pre-built label maps from label_maps.json.

    Returns:
        {task_name: {label_string: index}}
    """
    path = os.path.join(dataset_dir, "label_maps.json")
    with open(path, "r") as f:
        raw = json.load(f)

    label_maps = {}
    for task in DOWNSTREAM_TASKS:
        if task in raw:
            label_maps[task] = raw[task]

    return label_maps


def load_classification_csv(csv_path: str) -> pd.DataFrame:
    """Load a classification CSV with pre-consolidated labels."""
    df = pd.read_csv(csv_path)
    return df


def load_gold_dataset(dataset_dir: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Load classification_gold.csv and label_maps.json together.

    Returns:
        (dataframe, label_maps)
    """
    csv_path = os.path.join(dataset_dir, "classification_gold.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(dataset_dir)
    return df, label_maps


def get_ground_truth_labels(
    row: pd.Series,
    label_maps: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    """
    Extract ground truth label indices from a classification CSV row.

    Uses the pre-computed {task}_label_id columns directly.
    Falls back to matching {task}_label against label_maps.
    """
    gt = {}
    for task in DOWNSTREAM_TASKS:
        if task not in label_maps:
            gt[task] = -1
            continue

        # Try pre-computed label ID column first
        id_col = f"{task}_label_id"
        if id_col in row.index and pd.notna(row.get(id_col)):
            val = int(row[id_col])
            gt[task] = val
            continue

        # Fallback: match label string
        label_col = f"{task}_label"
        if label_col in row.index and pd.notna(row.get(label_col)):
            label_str = str(row[label_col]).strip()
            if label_str in label_maps[task]:
                gt[task] = label_maps[task][label_str]
            else:
                gt[task] = -1
        else:
            gt[task] = -1

    return gt


def load_mri_images(mri_images_json: str) -> List[Image.Image]:
    """Load MRI images from JSON string."""
    if pd.isna(mri_images_json):
        return []

    try:
        img_list = json.loads(mri_images_json)
    except (json.JSONDecodeError, TypeError):
        return []

    images = []
    for img_info in img_list:
        path = img_info.get("path", "")
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                continue
    return images


def load_eeg_images(eeg_images_json: str) -> List[Image.Image]:
    """Load EEG montage images from JSON string."""
    if pd.isna(eeg_images_json):
        return []

    try:
        img_list = json.loads(eeg_images_json)
    except (json.JSONDecodeError, TypeError):
        return []

    images = []
    for img_info in img_list:
        path = img_info.get("path", "")
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                continue
    return images
