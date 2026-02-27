"""
EEG Image Classification Baselines for Epilepsy Diagnosis.

Trains ResNet-50 classifier on EEG montage screenshot images.
Same architecture as MRI classifier but for EEG modality.

Usage:
    python baselines/train_eeg_classifiers.py --tier gold --cv 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.train_mri_classifiers import train_image_classifier, main as _unused
from data.dataset import load_label_maps, load_classification_csv, DOWNSTREAM_TASKS


def main():
    parser = argparse.ArgumentParser(description="Train EEG Image Classification Baselines")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = "external_data/MME"
    save_dir = args.save_dir or f"logs/baselines/classifiers_{args.tier}"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(dataset_dir)

    has_eeg = df["eeg_images"].notna()
    print(f"Loaded {len(df)} patients, {has_eeg.sum()} with EEG images")

    print("\n" + "=" * 60)
    print("ResNet-50 EEG Image Classification Baselines")
    print("=" * 60)

    results = {}
    for task in DOWNSTREAM_TASKS:
        if task not in label_maps:
            continue
        result = train_image_classifier(
            df, label_maps, task, modality="eeg",
            n_splits=args.cv, device=args.device
        )
        if result:
            results[task] = result

    with open(os.path.join(save_dir, "eeg_resnet_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: EEG Image Classification")
    print("=" * 60)
    for task, r in results.items():
        print(f"  {task:<25s}: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%} ({r['n_samples']} pts)")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
