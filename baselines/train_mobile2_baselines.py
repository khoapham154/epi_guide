#!/usr/bin/env python3
"""
Classical Baselines for Mobile-2 Downstream Tasks.

Three feature-extraction + sklearn baselines evaluated with the same
leave-one-subject-out CV as the REVE model for fair comparison:

  1. GFP Features + XGBoost
  2. Band Power Features + XGBoost
  3. LaBraM (from-scratch training, existing Mobile2Model)

Usage:
    python baselines/train_mobile2_baselines.py --save_dir logs/baselines/mobile2/
    python baselines/train_mobile2_baselines.py --save_dir logs/baselines/mobile2/ --device cuda:6
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.mobile2_bids import (
    Mobile2BIDSDataset,
    get_loso_splits,
    REGION_NAMES,
    INTENSITY_NAMES,
)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def extract_gfp_features(eeg: np.ndarray, sr: int = 200) -> np.ndarray:
    """Extract Global Field Power features from an HD-EEG epoch.

    Args:
        eeg: (C, T) at sr Hz
        sr: sample rate

    Returns:
        Feature vector (9,):
          - GFP peak latency (ms)
          - GFP peak amplitude
          - GFP mean amplitude
          - GFP AUC in 3 time windows (early/mid/late)
          - Channel index of max amplitude at GFP peak
          - Std of channel amplitudes at GFP peak
          - Kurtosis of GFP
    """
    # GFP = std across channels at each time point
    gfp = eeg.std(axis=0)  # (T,)
    time_ms = np.arange(len(gfp)) / sr * 1000  # in ms

    peak_idx = np.argmax(gfp)
    peak_latency_ms = time_ms[peak_idx]
    peak_amplitude = gfp[peak_idx]
    mean_amplitude = gfp.mean()

    # AUC in 3 windows (defined relative to signal length)
    T = len(gfp)
    w1 = slice(0, min(T, int(10 * sr / 1000)))   # 0-10ms
    w2 = slice(int(10 * sr / 1000), min(T, int(50 * sr / 1000)))   # 10-50ms
    w3 = slice(int(50 * sr / 1000), T)            # 50ms-end

    auc_early = np.trapz(gfp[w1]) if w1.stop > w1.start else 0.0
    auc_mid = np.trapz(gfp[w2]) if w2.stop > w2.start else 0.0
    auc_late = np.trapz(gfp[w3]) if w3.stop > w3.start else 0.0

    # Channel with max amplitude at GFP peak
    max_channel = np.argmax(np.abs(eeg[:, peak_idx])) if peak_idx < eeg.shape[1] else 0
    ch_std = eeg[:, peak_idx].std() if peak_idx < eeg.shape[1] else 0.0

    # Kurtosis of GFP
    gfp_mean = gfp.mean()
    gfp_std = gfp.std() + 1e-8
    kurtosis = ((gfp - gfp_mean) ** 4).mean() / (gfp_std ** 4) - 3.0

    return np.array([
        peak_latency_ms, peak_amplitude, mean_amplitude,
        auc_early, auc_mid, auc_late,
        float(max_channel), ch_std, kurtosis,
    ], dtype=np.float32)


def extract_band_power_features(eeg: np.ndarray, sr: int = 200) -> np.ndarray:
    """Extract per-channel band power features.

    Args:
        eeg: (C, T) at sr Hz

    Returns:
        Feature vector (C * 5,) = (1280,) for 256 channels × 5 bands.
        Bands: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-45).
    """
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    C, T = eeg.shape

    nperseg = min(T, sr)  # 1 second or full signal if shorter
    features = []

    for ch in range(C):
        freqs, psd = welch(eeg[ch], fs=sr, nperseg=nperseg)
        total_power = np.trapz(psd, freqs) + 1e-10
        for low, high in bands:
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.trapz(psd[mask], freqs[mask])
            features.append(band_power / total_power)  # relative power

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def run_xgboost_baseline(
    feature_extractor,
    feat_name: str,
    task: str,
    bids_root: str,
    save_dir: str,
):
    """Run XGBoost baseline with LOSO CV."""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        print(f"  XGBoost not installed, skipping {feat_name} baseline.")
        return None

    splits = get_loso_splits(bids_root)
    all_results = []

    for fold, (train_subs, test_subs) in enumerate(splits):
        print(f"    Fold {fold}: test={test_subs[0]}")

        # Build datasets (run-averaged for all baselines)
        train_ds = Mobile2BIDSDataset(
            bids_root=bids_root, task=task, subjects=train_subs,
            use_run_average=True,
        )
        test_ds = Mobile2BIDSDataset(
            bids_root=bids_root, task=task, subjects=test_subs,
            use_run_average=True,
        )

        if len(train_ds) == 0 or len(test_ds) == 0:
            continue

        # Extract features
        X_train, y_train = [], []
        for i in range(len(train_ds)):
            sample = train_ds[i]
            eeg = sample["raw_eeg"].numpy()
            feat = feature_extractor(eeg)
            X_train.append(feat)
            y_train.append(sample["target"].item())

        X_test, y_test = [], []
        for i in range(len(test_ds)):
            sample = test_ds[i]
            eeg = sample["raw_eeg"].numpy()
            feat = feature_extractor(eeg)
            X_test.append(feat)
            y_test.append(sample["target"].item())

        X_train = np.stack(X_train)
        X_test = np.stack(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Train model
        if task == "source_localization":
            # 3 separate regressors for x, y, z
            preds = np.zeros((len(X_test), 3))
            for dim in range(3):
                model = XGBRegressor(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    verbosity=0,
                )
                model.fit(X_train, y_train[:, dim] if y_train.ndim > 1 else y_train)
                preds[:, dim] = model.predict(X_test)

            # For source loc, y_test has 3D targets
            # But our dataset returns scalar targets for each dim...
            # Actually, for XGBoost source loc we need to extract 3D targets
            # This requires reloading with target as full 3D vector
            # For simplicity, skip source loc for XGBoost (classical methods
            # like sLORETA are more appropriate baselines for regression)
            result = {"fold": fold, "test_subject": test_subs[0], "skipped": True}
        else:
            model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                eval_metric="mlogloss", verbosity=0,
                use_label_encoder=False,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            result = {
                "fold": fold,
                "test_subject": test_subs[0],
                "accuracy": acc,
                "f1_macro": f1,
                "n_test": len(y_test),
            }
            print(f"      Acc: {acc:.1%} | F1: {f1:.3f}")

        all_results.append(result)

    # Aggregate
    if not all_results:
        return None

    if task != "source_localization":
        accs = [r["accuracy"] for r in all_results if "accuracy" in r]
        f1s = [r["f1_macro"] for r in all_results if "f1_macro" in r]
        aggregate = {
            "accuracy": f"{np.mean(accs):.3f} ± {np.std(accs):.3f}" if accs else "N/A",
            "f1_macro": f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}" if f1s else "N/A",
        }
    else:
        aggregate = {"note": "Skipped for XGBoost (use sLORETA for source loc)"}

    output = {
        "method": feat_name,
        "task": task,
        "per_fold": all_results,
        "aggregate": aggregate,
    }

    fname = f"{feat_name.lower().replace(' ', '_').replace('+', '_')}_{task}_results.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"    Saved: {fname}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Mobile-2 classical baselines")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.default import Config
    config = Config()

    save_dir = args.save_dir or os.path.join(
        config.baseline.save_dir, "mobile2_baselines"
    )
    os.makedirs(save_dir, exist_ok=True)

    bids_root = config.mobile2_bids.bids_root

    print("=" * 60)
    print("Mobile-2 Classical Baselines")
    print("=" * 60)
    print(f"BIDS root: {bids_root}")
    print(f"Save dir:  {save_dir}")

    # Classification tasks only for XGBoost baselines
    # (Source localization needs specialized baselines like sLORETA)
    tasks = ["ez_region", "stim_intensity"]

    baselines = [
        (extract_gfp_features, "GFP_XGBoost"),
        (extract_band_power_features, "BandPower_XGBoost"),
    ]

    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Task: {task}")
        print(f"{'='*40}")

        for extractor, name in baselines:
            print(f"\n  --- {name} ---")
            t0 = time.time()
            run_xgboost_baseline(
                feature_extractor=extractor,
                feat_name=name,
                task=task,
                bids_root=bids_root,
                save_dir=save_dir,
            )
            print(f"  Elapsed: {time.time() - t0:.0f}s")

    print(f"\n{'='*60}")
    print(f"ALL BASELINES DONE. Results in: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
