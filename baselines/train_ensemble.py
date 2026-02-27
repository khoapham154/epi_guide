"""
Late-Fusion Ensemble Baseline.

Combines predictions from text, MRI, and EEG classifiers.
Methods: weighted average, stacking (logistic regression meta-learner).

Usage:
    python baselines/train_ensemble.py --tier gold
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_label_maps, load_classification_csv, DOWNSTREAM_TASKS


def load_classifier_results(results_dir):
    """Load OOF predictions from all trained classifiers."""
    classifier_files = {
        "tfidf_xgboost": "tfidf_xgboost_results.json",
        "pubmedbert": "pubmedbert_results.json",
        "mri_resnet": "mri_resnet_results.json",
        "eeg_resnet": "eeg_resnet_results.json",
    }

    results = {}
    for name, filename in classifier_files.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  Loaded {name}: {list(results[name].keys())}")
        else:
            print(f"  [MISSING] {name}: {path}")

    return results


def expand_oof_to_full(oof_probs, valid_mask, n_patients, n_classes):
    """Expand OOF predictions from valid-only indices back to full patient array.

    Args:
        oof_probs: array of shape (n_valid, n_classes) with OOF probabilities
        valid_mask: boolean array of shape (n_patients,) indicating which patients are valid
        n_patients: total number of patients
        n_classes: number of classes

    Returns:
        full_probs: array of shape (n_patients, n_classes) with uniform for invalid patients
        has_pred: boolean array of shape (n_patients,) indicating which patients have predictions
    """
    full_probs = np.full((n_patients, n_classes), 1.0 / n_classes)
    has_pred = np.zeros(n_patients, dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    n_valid = min(len(oof_probs), len(valid_indices))
    full_probs[valid_indices[:n_valid]] = oof_probs[:n_valid]
    has_pred[valid_indices[:n_valid]] = True
    return full_probs, has_pred


def get_valid_mask_for_classifier(clf_name, df, task):
    """Determine which patients a classifier produced predictions for."""
    id_col = f"{task}_label_id"
    has_labels = df[id_col].notna() & (df[id_col] >= 0) if id_col in df.columns else pd.Series(False, index=df.index)

    if clf_name in ("tfidf_xgboost", "pubmedbert"):
        # Text classifiers: valid = has text (any) AND has label
        has_text = (
            df["semiology_text"].notna() |
            df["mri_report_text"].notna() |
            df["eeg_report_text"].notna()
        )
        return (has_text & has_labels).values
    elif clf_name == "mri_resnet":
        has_images = df["mri_images"].notna()
        return (has_images & has_labels).values
    elif clf_name == "eeg_resnet":
        has_images = df["eeg_images"].notna()
        return (has_images & has_labels).values
    else:
        return has_labels.values


def weighted_average_ensemble(classifier_results, df, task, n_classes, n_patients):
    """Weighted average of prediction probabilities across classifiers."""
    all_probs = []
    all_weights = []
    all_has_pred = []

    for clf_name, clf_results in classifier_results.items():
        if task not in clf_results:
            continue
        task_result = clf_results[task]
        oof_probs = np.array(task_result["oof_probabilities"])
        acc = task_result["mean_accuracy"]

        valid_mask = get_valid_mask_for_classifier(clf_name, df, task)
        full_probs, has_pred = expand_oof_to_full(oof_probs, valid_mask, n_patients, n_classes)

        all_probs.append(full_probs)
        all_weights.append(acc)
        all_has_pred.append(has_pred)

    if not all_probs:
        return None, None

    # Weighted average (only where predictions exist)
    ensemble_probs = np.full((n_patients, n_classes), 1.0 / n_classes)
    for i in range(n_patients):
        # Collect classifiers that have predictions for this patient
        patient_probs = []
        patient_weights = []
        for j in range(len(all_probs)):
            if all_has_pred[j][i]:
                patient_probs.append(all_probs[j][i])
                patient_weights.append(all_weights[j])

        if patient_probs:
            total_w = sum(patient_weights)
            weighted = sum(w / total_w * p for w, p in zip(patient_weights, patient_probs))
            ensemble_probs[i] = weighted

    ensemble_preds = ensemble_probs.argmax(axis=1)
    return ensemble_preds, ensemble_probs


def stacking_ensemble(classifier_results, df, task, labels, n_classes, n_patients, n_splits=5):
    """Stacking with logistic regression meta-learner."""
    # Gather features: concatenated probabilities from all classifiers
    features = []
    available_clfs = []

    for clf_name, clf_results in classifier_results.items():
        if task not in clf_results:
            continue
        task_result = clf_results[task]
        oof_probs = np.array(task_result["oof_probabilities"])

        valid_mask = get_valid_mask_for_classifier(clf_name, df, task)
        full_probs, _ = expand_oof_to_full(oof_probs, valid_mask, n_patients, n_classes)

        features.append(full_probs)
        available_clfs.append(clf_name)

    if not features:
        return None, None

    X = np.concatenate(features, axis=1)
    y = labels

    valid_mask = y >= 0
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if len(y_valid) < n_splits:
        return None, None

    class_counts = Counter(y_valid)
    min_class = min(class_counts.values())
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        return None, None

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    all_preds = np.full(len(y_valid), -1)
    all_probs = np.zeros((len(y_valid), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_valid, y_valid)):
        meta_clf = LogisticRegression(
            C=1.0, max_iter=1000, multi_class="multinomial",
            solver="lbfgs", random_state=42
        )
        meta_clf.fit(X_valid[train_idx], y_valid[train_idx])
        all_preds[val_idx] = meta_clf.predict(X_valid[val_idx])
        all_probs[val_idx] = meta_clf.predict_proba(X_valid[val_idx])

    # Map back to full patient array
    full_preds = np.full(n_patients, -1)
    full_probs = np.full((n_patients, n_classes), 1.0 / n_classes)
    valid_indices = np.where(valid_mask)[0]
    full_preds[valid_indices] = all_preds
    full_probs[valid_indices] = all_probs

    return full_preds, full_probs


def main():
    parser = argparse.ArgumentParser(description="Late-Fusion Ensemble Baseline")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = "external_data/MME"
    results_dir = args.save_dir or f"logs/baselines/classifiers_{args.tier}"
    save_dir = results_dir
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(dataset_dir)
    n_patients = len(df)

    print(f"Loaded {n_patients} patients")
    print(f"\nLoading classifier results from {results_dir}...")
    classifier_results = load_classifier_results(results_dir)

    if not classifier_results:
        print("No classifier results found. Train individual classifiers first.")
        return

    ensemble_results = {}

    for task in DOWNSTREAM_TASKS:
        if task not in label_maps:
            continue

        inv_map = {v: k for k, v in label_maps[task].items()}
        label_names = [inv_map[i] for i in range(len(inv_map))]
        n_classes = len(label_names)

        # Get ground truth labels
        id_col = f"{task}_label_id"
        if id_col in df.columns:
            labels = df[id_col].fillna(-1).values.astype(int)
        else:
            labels = np.full(n_patients, -1)

        print(f"\n--- {task} ---")

        # Weighted average
        wa_preds, wa_probs = weighted_average_ensemble(
            classifier_results, df, task, n_classes, n_patients
        )

        if wa_preds is not None:
            valid = labels >= 0
            if valid.sum() > 0:
                wa_acc = accuracy_score(labels[valid], wa_preds[valid])
                print(f"  Weighted Average: {wa_acc:.3f}")
            else:
                wa_acc = 0.0
        else:
            wa_acc = 0.0
            wa_probs = None

        # Stacking
        st_preds, st_probs = stacking_ensemble(
            classifier_results, df, task, labels, n_classes, n_patients, n_splits=args.cv
        )

        if st_preds is not None:
            valid = (labels >= 0) & (st_preds >= 0)
            if valid.sum() > 0:
                st_acc = accuracy_score(labels[valid], st_preds[valid])
                print(f"  Stacking:         {st_acc:.3f}")
            else:
                st_acc = 0.0
        else:
            st_acc = 0.0
            st_probs = None

        ensemble_results[task] = {
            "weighted_average_accuracy": float(wa_acc),
            "stacking_accuracy": float(st_acc),
            "weighted_average_probabilities": wa_probs.tolist() if wa_probs is not None else None,
            "stacking_probabilities": st_probs.tolist() if st_probs is not None else None,
            "n_patients": n_patients,
            "label_names": label_names,
        }

    # Save
    with open(os.path.join(save_dir, "ensemble_results.json"), "w") as f:
        json.dump(ensemble_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Ensemble Baselines")
    print("=" * 60)
    print(f"{'Task':<25s} | {'Weighted Avg':>12s} | {'Stacking':>12s}")
    print("-" * 55)
    for task, r in ensemble_results.items():
        print(f"  {task:<23s} | {r['weighted_average_accuracy']:>11.1%} | {r['stacking_accuracy']:>11.1%}")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
