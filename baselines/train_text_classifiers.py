"""
Text Classification Baselines for Epilepsy Diagnosis.

Trains discriminative classifiers on clinical text (semiology + MRI report + EEG report).
Models:
  1. TF-IDF + XGBoost (lightweight baseline)
  2. PubMedBERT fine-tuned with classification heads

Usage:
    python baselines/train_text_classifiers.py --tier gold --cv 5
    python baselines/train_text_classifiers.py --tier gold --cv 5 --model pubmedbert
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_label_maps, load_classification_csv, DOWNSTREAM_TASKS


def prepare_text_data(df: pd.DataFrame, label_maps: dict):
    """Prepare text features and labels for each task."""
    # Concatenate all text fields
    texts = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row.get("semiology_text")):
            parts.append(f"SEMIOLOGY: {row['semiology_text']}")
        if pd.notna(row.get("mri_report_text")):
            parts.append(f"MRI: {row['mri_report_text']}")
        if pd.notna(row.get("eeg_report_text")):
            parts.append(f"EEG: {row['eeg_report_text']}")
        texts.append(" ".join(parts) if parts else "")

    # Get labels per task
    task_labels = {}
    for task in DOWNSTREAM_TASKS:
        if task not in label_maps:
            continue
        id_col = f"{task}_label_id"
        if id_col in df.columns:
            labels = df[id_col].values.astype(float)
        else:
            label_col = f"{task}_label"
            labels = np.full(len(df), -1.0)
            if label_col in df.columns:
                for i, val in enumerate(df[label_col]):
                    if pd.notna(val) and str(val).strip() in label_maps[task]:
                        labels[i] = label_maps[task][str(val).strip()]
        task_labels[task] = labels

    return texts, task_labels


def train_tfidf_xgboost(texts, labels, task_name, label_names, n_splits=5, save_dir=None):
    """Train TF-IDF + XGBoost classifier with cross-validation."""
    from xgboost import XGBClassifier

    # Filter valid samples
    valid_mask = labels >= 0
    X_texts = [texts[i] for i in range(len(texts)) if valid_mask[i]]
    y = labels[valid_mask].astype(int)

    if len(X_texts) < n_splits:
        print(f"  [SKIP] {task_name}: only {len(X_texts)} valid samples")
        return None

    # Check class distribution
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    if min_class_count < n_splits:
        print(f"  [WARN] {task_name}: smallest class has {min_class_count} samples, reducing splits")
        n_splits = min(n_splits, min_class_count)
        if n_splits < 2:
            print(f"  [SKIP] {task_name}: not enough samples for CV")
            return None

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs = []
    all_preds = np.full(len(y), -1)
    all_probs = np.zeros((len(y), len(label_names)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_texts, y)):
        X_train = [X_texts[i] for i in train_idx]
        X_val = [X_texts[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit TF-IDF on train
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_val_tfidf = tfidf.transform(X_val)

        # Train XGBoost
        n_classes = len(label_names)
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_train_tfidf, y_train, eval_set=[(X_val_tfidf, y_val)], verbose=False)

        # Predict
        y_pred = clf.predict(X_val_tfidf)
        y_prob = clf.predict_proba(X_val_tfidf)

        all_preds[val_idx] = y_pred
        all_probs[val_idx] = y_prob

        fold_acc = accuracy_score(y_val, y_pred)
        fold_accs.append(fold_acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"  {task_name}: {mean_acc:.3f} +/- {std_acc:.3f} ({len(y)} samples)")
    print(f"    Per-fold: {[f'{a:.3f}' for a in fold_accs]}")

    # Classification report on all OOF predictions
    valid_oof = all_preds >= 0
    if valid_oof.sum() > 0:
        report = classification_report(
            y[valid_oof], all_preds[valid_oof],
            target_names=label_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y[valid_oof], all_preds[valid_oof])
    else:
        report = {}
        cm = None

    result = {
        "task": task_name,
        "model": "tfidf_xgboost",
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "fold_accuracies": [float(a) for a in fold_accs],
        "n_samples": int(len(y)),
        "n_splits": n_splits,
        "class_distribution": {label_names[k]: int(v) for k, v in class_counts.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "oof_predictions": all_preds.tolist(),
        "oof_probabilities": all_probs.tolist(),
    }

    return result


def train_pubmedbert(texts, labels, task_name, label_names, n_splits=5, save_dir=None, device="cuda:0"):
    """Train PubMedBERT classifier with cross-validation."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModel

    # Filter valid samples
    valid_mask = labels >= 0
    X_texts = [texts[i] for i in range(len(texts)) if valid_mask[i]]
    y = labels[valid_mask].astype(int)

    if len(X_texts) < n_splits:
        print(f"  [SKIP] {task_name}: only {len(X_texts)} valid samples")
        return None

    class_counts = Counter(y)
    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        print(f"  [SKIP] {task_name}: smallest class has {min_class_count} samples")
        return None

    actual_splits = min(n_splits, min_class_count)

    # Load tokenizer
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize all texts
    print(f"  Tokenizing {len(X_texts)} texts for {task_name}...")
    encodings = tokenizer(
        X_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )

    n_classes = len(label_names)
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

    fold_accs = []
    all_preds = np.full(len(y), -1)
    all_probs = np.zeros((len(y), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_texts, y)):
        print(f"    Fold {fold+1}/{actual_splits}...")

        # Load fresh model each fold
        base_model = AutoModel.from_pretrained(model_name).to(device)

        # Simple classification head
        classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(base_model.config.hidden_size, n_classes),
        ).to(device)

        # Training data
        train_input_ids = encodings["input_ids"][train_idx].to(device)
        train_attention_mask = encodings["attention_mask"][train_idx].to(device)
        train_labels = torch.tensor(y[train_idx], dtype=torch.long).to(device)

        val_input_ids = encodings["input_ids"][val_idx].to(device)
        val_attention_mask = encodings["attention_mask"][val_idx].to(device)
        val_labels = torch.tensor(y[val_idx], dtype=torch.long).to(device)

        # Handle class imbalance with weighted loss
        class_weights = torch.zeros(n_classes, device=device)
        for c in range(n_classes):
            count = (train_labels == c).sum().item()
            class_weights[c] = len(train_labels) / (n_classes * max(count, 1))
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        optimizer = torch.optim.AdamW(
            list(base_model.parameters()) + list(classifier.parameters()),
            lr=2e-5, weight_decay=0.01,
        )

        # Training loop
        base_model.train()
        classifier.train()
        batch_size = 8
        n_epochs = 10
        best_val_acc = 0
        patience = 3
        patience_counter = 0

        for epoch in range(n_epochs):
            # Shuffle training data
            perm = torch.randperm(len(train_labels))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(train_labels), batch_size):
                batch_idx = perm[i:i+batch_size]
                outputs = base_model(
                    input_ids=train_input_ids[batch_idx],
                    attention_mask=train_attention_mask[batch_idx],
                )
                cls_output = outputs.last_hidden_state[:, 0]
                logits = classifier(cls_output)
                loss = criterion(logits, train_labels[batch_idx])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(base_model.parameters()) + list(classifier.parameters()), 1.0
                )
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validate
            base_model.eval()
            classifier.eval()
            with torch.no_grad():
                val_outputs = base_model(input_ids=val_input_ids, attention_mask=val_attention_mask)
                val_logits = classifier(val_outputs.last_hidden_state[:, 0])
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_preds = val_preds.cpu().numpy()
                best_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
                patience_counter = 0
            else:
                patience_counter += 1

            base_model.train()
            classifier.train()

            if patience_counter >= patience:
                break

        all_preds[val_idx] = best_preds
        all_probs[val_idx] = best_probs
        fold_accs.append(best_val_acc)

        # Cleanup
        del base_model, classifier, optimizer
        torch.cuda.empty_cache()

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"  {task_name}: {mean_acc:.3f} +/- {std_acc:.3f} ({len(y)} samples)")
    print(f"    Per-fold: {[f'{a:.3f}' for a in fold_accs]}")

    # Classification report
    valid_oof = all_preds >= 0
    if valid_oof.sum() > 0:
        report = classification_report(
            y[valid_oof], all_preds[valid_oof],
            target_names=label_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y[valid_oof], all_preds[valid_oof])
    else:
        report = {}
        cm = None

    result = {
        "task": task_name,
        "model": "pubmedbert",
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "fold_accuracies": [float(a) for a in fold_accs],
        "n_samples": int(len(y)),
        "n_splits": actual_splits,
        "class_distribution": {label_names[k]: int(v) for k, v in class_counts.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "oof_predictions": all_preds.tolist(),
        "oof_probabilities": all_probs.tolist(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Train Text Classification Baselines")
    parser.add_argument("--tier", type=str, default="gold", choices=["gold", "silver", "bronze"])
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--model", type=str, default="all", choices=["all", "tfidf", "pubmedbert"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Setup
    dataset_dir = "external_data/MME"
    save_dir = args.save_dir or f"logs/baselines/classifiers_{args.tier}"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    csv_path = os.path.join(dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(dataset_dir)

    print(f"Loaded {len(df)} patients from {args.tier} tier")
    print(f"Tasks: {list(label_maps.keys())}")

    # Prepare text data
    texts, task_labels = prepare_text_data(df, label_maps)

    # Print label distribution
    print("\nLabel distribution:")
    for task, labels in task_labels.items():
        valid = labels[labels >= 0]
        counts = Counter(valid.astype(int))
        inv_map = {v: k for k, v in label_maps[task].items()}
        dist = {inv_map.get(k, f"class_{k}"): v for k, v in sorted(counts.items())}
        print(f"  {task}: {len(valid)} valid — {dist}")

    all_results = {}

    # --- TF-IDF + XGBoost ---
    if args.model in ["all", "tfidf"]:
        print("\n" + "=" * 60)
        print("TF-IDF + XGBoost Baselines")
        print("=" * 60)

        tfidf_results = {}
        for task in DOWNSTREAM_TASKS:
            if task not in label_maps:
                continue
            inv_map = {v: k for k, v in label_maps[task].items()}
            label_names = [inv_map[i] for i in range(len(inv_map))]

            result = train_tfidf_xgboost(
                texts, task_labels[task], task, label_names,
                n_splits=args.cv, save_dir=save_dir
            )
            if result:
                tfidf_results[task] = result

        all_results["tfidf_xgboost"] = tfidf_results

        # Save
        with open(os.path.join(save_dir, "tfidf_xgboost_results.json"), "w") as f:
            json.dump(tfidf_results, f, indent=2)

    # --- PubMedBERT ---
    if args.model in ["all", "pubmedbert"]:
        print("\n" + "=" * 60)
        print("PubMedBERT Fine-tuned Baselines")
        print("=" * 60)

        bert_results = {}
        for task in DOWNSTREAM_TASKS:
            if task not in label_maps:
                continue
            inv_map = {v: k for k, v in label_maps[task].items()}
            label_names = [inv_map[i] for i in range(len(inv_map))]

            result = train_pubmedbert(
                texts, task_labels[task], task, label_names,
                n_splits=args.cv, save_dir=save_dir, device=args.device,
            )
            if result:
                bert_results[task] = result

        all_results["pubmedbert"] = bert_results

        with open(os.path.join(save_dir, "pubmedbert_results.json"), "w") as f:
            json.dump(bert_results, f, indent=2)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Text Classification Baselines")
    print("=" * 60)
    print(f"{'Task':<25s} | {'TF-IDF+XGB':>12s} | {'PubMedBERT':>12s}")
    print("-" * 55)

    for task in DOWNSTREAM_TASKS:
        tfidf_acc = "N/A"
        bert_acc = "N/A"
        if "tfidf_xgboost" in all_results and task in all_results["tfidf_xgboost"]:
            r = all_results["tfidf_xgboost"][task]
            tfidf_acc = f"{r['mean_accuracy']:.1%}±{r['std_accuracy']:.1%}"
        if "pubmedbert" in all_results and task in all_results["pubmedbert"]:
            r = all_results["pubmedbert"][task]
            bert_acc = f"{r['mean_accuracy']:.1%}±{r['std_accuracy']:.1%}"
        print(f"  {task:<23s} | {tfidf_acc:>12s} | {bert_acc:>12s}")

    # Save combined results
    with open(os.path.join(save_dir, "all_text_classifier_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
