"""
MRI Image Classification Baselines for Epilepsy Diagnosis.

Trains discriminative classifiers on MRI subfigure images.
Models:
  1. ResNet-50 pretrained on ImageNet, fine-tuned
  2. BiomedCLIP embeddings + MLP head

Patient-level aggregation: mean-pool over per-subfigure predictions.

Usage:
    python baselines/train_mri_classifiers.py --tier gold --cv 5
    python baselines/train_mri_classifiers.py --tier gold --cv 5 --model resnet
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_label_maps, load_classification_csv, load_mri_images, DOWNSTREAM_TASKS


class PatientImageDataset(Dataset):
    """Dataset that loads all subfigure images for a patient."""

    def __init__(self, patient_ids, image_json_list, labels, transform=None, max_images=16):
        self.patient_ids = patient_ids
        self.image_json_list = image_json_list
        self.labels = labels
        self.transform = transform
        self.max_images = max_images

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        images = load_mri_images(self.image_json_list[idx])

        if not images:
            # Return a single black image as placeholder
            img_tensor = torch.zeros(1, 3, 224, 224)
            return img_tensor, self.labels[idx], 0

        # Limit number of images
        if len(images) > self.max_images:
            # Sample uniformly
            indices = np.linspace(0, len(images) - 1, self.max_images, dtype=int)
            images = [images[i] for i in indices]

        # Transform images
        img_tensors = []
        for img in images:
            if self.transform:
                img_tensors.append(self.transform(img))
            else:
                img_tensors.append(transforms.ToTensor()(img))

        img_tensor = torch.stack(img_tensors)  # (N, 3, 224, 224)
        return img_tensor, self.labels[idx], len(images)


def collate_patient_images(batch):
    """Custom collate that handles variable number of images per patient."""
    all_images = []
    all_labels = []
    all_counts = []

    for images, label, count in batch:
        all_images.append(images)
        all_labels.append(label)
        all_counts.append(count)

    return all_images, torch.tensor(all_labels), all_counts


class ResNetClassifier(nn.Module):
    """ResNet-50 with mean-pooling over patient subfigures."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward_single(self, x):
        """Process a single image tensor (B, 3, 224, 224) -> (B, n_classes)."""
        features = self.backbone(x)
        return self.classifier(features)

    def forward_patient(self, image_list):
        """Process variable-length image lists, mean-pool predictions per patient."""
        patient_logits = []
        for images in image_list:
            if images.shape[0] == 0:
                patient_logits.append(torch.zeros(1, self.classifier[-1].out_features,
                                                   device=images.device))
                continue
            device = next(self.parameters()).device
            images = images.to(device)
            logits = self.forward_single(images)  # (N, n_classes)
            # Mean pool over subfigures
            patient_logits.append(logits.mean(dim=0, keepdim=True))

        return torch.cat(patient_logits, dim=0)  # (B, n_classes)


def train_resnet_fold(model, train_dataset, val_dataset, n_classes, device, n_epochs=20, lr=1e-4):
    """Train one fold of ResNet classifier."""
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_patient_images, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            collate_fn=collate_patient_images, num_workers=4)

    # Compute class weights (access labels directly, don't load images)
    train_labels = list(train_dataset.labels)
    label_counts = Counter(train_labels)
    weights = torch.zeros(n_classes, device=device)
    for c in range(n_classes):
        weights[c] = len(train_labels) / (n_classes * max(label_counts.get(c, 1), 1))

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0
    best_preds = None
    best_probs = None
    patience = 5
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for images_list, labels, counts in train_loader:
            labels = labels.to(device)
            logits = model.forward_patient(images_list)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        all_val_preds = []
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for images_list, labels, counts in val_loader:
                labels = labels.to(device)
                logits = model.forward_patient(images_list)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_val_labels, all_val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = np.array(all_val_preds)
            best_probs = np.array(all_val_probs)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_acc, best_preds, best_probs


def train_image_classifier(df, label_maps, task, modality="mri", n_splits=5, device="cuda:0"):
    """Train image classifier for a single task with cross-validation."""
    # Get image column
    img_col = "mri_images" if modality == "mri" else "eeg_images"

    # Filter patients with images and valid labels
    id_col = f"{task}_label_id"
    has_images = df[img_col].notna()
    has_labels = df[id_col].notna() & (df[id_col] >= 0) if id_col in df.columns else pd.Series(False, index=df.index)
    valid_mask = has_images & has_labels

    df_valid = df[valid_mask].reset_index(drop=True)

    if len(df_valid) < 5:
        print(f"  [SKIP] {task}: only {len(df_valid)} patients with {modality} images + labels")
        return None

    inv_map = {v: k for k, v in label_maps[task].items()}
    label_names = [inv_map[i] for i in range(len(inv_map))]
    n_classes = len(label_names)

    labels = df_valid[id_col].values.astype(int)
    image_jsons = df_valid[img_col].values
    patient_ids = df_valid["patient_id"].values

    class_counts = Counter(labels)
    min_class = min(class_counts.values())
    actual_splits = min(n_splits, min_class)

    if actual_splits < 2:
        print(f"  [SKIP] {task}: smallest class has {min_class} samples")
        return None

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    fold_accs = []
    all_preds = np.full(len(labels), -1)
    all_probs = np.zeros((len(labels), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        print(f"    Fold {fold+1}/{actual_splits}...")

        train_dataset = PatientImageDataset(
            patient_ids[train_idx], image_jsons[train_idx], labels[train_idx],
            transform=train_transform
        )
        val_dataset = PatientImageDataset(
            patient_ids[val_idx], image_jsons[val_idx], labels[val_idx],
            transform=val_transform
        )

        model = ResNetClassifier(n_classes).to(device)
        fold_acc, preds, probs = train_resnet_fold(
            model, train_dataset, val_dataset, n_classes, device
        )

        all_preds[val_idx] = preds
        all_probs[val_idx] = probs
        fold_accs.append(fold_acc)

        del model
        torch.cuda.empty_cache()

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print(f"  {task}: {mean_acc:.3f} +/- {std_acc:.3f} ({len(labels)} patients)")

    valid_oof = all_preds >= 0
    if valid_oof.sum() > 0:
        report = classification_report(
            labels[valid_oof], all_preds[valid_oof],
            target_names=label_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(labels[valid_oof], all_preds[valid_oof])
    else:
        report = {}
        cm = None

    return {
        "task": task,
        "model": f"resnet50_{modality}",
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "fold_accuracies": [float(a) for a in fold_accs],
        "n_samples": int(len(labels)),
        "n_splits": actual_splits,
        "class_distribution": {label_names[k]: int(v) for k, v in class_counts.items()},
        "classification_report": report,
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "oof_predictions": all_preds.tolist(),
        "oof_probabilities": all_probs.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train MRI Image Classification Baselines")
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

    # Count patients with MRI images
    has_mri = df["mri_images"].notna()
    print(f"Loaded {len(df)} patients, {has_mri.sum()} with MRI images")

    print("\n" + "=" * 60)
    print("ResNet-50 MRI Image Classification Baselines")
    print("=" * 60)

    results = {}
    for task in DOWNSTREAM_TASKS:
        if task not in label_maps:
            continue
        result = train_image_classifier(
            df, label_maps, task, modality="mri",
            n_splits=args.cv, device=args.device
        )
        if result:
            results[task] = result

    # Save
    with open(os.path.join(save_dir, "mri_resnet_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: MRI Image Classification")
    print("=" * 60)
    for task, r in results.items():
        print(f"  {task:<25s}: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%} ({r['n_samples']} pts)")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
