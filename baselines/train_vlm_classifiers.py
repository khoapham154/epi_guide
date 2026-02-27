"""
MedSigLIP-448 Vision Encoder Classification Baselines for Epilepsy Diagnosis.

Uses google/medsiglip-448 as a frozen feature extractor with a trainable
classification head. Replaces ResNet-50 for the hybrid pipeline.

Patient-level aggregation: mean-pool over per-subfigure embeddings.

Usage:
    python baselines/train_vlm_classifiers.py --tier gold --cv 5
    python baselines/train_vlm_classifiers.py --tier gold --cv 5 --modality eeg
    python baselines/train_vlm_classifiers.py --tier gold --cv 5 --devices cuda:0,cuda:1
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_label_maps, load_classification_csv, load_mri_images, load_eeg_images, DOWNSTREAM_TASKS


class MedSigLIPDataset(Dataset):
    """Dataset that loads patient subfigure images for MedSigLIP processing."""

    def __init__(self, patient_ids, image_json_list, labels, processor, modality="mri", max_images=16):
        self.patient_ids = patient_ids
        self.image_json_list = image_json_list
        self.labels = labels
        self.processor = processor
        self.modality = modality
        self.max_images = max_images
        self._load_fn = load_mri_images if modality == "mri" else load_eeg_images

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        images = self._load_fn(self.image_json_list[idx])

        if not images:
            # Return a single black image placeholder
            dummy = Image.new("RGB", (448, 448), (0, 0, 0))
            inputs = self.processor(images=dummy, return_tensors="pt")
            pixel_values = inputs["pixel_values"]  # (1, 3, 448, 448)
            return pixel_values, self.labels[idx], 0

        if len(images) > self.max_images:
            indices = np.linspace(0, len(images) - 1, self.max_images, dtype=int)
            images = [images[i] for i in indices]

        # Process all images through MedSigLIP processor
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"]  # (N, 3, 448, 448)
        return pixel_values, self.labels[idx], len(images)


def collate_patient_images(batch):
    """Custom collate for variable number of images per patient."""
    all_images = []
    all_labels = []
    all_counts = []

    for images, label, count in batch:
        all_images.append(images)
        all_labels.append(label)
        all_counts.append(count)

    return all_images, torch.tensor(all_labels), all_counts


class MedSigLIPClassifier(nn.Module):
    """Frozen MedSigLIP-448 encoder + trainable classification head."""

    def __init__(self, vision_model, feat_dim, n_classes):
        super().__init__()
        self.vision_model = vision_model
        # Freeze encoder
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward_single(self, pixel_values):
        """Extract features and classify. (B, 3, 448, 448) -> (B, n_classes)."""
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
            # Use pooler_output if available, else mean over patch tokens
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output  # (B, feat_dim)
            else:
                features = outputs.last_hidden_state.mean(dim=1)  # (B, feat_dim)
        return self.classifier(features)

    def forward_patient(self, image_list):
        """Process variable-length image lists, mean-pool predictions per patient."""
        patient_logits = []
        for images in image_list:
            if images.shape[0] == 0:
                out_dim = self.classifier[-1].out_features
                patient_logits.append(torch.zeros(1, out_dim, device=next(self.classifier.parameters()).device))
                continue
            device = next(self.classifier.parameters()).device
            images = images.to(device)
            logits = self.forward_single(images)  # (N, n_classes)
            patient_logits.append(logits.mean(dim=0, keepdim=True))
        return torch.cat(patient_logits, dim=0)  # (B, n_classes)


def load_medsiglip_encoder(model_name="google/medsiglip-448", hf_token=None, device="cuda:0"):
    """Load MedSigLIP vision encoder and determine feature dimension."""
    from transformers import AutoModel, AutoProcessor

    print(f"Loading MedSigLIP encoder from {model_name}...")
    full_model = AutoModel.from_pretrained(model_name, token=hf_token, torch_dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

    # Extract vision model
    vision_model = full_model.vision_model

    # Determine feature dimension with a dummy forward pass
    vision_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 448, 448)
        outputs = vision_model(pixel_values=dummy)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat_dim = outputs.pooler_output.shape[-1]
        else:
            feat_dim = outputs.last_hidden_state.shape[-1]

    print(f"  Vision encoder feature dim: {feat_dim}")

    # Delete full model to free memory (we only need vision_model)
    del full_model
    torch.cuda.empty_cache()

    return vision_model, processor, feat_dim


def train_vlm_fold(model, train_dataset, val_dataset, n_classes, device, n_epochs=20, lr=1e-3):
    """Train one fold of VLM classifier (only classification head trains)."""
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_patient_images, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            collate_fn=collate_patient_images, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    # Compute class weights
    train_labels = list(train_dataset.labels)
    label_counts = Counter(train_labels)
    weights = torch.zeros(n_classes, device=device)
    for c in range(n_classes):
        weights[c] = len(train_labels) / (n_classes * max(label_counts.get(c, 1), 1))

    criterion = nn.CrossEntropyLoss(weight=weights)
    # Only optimize classifier head (encoder is frozen)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0
    best_preds = None
    best_probs = None
    patience = 5
    patience_counter = 0

    for epoch in range(n_epochs):
        model.classifier.train()
        for images_list, labels, counts in train_loader:
            labels = labels.to(device)
            logits = model.forward_patient(images_list)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.classifier.eval()
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


def train_vlm_classifier(df, label_maps, task, vision_model, processor, feat_dim,
                          modality="mri", n_splits=5, device="cuda:0"):
    """Train VLM classifier for a single task with cross-validation."""
    img_col = "mri_images" if modality == "mri" else "eeg_images"
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

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    fold_accs = []
    all_preds = np.full(len(labels), -1)
    all_probs = np.zeros((len(labels), n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        print(f"    Fold {fold+1}/{actual_splits}...")

        train_dataset = MedSigLIPDataset(
            patient_ids[train_idx], image_jsons[train_idx], labels[train_idx],
            processor=processor, modality=modality,
        )
        val_dataset = MedSigLIPDataset(
            patient_ids[val_idx], image_jsons[val_idx], labels[val_idx],
            processor=processor, modality=modality,
        )

        model = MedSigLIPClassifier(vision_model, feat_dim, n_classes).to(device)
        fold_acc, preds, probs = train_vlm_fold(
            model, train_dataset, val_dataset, n_classes, device
        )

        all_preds[val_idx] = preds
        all_probs[val_idx] = probs
        fold_accs.append(fold_acc)

        # Only delete classifier head (vision_model is shared)
        del model.classifier
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
        "model": f"medsiglip_{modality}",
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
    parser = argparse.ArgumentParser(description="Train MedSigLIP-448 Image Classification Baselines")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel task training")
    parser.add_argument("--modality", type=str, default="mri", choices=["mri", "eeg", "both"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = "external_data/MME"
    save_dir = args.save_dir or f"logs/baselines/classifiers_{args.tier}"
    os.makedirs(save_dir, exist_ok=True)

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from configs.default import Config
        hf_token = Config().hf_token

    csv_path = os.path.join(dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(dataset_dir)

    # Load MedSigLIP encoder once (shared across tasks and folds)
    vision_model, processor, feat_dim = load_medsiglip_encoder(hf_token=hf_token, device=args.device)
    vision_model = vision_model.to(args.device)

    modalities = ["mri", "eeg"] if args.modality == "both" else [args.modality]

    for modality in modalities:
        img_col = "mri_images" if modality == "mri" else "eeg_images"
        has_imgs = df[img_col].notna()
        print(f"\nLoaded {len(df)} patients, {has_imgs.sum()} with {modality.upper()} images")

        print(f"\n{'=' * 60}")
        print(f"MedSigLIP-448 {modality.upper()} Image Classification Baselines")
        print(f"{'=' * 60}")

        results = {}

        if args.devices:
            # Parallel training across GPUs
            devices = [d.strip() for d in args.devices.split(",")]
            tasks = [t for t in DOWNSTREAM_TASKS if t in label_maps]

            def train_task_on_device(task_device):
                task, dev = task_device
                vm = vision_model.to(dev) if dev != args.device else vision_model
                return task, train_vlm_classifier(
                    df, label_maps, task, vm, processor, feat_dim,
                    modality=modality, n_splits=args.cv, device=dev,
                )

            task_devices = [(t, devices[i % len(devices)]) for i, t in enumerate(tasks)]
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                for task, result in executor.map(train_task_on_device, task_devices):
                    if result:
                        results[task] = result
        else:
            # Sequential training
            for task in DOWNSTREAM_TASKS:
                if task not in label_maps:
                    continue
                result = train_vlm_classifier(
                    df, label_maps, task, vision_model, processor, feat_dim,
                    modality=modality, n_splits=args.cv, device=args.device,
                )
                if result:
                    results[task] = result

        # Save
        out_file = f"medsiglip_{modality}_results.json"
        with open(os.path.join(save_dir, out_file), "w") as f:
            json.dump(results, f, indent=2)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: MedSigLIP-448 {modality.upper()} Image Classification")
        print(f"{'=' * 60}")
        for task, r in results.items():
            print(f"  {task:<25s}: {r['mean_accuracy']:.1%} ± {r['std_accuracy']:.1%} ({r['n_samples']} pts)")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
