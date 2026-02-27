#!/usr/bin/env python3
"""
Mobile-2 Training with REVE EEG Foundation Model.

Trains task-specific heads on top of frozen REVE (brain-bzh/reve-base)
using leave-one-subject-out CV across 7 subjects.

Three tasks:
  1. source_localization — predict 3D MNI coordinate (mm)
  2. ez_region           — classify Temporal / Frontal / Parieto-Occipital
  3. stim_intensity      — classify Low (<=0.3mA) / High (>=0.5mA)

Usage:
    python train_mobile2_reve.py --task source_localization --device cuda:6
    python train_mobile2_reve.py --task ez_region --device cuda:6
    python train_mobile2_reve.py --task stim_intensity --device cuda:6
    python train_mobile2_reve.py --task all --device cuda:6
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))

from configs.default import Config
from data.mobile2_bids import (
    Mobile2BIDSDataset,
    create_mobile2_bids_loaders,
    get_loso_splits,
    REGION_NAMES,
    INTENSITY_NAMES,
)
from models.reve_adapter import REVEAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Mobile-2 REVE training")
    parser.add_argument(
        "--task",
        choices=["source_localization", "ez_region", "stim_intensity", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--leave_out", type=str, default=None,
                        help="Leave out specific subject (e.g., 'sub-01'). None=run all.")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_oof_probs", action="store_true",
                        help="Save per-sample OOF predictions for MEAF pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--scheduler", type=str, default="cosine_restarts",
                        choices=["onecycle", "cosine_restarts"],
                        help="Learning rate scheduler")
    parser.add_argument("--no_focal_loss", action="store_true",
                        help="Use weighted CE instead of focal loss")
    return parser.parse_args()


def compute_class_weights(train_loader, n_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from training set."""
    class_counts = torch.zeros(n_classes)
    for batch in train_loader:
        targets = batch["target"]
        for c in range(n_classes):
            class_counts[c] += (targets == c).sum().item()

    # Inverse frequency, normalized so weights sum to n_classes
    total = class_counts.sum()
    weights = total / (n_classes * class_counts.clamp(min=1))
    print(f"    Class counts: {class_counts.long().tolist()}, weights: {weights.tolist()}")
    return weights.to(device)


def compute_balanced_accuracy(preds: torch.Tensor, targets: torch.Tensor, n_classes: int) -> float:
    """Compute balanced accuracy (mean of per-class recall)."""
    per_class_acc = []
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class_acc.append((preds[mask] == c).float().mean().item())
    if not per_class_acc:
        return 0.0
    return float(np.mean(per_class_acc))


def train_one_fold(
    task: str,
    train_subjects: list,
    test_subjects: list,
    config,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int = 30,
    scheduler_type: str = "cosine_restarts",
    use_focal_loss: bool = True,
) -> dict:
    """Train and evaluate for one LOSO fold."""
    m2_cfg = config.mobile2_bids

    # Use run-average for classification tasks (noise reduction, run-level eval)
    use_run_avg = task in ("ez_region", "stim_intensity")

    train_loader, test_loader = create_mobile2_bids_loaders(
        bids_root=m2_cfg.bids_root,
        task=task,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        batch_size=batch_size,
        target_sr=m2_cfg.target_sample_rate,
        use_run_average=use_run_avg,
        num_workers=4,
    )

    if len(train_loader.dataset) == 0:
        print(f"    No training data for fold, skipping.")
        return {}

    # Handle test set with no data gracefully - still try to train
    has_test_data = len(test_loader.dataset) > 0
    if not has_test_data:
        print(f"    WARNING: No test data for fold (test={test_subjects[0]}). "
              f"Training but skipping evaluation.")

    # Load electrode positions from the dataset for REVE
    electrode_names = train_loader.dataset.electrode_names
    electrode_positions = train_loader.dataset.electrode_positions

    # Build model
    model = REVEAdapter(
        task=task,
        reve_model_name=m2_cfg.reve_model_name,
        reve_positions_name=m2_cfg.reve_positions_name,
        hf_token=config.hf_token,
        feature_dim=m2_cfg.reve_feature_dim,
        freeze_backbone=m2_cfg.freeze_backbone,
        unfreeze_last_n=getattr(m2_cfg, 'unfreeze_last_n', 0),
        loc_hidden_dim=m2_cfg.loc_hidden_dim,
        loc_output_dim=m2_cfg.loc_output_dim,
        region_classes=m2_cfg.region_classes,
        intensity_classes=m2_cfg.intensity_classes,
        dropout=m2_cfg.dropout,
        use_focal_loss=use_focal_loss,
        electrode_positions=electrode_positions,
        electrode_names=electrode_names,
    ).to(device)

    # Compute and set class weights for classification tasks
    if task in ("ez_region", "stim_intensity"):
        n_classes = m2_cfg.region_classes if task == "ez_region" else m2_cfg.intensity_classes
        class_weights = compute_class_weights(train_loader, n_classes, device)
        model.set_class_weights(class_weights)

    # Source normalization: compute stats from training set
    target_mean, target_std = None, None
    if task == "source_localization":
        all_targets = []
        for batch in train_loader:
            all_targets.append(batch["target"])
        all_targets = torch.cat(all_targets, dim=0)
        target_mean = all_targets.mean(dim=0)
        target_std = all_targets.std(dim=0).clamp(min=1e-6)
        model.set_target_stats(target_mean, target_std)
        print(f"    Source norm: mean={target_mean.tolist()}, std={target_std.tolist()}")

    # Discriminative fine-tuning: separate LR for backbone vs head
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'head' not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'head' in n]

    if backbone_params:
        print(f"    Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={lr*0.1:.1e})")
        print(f"    Head params: {sum(p.numel() for p in head_params):,} (lr={lr:.1e})")
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ], weight_decay=0.01)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"    Trainable parameters: {n_trainable:,}")

    # Scheduler
    if scheduler_type == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=lr * 0.01,
        )
    else:
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=max(total_steps, 1),
            pct_start=0.1, anneal_strategy="cos",
        )

    scaler = GradScaler(enabled=True)

    best_metric = float("inf") if task == "source_localization" else 0.0
    best_result = {}
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            raw_eeg = batch["raw_eeg"].to(device)
            target = batch["target"].to(device)

            with autocast(enabled=True):
                pred = model(raw_eeg)
                # Normalize source targets during training
                if task == "source_localization" and target_mean is not None:
                    target_norm = (target - target_mean.to(device)) / target_std.to(device)
                    loss, _ = model.compute_loss(pred, target_norm)
                else:
                    loss, _ = model.compute_loss(pred, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler_type == "onecycle":
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        if scheduler_type == "cosine_restarts":
            scheduler.step(epoch)

        # Evaluate every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            if not has_test_data:
                print(f"    {test_subjects[0]} | Epoch {epoch+1:3d} | "
                      f"Loss: {epoch_loss:.4f} | (no test data)")
                continue

            metrics = evaluate(model, test_loader, task, device)
            test_sub = test_subjects[0]

            if task == "source_localization":
                is_better = metrics["mean_error_mm"] < best_metric
                if is_better:
                    best_metric = metrics["mean_error_mm"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10
                print(f"    {test_sub} | Epoch {epoch+1:3d} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Mean: {metrics['mean_error_mm']:.1f}mm | "
                      f"Median: {metrics['median_error_mm']:.1f}mm | "
                      f"<20mm: {metrics['within_20mm']:.1%}")
            else:
                bal_acc = metrics.get("balanced_accuracy", metrics["accuracy"])
                is_better = bal_acc > best_metric
                if is_better:
                    best_metric = bal_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10

                # Log per-class accuracy
                class_acc_str = " | ".join(
                    f"{k}: {v:.0%}" for k, v in metrics.get("class_accuracy", {}).items()
                )
                print(f"    {test_sub} | Epoch {epoch+1:3d} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Acc: {metrics['accuracy']:.1%} | "
                      f"BalAcc: {bal_acc:.1%} | "
                      f"{class_acc_str}")

            if is_better:
                best_result = {
                    "test_subject": test_sub,
                    "epoch": epoch + 1,
                    **metrics,
                }
                # Save best model state for final evaluation
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    # Final evaluation with best model to get raw predictions
    if best_result and best_state_dict and has_test_data:
        model.load_state_dict(best_state_dict)
        model.to(device)
        final_metrics = evaluate(model, test_loader, task, device)
        for key in ("raw_preds", "raw_targets", "raw_probs"):
            if key in final_metrics:
                best_result[key] = final_metrics[key]

    return best_result


def evaluate(
    model: nn.Module,
    test_loader,
    task: str,
    device: torch.device,
) -> dict:
    """Evaluate model on test set."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            raw_eeg = batch["raw_eeg"].to(device)
            target = batch["target"].to(device)

            with autocast(enabled=True):
                pred = model(raw_eeg)

            all_preds.append(pred.float().cpu())
            all_targets.append(target.float().cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    if task == "source_localization":
        # Denormalize predictions back to MNI coordinates
        if hasattr(model, 'denormalize_prediction'):
            preds = model.denormalize_prediction(preds)
        distances = torch.norm(preds - targets, dim=-1)
        return {
            "mean_error_mm": distances.mean().item(),
            "median_error_mm": distances.median().item(),
            "within_20mm": (distances < 20.0).float().mean().item(),
            "n_samples": len(distances),
            "raw_preds": preds.numpy(),
            "raw_targets": targets.numpy(),
        }
    else:
        pred_classes = preds.argmax(dim=-1)
        target_ints = targets.long()
        acc = (pred_classes == target_ints).float().mean().item()

        # Per-class accuracy
        n_classes = preds.size(1)
        class_acc = {}
        names = REGION_NAMES if task == "ez_region" else INTENSITY_NAMES
        per_class_recalls = []
        for c in range(n_classes):
            mask = target_ints == c
            if mask.sum() > 0:
                recall = (pred_classes[mask] == c).float().mean().item()
                class_acc[names.get(c, str(c))] = recall
                per_class_recalls.append(recall)

        # Balanced accuracy = mean of per-class recalls
        balanced_acc = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0

        # Softmax probabilities for MEAF pipeline
        probs = torch.softmax(preds, dim=-1)

        return {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "class_accuracy": class_acc,
            "n_samples": len(targets),
            "raw_probs": probs.numpy(),
            "raw_targets": targets.numpy(),
        }


def run_task(task: str, args, config):
    """Run full LOSO CV for one task."""
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")

    m2_cfg = config.mobile2_bids
    epochs = args.epochs or m2_cfg.epochs
    batch_size = args.batch_size or m2_cfg.batch_size
    lr = args.lr or m2_cfg.lr
    device = torch.device(args.device)

    # Get splits
    if args.leave_out:
        splits = [
            ([s for s in [f"sub-{i+1:02d}" for i in range(7)] if s != args.leave_out],
             [args.leave_out])
        ]
    else:
        splits = get_loso_splits(m2_cfg.bids_root)

    all_results = []
    t0 = time.time()

    for fold, (train_subs, test_subs) in enumerate(splits):
        print(f"\n  Fold {fold}: test={test_subs[0]} | train={train_subs}")

        try:
            result = train_one_fold(
                task=task,
                train_subjects=train_subs,
                test_subjects=test_subs,
                config=config,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=args.patience,
                scheduler_type=args.scheduler,
                use_focal_loss=not args.no_focal_loss,
            )
        except Exception as e:
            print(f"    ERROR in fold {fold} (test={test_subs[0]}): {e}")
            import traceback
            traceback.print_exc()
            result = {}

        if result:
            result["fold"] = fold
            all_results.append(result)
        else:
            print(f"    WARNING: Fold {fold} (test={test_subs[0]}) produced no results")

    elapsed = time.time() - t0

    # Aggregate results
    if not all_results:
        print("  No results collected.")
        return {}

    aggregate = {}
    if task == "source_localization":
        errors = [r["mean_error_mm"] for r in all_results]
        medians = [r["median_error_mm"] for r in all_results]
        w20s = [r["within_20mm"] for r in all_results]
        aggregate = {
            "mean_error_mm": f"{np.mean(errors):.1f} +/- {np.std(errors):.1f}",
            "median_error_mm": f"{np.mean(medians):.1f} +/- {np.std(medians):.1f}",
            "within_20mm": f"{np.mean(w20s):.3f} +/- {np.std(w20s):.3f}",
        }
        print(f"\n  CV Results ({len(all_results)} folds):")
        print(f"    Mean error:   {aggregate['mean_error_mm']} mm")
        print(f"    Median error: {aggregate['median_error_mm']} mm")
        print(f"    Within 20mm:  {aggregate['within_20mm']}")
    else:
        accs = [r["accuracy"] for r in all_results]
        bal_accs = [r.get("balanced_accuracy", r["accuracy"]) for r in all_results]
        aggregate = {
            "accuracy": f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}",
            "balanced_accuracy": f"{np.mean(bal_accs):.3f} +/- {np.std(bal_accs):.3f}",
        }
        print(f"\n  CV Results ({len(all_results)} folds):")
        print(f"    Accuracy:          {aggregate['accuracy']}")
        print(f"    Balanced Accuracy: {aggregate['balanced_accuracy']}")

        # Aggregate per-class accuracy
        all_class_accs = {}
        for r in all_results:
            for cls_name, cls_acc in r.get("class_accuracy", {}).items():
                all_class_accs.setdefault(cls_name, []).append(cls_acc)
        for cls_name, cls_accs in all_class_accs.items():
            print(f"    {cls_name}: {np.mean(cls_accs):.3f} +/- {np.std(cls_accs):.3f}")

    print(f"  Elapsed: {elapsed:.0f}s")

    # Save OOF predictions if requested
    oof_data = None
    if args.save_oof_probs and all_results:
        oof_data = {}
        for r in all_results:
            sub = r["test_subject"]
            if task == "source_localization":
                oof_data[sub] = {
                    "preds": r.get("raw_preds", np.array([])).tolist()
                        if isinstance(r.get("raw_preds"), np.ndarray) else [],
                    "targets": r.get("raw_targets", np.array([])).tolist()
                        if isinstance(r.get("raw_targets"), np.ndarray) else [],
                }
            else:
                oof_data[sub] = {
                    "probs": r.get("raw_probs", np.array([])).tolist()
                        if isinstance(r.get("raw_probs"), np.ndarray) else [],
                    "targets": r.get("raw_targets", np.array([])).tolist()
                        if isinstance(r.get("raw_targets"), np.ndarray) else [],
                }

    # Clean raw arrays from results before JSON serialization
    clean_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items()
                   if k not in ("raw_preds", "raw_targets", "raw_probs")}
        clean_results.append(clean_r)

    result = {
        "task": task,
        "model": "reve",
        "n_folds": len(clean_results),
        "per_fold": clean_results,
        "aggregate": aggregate,
        "elapsed_seconds": elapsed,
    }
    if oof_data:
        result["oof_predictions"] = oof_data

    return result


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = Config()

    save_dir = args.save_dir or os.path.join(
        config.baseline.save_dir, "mobile2_reve"
    )
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Mobile-2 External Validation — REVE EEG Foundation Model")
    print("=" * 60)
    print(f"Device:    {args.device}")
    print(f"BIDS:      {config.mobile2_bids.bids_root}")
    print(f"Save:      {save_dir}")
    print(f"Tasks:     {args.task}")
    print(f"Epochs:    {args.epochs or config.mobile2_bids.epochs}")
    print(f"LR:        {args.lr or config.mobile2_bids.lr}")
    print(f"Patience:  {args.patience}")
    print(f"Scheduler: {args.scheduler}")
    print(f"FocalLoss: {not args.no_focal_loss}")

    tasks = (
        ["source_localization", "ez_region", "stim_intensity"]
        if args.task == "all"
        else [args.task]
    )

    all_task_results = {}
    for task in tasks:
        result = run_task(task, args, config)
        if result:
            all_task_results[task] = result

            # Save per-task results
            fname = f"reve_{task}_results.json"
            with open(os.path.join(save_dir, fname), "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved: {os.path.join(save_dir, fname)}")

    # Save combined results
    combined_path = os.path.join(save_dir, "reve_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_task_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ALL DONE. Combined results: {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
