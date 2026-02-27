#!/usr/bin/env python3
"""
Mobile-2 Source Localization Training Script.

Uses the LaBraM encoder (Path B of EEG Agent) for HD-EEG source localization.
Leave-one-subject-out cross-validation across 38 subjects.

Usage:
    python train_mobile2.py
    python train_mobile2.py --leave_out 5 --epochs 100
    python train_mobile2.py --from_eeg_agent checkpoints/stage3_meaf_best.pt
"""

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from configs import Config
from data.mobile2 import create_mobile2_loaders
from models.source_localization import Mobile2Model


def parse_args():
    parser = argparse.ArgumentParser(description="Mobile-2 source localization")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--leave_out", type=int, default=None,
                        help="Leave out specific subject (None=run all)")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from_eeg_agent", default=None,
                        help="Path to MEAF checkpoint to initialize LaBraM weights from")
    parser.add_argument("--log_dir", default="logs/mobile2")
    parser.add_argument("--checkpoint_dir", default="checkpoints/mobile2")
    return parser.parse_args()


def train_one_subject(
    model, train_loader, test_loader, device, args, config, leave_out_subject
):
    """Train and evaluate for one leave-out split."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=0.1, anneal_strategy="cos"
    )
    scaler = GradScaler(enabled=args.fp16 and not args.no_fp16)

    best_error = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            raw_eeg = batch["raw_eeg"].to(device)
            channel_ids = batch["channel_ids"].to(device)
            true_coords = batch["seeg_coords"].to(device)

            use_fp16 = args.fp16 and not args.no_fp16
            with autocast(enabled=use_fp16):
                pred_coords = model(raw_eeg, channel_ids)
                loss, _ = model.compute_loss(pred_coords, true_coords)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            all_errors = []
            with torch.no_grad():
                for batch in test_loader:
                    raw_eeg = batch["raw_eeg"].to(device)
                    channel_ids = batch["channel_ids"].to(device)
                    true_coords = batch["seeg_coords"].to(device)

                    pred_coords = model(raw_eeg, channel_ids)
                    distances = torch.norm(pred_coords - true_coords, dim=-1)
                    all_errors.extend(distances.cpu().numpy().tolist())

            if all_errors:
                mean_error = np.mean(all_errors)
                median_error = np.median(all_errors)
                within_20 = np.mean([e < 20 for e in all_errors])

                print(f"  Subject {leave_out_subject} | Epoch {epoch+1} | "
                      f"Loss: {epoch_loss:.4f} | "
                      f"Mean: {mean_error:.1f}mm | Median: {median_error:.1f}mm | "
                      f"<20mm: {within_20:.1%}")

                if mean_error < best_error:
                    best_error = mean_error

    return {
        "leave_out_subject": leave_out_subject,
        "best_mean_error_mm": best_error,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    print(f"Device: {device}")
    print(f"Mobile-2 data dir: {config.mobile2.data_dir}")

    # Determine subjects to evaluate
    if args.leave_out is not None:
        subjects = [args.leave_out]
    else:
        subjects = list(range(config.mobile2.n_subjects))

    all_results = []

    for leave_out in subjects:
        print(f"\n{'='*50}")
        print(f"Leave-one-out: Subject {leave_out}")
        print(f"{'='*50}")

        # Create data loaders
        train_loader, test_loader = create_mobile2_loaders(
            data_dir=config.mobile2.data_dir,
            n_subjects=config.mobile2.n_subjects,
            leave_out_subject=leave_out,
            batch_size=args.batch_size,
            original_sample_rate=config.mobile2.original_sample_rate,
            target_sample_rate=config.mobile2.target_sample_rate,
            num_channels=config.mobile2.num_channels,
            epoch_samples=config.mobile2.epoch_samples,
        )

        if len(train_loader.dataset) == 0:
            print(f"  No training data, skipping.")
            continue

        # Build model
        model = Mobile2Model(
            labram_hidden_dim=config.eeg_signal.labram_hidden_dim,
            labram_num_layers=config.eeg_signal.labram_num_layers,
            labram_num_heads=config.eeg_signal.labram_num_heads,
            labram_patch_size=config.eeg_signal.labram_patch_size,
            labram_max_channels=config.eeg_signal.labram_max_channels,
            projection_dim=config.eeg_signal.projector_hidden_dim,
            loc_hidden_dim=config.mobile2.loc_hidden_dim,
            loc_output_dim=config.mobile2.loc_output_dim,
        )

        # Load pretrained EEG agent weights
        if args.from_eeg_agent:
            ckpt = torch.load(args.from_eeg_agent, weights_only=False)
            eeg_state = {k.replace("eeg_agent.", ""): v
                         for k, v in ckpt["model_state_dict"].items()
                         if k.startswith("eeg_agent.")}
            model.load_eeg_agent_weights(eeg_state)
            print(f"  Loaded LaBraM weights from {args.from_eeg_agent}")

        model = model.to(device)

        result = train_one_subject(
            model, train_loader, test_loader, device, args, config, leave_out
        )
        all_results.append(result)

    # Summary
    if all_results:
        errors = [r["best_mean_error_mm"] for r in all_results]
        print(f"\n{'='*50}")
        print(f"Cross-validation results ({len(all_results)} subjects):")
        print(f"  Mean localization error: {np.mean(errors):.1f} +/- {np.std(errors):.1f} mm")
        print(f"{'='*50}")

        with open(os.path.join(args.log_dir, "mobile2_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
