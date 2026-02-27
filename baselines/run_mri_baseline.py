"""
MRI Agent Baseline Evaluation.

Evaluates MedGemma-1.5-4B-it on classification_gold.csv.
Zero-shot inference on brain MRI images.

Usage:
    python baselines/run_mri_baseline.py --tier gold --device cuda:0
    python baselines/run_mri_baseline.py --tier gold --max_patients 5 --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.default import Config
from models.mri_agent import MRIAgent
from models.report_parser import parse_to_label_indices
from data.dataset import (
    load_label_maps, load_classification_csv, get_ground_truth_labels,
    load_mri_images, DOWNSTREAM_TASKS,
)


def main():
    parser = argparse.ArgumentParser(description="MRI Agent Baseline")
    parser.add_argument("--tier", type=str, default="gold", choices=["gold", "silver", "bronze"])
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = Config()
    save_dir = args.save_dir or os.path.join(config.baseline.save_dir, f"full_{args.tier}")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    csv_path = os.path.join(
        config.data.quality_dataset_dir, f"classification_{args.tier}.csv"
    )
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(config.data.quality_dataset_dir)

    # Filter to patients with MRI images
    has_mri = df["mri_images"].notna()
    df = df[has_mri].reset_index(drop=True)
    print(f"Loaded {len(df)} patients with MRI images")

    if args.max_patients:
        df = df.head(args.max_patients)

    # Init agent
    agent = MRIAgent(
        model_name=config.mri_agent.model_name,
        torch_dtype=config.mri_agent.torch_dtype,
        max_new_tokens=config.mri_agent.max_new_tokens,
        temperature=config.mri_agent.temperature,
        do_sample=config.mri_agent.do_sample,
        system_prompt=config.mri_agent.system_prompt,
        hf_token=config.hf_token,
        repetition_penalty=config.mri_agent.repetition_penalty,
        no_repeat_ngram_size=config.mri_agent.no_repeat_ngram_size,
    )
    agent.load_model(device=args.device)

    # Run evaluation
    results = {
        "patient_ids": [],
        "generated_reports": [],
        "parsed_labels": [],
        "ground_truth_labels": [],
        "num_images": [],
    }

    print(f"\nGenerating MRI reports for {len(df)} patients...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row["patient_id"]
        images = load_mri_images(row.get("mri_images"))

        if not images:
            report = "No valid MRI images found."
        else:
            try:
                report = agent.generate_report(images)
            except Exception as e:
                print(f"\nError for {patient_id}: {e}")
                report = f"ERROR: {e}"

        parsed = parse_to_label_indices(report, label_maps)
        gt = get_ground_truth_labels(row, label_maps)

        results["patient_ids"].append(patient_id)
        results["generated_reports"].append(report)
        results["parsed_labels"].append(parsed)
        results["ground_truth_labels"].append(gt)
        results["num_images"].append(len(images))

    # Compute metrics
    metrics = {}
    for task in label_maps:
        y_true = [r[task] for r in results["ground_truth_labels"]]
        y_pred = [r[task] for r in results["parsed_labels"]]
        valid = [(t, p) for t, p in zip(y_true, y_pred) if t >= 0]

        if valid:
            correct = sum(1 for t, p in valid if t == p)
            metrics[task] = {"accuracy": correct / len(valid), "n_samples": len(valid)}
        else:
            metrics[task] = {"accuracy": 0.0, "n_samples": 0}

    # Print
    print("\n" + "=" * 60)
    print("MRI AGENT BASELINE RESULTS (MedGemma-1.5-4B-it)")
    print("=" * 60)
    for task, m in metrics.items():
        print(f"  {task:25s}: {m['accuracy']:.1%} ({m['n_samples']} samples)")

    avg_imgs = sum(results["num_images"]) / max(len(results["num_images"]), 1)
    print(f"\n  Avg images/patient: {avg_imgs:.1f}")

    # Save
    output = {"metrics": metrics, **results}
    with open(os.path.join(save_dir, "mri_baseline_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    with open(os.path.join(save_dir, "mri_baseline_examples.txt"), "w") as f:
        for i in range(min(10, len(results["patient_ids"]))):
            f.write(f"{'=' * 60}\nPatient: {results['patient_ids'][i]}\n")
            f.write(f"Images: {results['num_images'][i]}\n{'=' * 60}\n\n")
            f.write(f"GENERATED:\n{results['generated_reports'][i]}\n\n")
            f.write(f"PARSED: {results['parsed_labels'][i]}\n")
            f.write(f"GT:     {results['ground_truth_labels'][i]}\n\n")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
