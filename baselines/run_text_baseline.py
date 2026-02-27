"""
Text Agent Baseline Evaluation.

Evaluates MedGemma-27B-text-it on classification_gold.csv.
Zero-shot, NO RAG (RAG is only in orchestrator).

Usage:
    python baselines/run_text_baseline.py --tier gold
    python baselines/run_text_baseline.py --tier gold --max_patients 5
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
from models.text_agent import TextAgent
from models.report_parser import parse_to_label_indices
from data.dataset import load_label_maps, load_classification_csv, get_ground_truth_labels, DOWNSTREAM_TASKS


def main():
    parser = argparse.ArgumentParser(description="Text Agent Baseline")
    parser.add_argument("--tier", type=str, default="gold", choices=["gold", "silver", "bronze"])
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
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

    # Filter to patients with text
    has_text = (
        df["semiology_text"].notna() |
        df["mri_report_text"].notna() |
        df["eeg_report_text"].notna()
    )
    df = df[has_text].reset_index(drop=True)
    print(f"Loaded {len(df)} patients with text data")

    if args.max_patients:
        df = df.head(args.max_patients)

    print(f"Label maps: {', '.join(f'{t}: {len(m)} classes' for t, m in label_maps.items())}")

    # Init agent
    agent = TextAgent(
        model_name=config.text_agent.model_name,
        torch_dtype=config.text_agent.torch_dtype,
        device_map=config.text_agent.device_map,
        max_new_tokens=config.text_agent.max_new_tokens,
        temperature=config.text_agent.temperature,
        do_sample=config.text_agent.do_sample,
        repetition_penalty=config.text_agent.repetition_penalty,
        system_prompt=config.text_agent.system_prompt,
        hf_token=config.hf_token,
    )
    agent.load_model()

    # Run evaluation
    results = {
        "patient_ids": [],
        "generated_summaries": [],
        "parsed_labels": [],
        "ground_truth_labels": [],
    }

    print(f"\nGenerating summaries for {len(df)} patients...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row["patient_id"]

        semiology = str(row["semiology_text"]) if pd.notna(row.get("semiology_text")) else None
        mri_report = str(row["mri_report_text"]) if pd.notna(row.get("mri_report_text")) else None
        eeg_report = str(row["eeg_report_text"]) if pd.notna(row.get("eeg_report_text")) else None
        demographics = str(row["demographics_notes"]) if pd.notna(row.get("demographics_notes")) else None
        raw_facts = str(row["raw_facts"]) if pd.notna(row.get("raw_facts")) else None

        try:
            summary = agent.generate_summary(
                demographics_notes=demographics,
                raw_facts=raw_facts,
                semiology=semiology,
                mri_report=mri_report,
                eeg_report=eeg_report,
            )
        except Exception as e:
            print(f"\nError for {patient_id}: {e}")
            summary = f"ERROR: {e}"

        parsed = parse_to_label_indices(summary, label_maps)
        gt = get_ground_truth_labels(row, label_maps)

        results["patient_ids"].append(patient_id)
        results["generated_summaries"].append(summary)
        results["parsed_labels"].append(parsed)
        results["ground_truth_labels"].append(gt)

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
    print("TEXT AGENT BASELINE RESULTS (MedGemma-27B-text-it)")
    print("=" * 60)
    for task, m in metrics.items():
        print(f"  {task:25s}: {m['accuracy']:.1%} ({m['n_samples']} samples)")

    # Save
    output = {"metrics": metrics, **results}
    with open(os.path.join(save_dir, "text_baseline_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    with open(os.path.join(save_dir, "text_baseline_examples.txt"), "w") as f:
        for i in range(min(10, len(results["patient_ids"]))):
            f.write(f"{'=' * 60}\nPatient: {results['patient_ids'][i]}\n{'=' * 60}\n\n")
            f.write(f"GENERATED:\n{results['generated_summaries'][i]}\n\n")
            f.write(f"PARSED: {results['parsed_labels'][i]}\n")
            f.write(f"GT:     {results['ground_truth_labels'][i]}\n\n")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
