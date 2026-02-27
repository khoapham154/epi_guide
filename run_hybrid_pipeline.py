"""
Hybrid Pipeline: Discriminative + Generative Agentic System.

Combines trained discriminative classifiers with LLM-based agents for
epilepsy diagnosis. The pipeline:
1. Runs discriminative classifiers (fast, batch)
2. Runs modality agents (text/MRI/EEG)
3. Injects both into the hybrid orchestrator with RAG

Usage:
    python run_hybrid_pipeline.py --tier gold --max_patients 20
    python run_hybrid_pipeline.py --tier gold  # full evaluation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from configs.default import Config
from data.dataset import (
    load_label_maps, load_classification_csv, load_mri_images,
    load_eeg_images, get_ground_truth_labels, DOWNSTREAM_TASKS,
)
from models.report_parser import parse_to_label_indices
from models.hybrid_orchestrator import HybridOrchestrator, predictions_from_probabilities


def load_classifier_predictions(results_dir, label_maps, n_patients):
    """Load OOF predictions from trained classifiers for all patients."""
    classifier_files = {
        "text_classifier": ["tfidf_xgboost_results.json", "pubmedbert_results.json"],
        "mri_classifier": ["medsiglip_mri_results.json", "mri_resnet_results.json"],
        "eeg_classifier": ["medsiglip_eeg_results.json", "eeg_resnet_results.json"],
    }

    # For each classifier type, pick the best performing model's predictions
    all_predictions = {}

    for clf_type, filenames in classifier_files.items():
        best_predictions = {}

        for filename in filenames:
            path = os.path.join(results_dir, filename)
            if not os.path.exists(path):
                continue

            with open(path) as f:
                results = json.load(f)

            for task, task_result in results.items():
                probs = np.array(task_result.get("oof_probabilities", []))
                acc = task_result.get("mean_accuracy", 0)

                if task not in best_predictions or acc > best_predictions[task]["accuracy"]:
                    best_predictions[task] = {
                        "probabilities": probs,
                        "accuracy": acc,
                        "model": filename.replace("_results.json", ""),
                    }

        if best_predictions:
            all_predictions[clf_type] = best_predictions

    return all_predictions


def get_patient_predictions(all_predictions, patient_idx, label_maps):
    """Get discriminative predictions for a single patient."""
    patient_preds = {}

    for clf_type, task_preds in all_predictions.items():
        clf_patient_preds = {}
        for task, info in task_preds.items():
            probs = info["probabilities"]
            if patient_idx < len(probs):
                patient_probs = probs[patient_idx]
                # Convert to list of (label, prob) tuples
                inv_map = {v: k for k, v in label_maps[task].items()}
                task_pred_list = [
                    (inv_map.get(i, f"class_{i}"), float(p))
                    for i, p in enumerate(patient_probs)
                ]
                clf_patient_preds[task] = task_pred_list
            else:
                clf_patient_preds[task] = None

        if any(v is not None for v in clf_patient_preds.values()):
            patient_preds[clf_type] = clf_patient_preds
        else:
            patient_preds[clf_type] = None

    return patient_preds


def main():
    parser = argparse.ArgumentParser(description="Hybrid Pipeline")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--classifier_dir", type=str, default=None)
    parser.add_argument("--agent_reports", type=str, default=None,
                        help="Path to pre-generated agent_reports.json (skip agent generation)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--skip_agents", action="store_true",
                        help="Use cached agent reports instead of re-running agents")
    args = parser.parse_args()

    config = Config()
    classifier_dir = args.classifier_dir or f"logs/baselines/classifiers_{args.tier}"
    save_dir = args.save_dir or f"logs/baselines/hybrid_{args.tier}"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    csv_path = os.path.join(config.data.quality_dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(config.data.quality_dataset_dir)

    if args.max_patients:
        df = df.head(args.max_patients)

    n_patients = len(df)
    print(f"Loaded {n_patients} patients from {args.tier} tier")

    # Step 1: Load discriminative predictions
    print(f"\nLoading classifier predictions from {classifier_dir}...")
    all_predictions = load_classifier_predictions(classifier_dir, label_maps, n_patients)

    if all_predictions:
        print(f"  Loaded classifiers: {list(all_predictions.keys())}")
        for clf_type, task_preds in all_predictions.items():
            tasks_avail = [t for t, info in task_preds.items() if len(info["probabilities"]) > 0]
            print(f"    {clf_type}: {tasks_avail}")
    else:
        print("  WARNING: No classifier predictions found. Running without discriminative signals.")

    # Step 2: Load or generate agent reports
    agent_reports = {"text_reports": {}, "mri_reports": {}, "eeg_reports": {}}

    agent_reports_path = args.agent_reports or os.path.join(
        config.baseline.save_dir, f"full_{args.tier}", "agent_reports.json"
    )

    if args.skip_agents and os.path.exists(agent_reports_path):
        print(f"\nLoading cached agent reports from {agent_reports_path}...")
        with open(agent_reports_path) as f:
            cached = json.load(f)
        agent_reports = cached
        print(f"  Loaded {len(cached.get('text_reports', {}))} text, "
              f"{len(cached.get('mri_reports', {}))} MRI, "
              f"{len(cached.get('eeg_reports', {}))} EEG reports")
    else:
        print("\nGenerating agent reports (this may take a while)...")
        # Import and run agents
        from models.text_agent import TextAgent

        text_agent = TextAgent(
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
        text_agent.load_model()

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Text Agent"):
            pid = row["patient_id"]
            semiology = str(row["semiology_text"]) if pd.notna(row.get("semiology_text")) else None
            mri_report = str(row["mri_report_text"]) if pd.notna(row.get("mri_report_text")) else None
            eeg_report = str(row["eeg_report_text"]) if pd.notna(row.get("eeg_report_text")) else None

            try:
                report = text_agent.generate_summary(semiology, mri_report, eeg_report)
            except Exception as e:
                report = f"ERROR: {e}"

            agent_reports["text_reports"][pid] = report

        text_agent.unload_model()

        # Save agent reports for future use
        with open(os.path.join(save_dir, "agent_reports.json"), "w") as f:
            json.dump(agent_reports, f, indent=2)

    # Step 3: Initialize hybrid orchestrator
    print("\nInitializing hybrid orchestrator...")
    orchestrator = HybridOrchestrator(
        model_name=config.orchestrator.model_name,
        torch_dtype=config.orchestrator.torch_dtype,
        device_map=config.orchestrator.device_map,
        max_new_tokens=config.orchestrator.max_new_tokens,
        temperature=config.orchestrator.temperature,
        do_sample=config.orchestrator.do_sample,
        repetition_penalty=config.orchestrator.repetition_penalty,
        system_prompt=config.orchestrator.system_prompt,
        hf_token=config.hf_token,
        rag_top_k=config.orchestrator.rag_top_k,
        rag_embedding_model=config.orchestrator.rag_embedding_model,
        prediction_format=config.orchestrator.prediction_format,
    )
    orchestrator.load_model()

    # Step 4: Run hybrid pipeline
    results = {
        "patient_ids": [],
        "hybrid_diagnoses": [],
        "parsed_labels": [],
        "ground_truth_labels": [],
        "discriminative_predictions": [],
    }

    print(f"\nRunning hybrid pipeline for {len(df)} patients...")
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        pid = row["patient_id"]

        # Get discriminative predictions for this patient
        patient_preds = get_patient_predictions(all_predictions, idx, label_maps)

        # Get agent reports
        text_report = agent_reports.get("text_reports", {}).get(pid, "")
        mri_report = agent_reports.get("mri_reports", {}).get(pid, "")
        eeg_report = agent_reports.get("eeg_reports", {}).get(pid, "")

        # Run hybrid orchestrator
        try:
            diagnosis = orchestrator.generate_hybrid_diagnosis(
                text_report=text_report,
                mri_report=mri_report,
                eeg_report=eeg_report,
                discriminative_predictions=patient_preds,
            )
        except Exception as e:
            print(f"\nError for {pid}: {e}")
            diagnosis = f"ERROR: {e}"

        parsed = parse_to_label_indices(diagnosis, label_maps)
        gt = get_ground_truth_labels(row, label_maps)

        results["patient_ids"].append(pid)
        results["hybrid_diagnoses"].append(diagnosis)
        results["parsed_labels"].append(parsed)
        results["ground_truth_labels"].append(gt)
        results["discriminative_predictions"].append(
            {k: str(v) for k, v in patient_preds.items()} if patient_preds else {}
        )

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

    # Print results
    print("\n" + "=" * 60)
    print("HYBRID PIPELINE RESULTS")
    print("=" * 60)
    for task, m in metrics.items():
        print(f"  {task:25s}: {m['accuracy']:.1%} ({m['n_samples']} samples)")

    # Save
    output = {"metrics": metrics, **results}
    with open(os.path.join(save_dir, "hybrid_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Save examples
    with open(os.path.join(save_dir, "hybrid_examples.txt"), "w") as f:
        for i in range(min(10, len(results["patient_ids"]))):
            f.write(f"{'=' * 60}\nPatient: {results['patient_ids'][i]}\n{'=' * 60}\n\n")
            f.write(f"DISCRIMINATIVE PREDS:\n{results['discriminative_predictions'][i]}\n\n")
            f.write(f"HYBRID DIAGNOSIS:\n{results['hybrid_diagnoses'][i]}\n\n")
            f.write(f"PARSED: {results['parsed_labels'][i]}\n")
            f.write(f"GT:     {results['ground_truth_labels'][i]}\n\n")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
