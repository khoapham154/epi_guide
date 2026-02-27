"""
Multi-Turn Hybrid Pipeline with Multi-GPU Support.

Combines:
  - Multi-GPU: All agents loaded simultaneously on separate GPUs
  - Multi-Turn: Orchestrator asks follow-up questions to agents
  - Hybrid: Discriminative predictions + generative agent reports
  - Binary injection: Per-class probability signals

GPU Layout (8x A100-80GB):
  GPUs 0-1: Text Agent (MedGemma-27B, ~54GB)
  GPU 2:    MRI Agent (MedGemma-1.5-4B, ~8GB)
  GPU 3:    EEG Agent (MedGemma-1.5-4B, ~8GB)
  GPUs 4-7: Orchestrator (GPT-OSS-120B, ~240GB)

Usage:
    python run_multiturn_pipeline.py --tier gold
    python run_multiturn_pipeline.py --tier gold --max_patients 20 --skip_agents
    python run_multiturn_pipeline.py --tier gold --no_multiturn  # one-pass mode
"""

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Reduce GPU memory fragmentation when loading multiple large models simultaneously.
# The caching allocator may hold reserved-but-unallocated blocks; expandable_segments
# allows those blocks to be returned to the OS and reallocated where needed.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from configs.default import Config
from data.dataset import (
    load_label_maps, load_classification_csv, get_ground_truth_labels,
    load_mri_images, load_eeg_images, DOWNSTREAM_TASKS,
)
from models.text_agent import TextAgent
from models.mri_agent import MRIAgent
from models.hybrid_orchestrator import HybridOrchestrator
from models.multi_turn_pipeline import MultiTurnPipeline
from models.report_parser import parse_to_label_indices


def parse_gpu_ids(gpu_str: str) -> dict:
    """Convert GPU string like '4,5,6,7' to max_memory dict.

    Sets each assigned GPU's budget to its current free memory minus a
    2 GiB safety margin. All other GPUs are set to '0GiB' so device_map=auto
    cannot spill onto them (reserved for other agents).
    """
    gpu_ids = [int(g.strip()) for g in gpu_str.split(",")]
    n_gpus = torch.cuda.device_count()
    max_memory = {}
    for i in range(n_gpus):
        if i in gpu_ids:
            free_bytes = torch.cuda.mem_get_info(i)[0]
            available_gib = max(1, int(free_bytes / (1024 ** 3)) - 2)
            max_memory[i] = f"{available_gib}GiB"
        else:
            max_memory[i] = "0GiB"
    return max_memory


def load_all_agents(config: Config):
    """Load all agents simultaneously on separate GPUs."""
    pipeline_cfg = config.pipeline

    print("=" * 60)
    print("Loading all agents on separate GPUs...")
    print("=" * 60)

    # Text Agent: GPUs 0-1
    text_gpus = parse_gpu_ids(pipeline_cfg.text_agent_gpus)
    text_agent = TextAgent(
        model_name=config.text_agent.model_name,
        torch_dtype=config.text_agent.torch_dtype,
        max_new_tokens=config.text_agent.max_new_tokens,
        temperature=config.text_agent.temperature,
        do_sample=config.text_agent.do_sample,
        repetition_penalty=config.text_agent.repetition_penalty,
        system_prompt=config.text_agent.system_prompt,
        hf_token=config.hf_token,
    )
    text_agent.load_model(device_map="auto", max_memory=text_gpus)

    # MRI Agent: GPU 2
    mri_device = f"cuda:{pipeline_cfg.mri_agent_gpu}"
    mri_agent = MRIAgent(
        model_name=config.mri_agent.model_name,
        torch_dtype=config.mri_agent.torch_dtype,
        max_new_tokens=config.mri_agent.max_new_tokens,
        temperature=config.mri_agent.temperature,
        do_sample=config.mri_agent.do_sample,
        repetition_penalty=config.mri_agent.repetition_penalty,
        no_repeat_ngram_size=config.mri_agent.no_repeat_ngram_size,
        system_prompt=config.mri_agent.system_prompt,
        hf_token=config.hf_token,
    )
    mri_agent.load_model(device=mri_device)

    # EEG Agent: GPU 3 (reuses MRIAgent class)
    eeg_device = f"cuda:{pipeline_cfg.eeg_agent_gpu}"
    eeg_agent = MRIAgent(
        model_name=config.eeg_image.model_name,
        torch_dtype=config.eeg_image.torch_dtype,
        max_new_tokens=config.eeg_image.max_new_tokens,
        temperature=config.eeg_image.temperature,
        do_sample=config.eeg_image.do_sample,
        repetition_penalty=config.eeg_image.repetition_penalty,
        no_repeat_ngram_size=config.eeg_image.no_repeat_ngram_size,
        system_prompt=config.eeg_image.system_prompt,
        hf_token=config.hf_token,
    )
    eeg_agent.load_model(device=eeg_device)

    # Orchestrator: GPUs 4-7
    orch_gpus = parse_gpu_ids(pipeline_cfg.orchestrator_gpus)
    orchestrator = HybridOrchestrator(
        model_name=config.orchestrator.model_name,
        torch_dtype=config.orchestrator.torch_dtype,
        max_new_tokens=config.orchestrator.max_new_tokens,
        temperature=config.orchestrator.temperature,
        do_sample=config.orchestrator.do_sample,
        repetition_penalty=config.orchestrator.repetition_penalty,
        system_prompt=config.orchestrator.system_prompt,
        rag_top_k=config.orchestrator.rag_top_k,
        rag_embedding_model=config.orchestrator.rag_embedding_model,
        hf_token=config.hf_token,
        prediction_format=config.orchestrator.prediction_format,
    )
    orchestrator.load_model(device_map="auto", max_memory=orch_gpus)

    print(f"\nAll agents loaded. GPU memory usage:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {alloc:.1f}GB / {total:.1f}GB")

    return text_agent, mri_agent, eeg_agent, orchestrator


def generate_reports_parallel(
    text_agent, mri_agent, eeg_agent, df: pd.DataFrame
) -> dict:
    """Generate all agent reports in parallel using ThreadPoolExecutor."""
    print("\nGenerating agent reports (parallel)...")

    def gen_text_reports():
        reports = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Text Agent"):
            pid = row["patient_id"]
            sem = str(row["semiology_text"]) if pd.notna(row.get("semiology_text")) else None
            mri_t = str(row["mri_report_text"]) if pd.notna(row.get("mri_report_text")) else None
            eeg_t = str(row["eeg_report_text"]) if pd.notna(row.get("eeg_report_text")) else None
            demographics = str(row["demographics_notes"]) if pd.notna(row.get("demographics_notes")) else None
            raw_facts_val = str(row["raw_facts"]) if pd.notna(row.get("raw_facts")) else None
            if sem or mri_t or eeg_t:
                try:
                    reports[pid] = text_agent.generate_summary(
                        demographics_notes=demographics,
                        raw_facts=raw_facts_val,
                        semiology=sem,
                        mri_report=mri_t,
                        eeg_report=eeg_t,
                    )
                except Exception as e:
                    reports[pid] = f"ERROR: {e}"
            else:
                reports[pid] = ""
        return reports

    def gen_mri_reports():
        reports = {}
        has_mri = df[df["mri_images"].notna()]
        for _, row in tqdm(has_mri.iterrows(), total=len(has_mri), desc="MRI Agent"):
            pid = row["patient_id"]
            images = load_mri_images(row.get("mri_images"))
            if images:
                try:
                    reports[pid] = mri_agent.generate_report(images)
                except Exception as e:
                    reports[pid] = f"ERROR: {e}"
        return reports

    def gen_eeg_reports():
        reports = {}
        has_eeg = df[df["eeg_images"].notna()]
        for _, row in tqdm(has_eeg.iterrows(), total=len(has_eeg), desc="EEG Agent"):
            pid = row["patient_id"]
            images = load_eeg_images(row.get("eeg_images"))
            if images:
                try:
                    reports[pid] = eeg_agent.generate_report(images)
                except Exception as e:
                    reports[pid] = f"ERROR: {e}"
        return reports

    with ThreadPoolExecutor(max_workers=3) as executor:
        text_future = executor.submit(gen_text_reports)
        mri_future = executor.submit(gen_mri_reports)
        eeg_future = executor.submit(gen_eeg_reports)

        text_reports = text_future.result()
        mri_reports = mri_future.result()
        eeg_reports = eeg_future.result()

    return {
        "text_reports": text_reports,
        "mri_reports": mri_reports,
        "eeg_reports": eeg_reports,
    }


def load_classifier_predictions(results_dir, label_maps):
    """Load discriminative classifier predictions."""
    classifier_files = {
        "text_classifier": ["pubmedbert_results.json", "tfidf_xgboost_results.json"],
        "mri_classifier": ["medsiglip_mri_results.json", "mri_resnet_results.json"],
        "eeg_classifier": ["medsiglip_eeg_results.json", "eeg_resnet_results.json"],
    }

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
    parser = argparse.ArgumentParser(description="Multi-Turn Hybrid Pipeline")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--classifier_dir", type=str, default=None)
    parser.add_argument("--agent_reports", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--skip_agents", action="store_true",
                        help="Use cached agent reports instead of re-running")
    parser.add_argument("--no_multiturn", action="store_true",
                        help="Disable multi-turn (one-pass mode)")
    args = parser.parse_args()

    config = Config()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or f"logs/baselines/multiturn_{args.tier}_{timestamp}"
    classifier_dir = args.classifier_dir or f"logs/baselines/classifiers_{args.tier}"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    csv_path = os.path.join(config.data.quality_dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(config.data.quality_dataset_dir)

    if args.max_patients:
        df = df.head(args.max_patients)

    n_patients = len(df)
    print(f"Pipeline: {n_patients} patients, {args.tier} tier")
    print(f"Multi-turn: {'disabled' if args.no_multiturn else f'enabled (max {config.multi_turn.max_rounds} rounds)'}")
    print(f"Prediction format: {config.orchestrator.prediction_format}")

    # Load discriminative predictions
    print(f"\nLoading classifier predictions from {classifier_dir}...")
    all_predictions = load_classifier_predictions(classifier_dir, label_maps)
    if all_predictions:
        for clf_type, task_preds in all_predictions.items():
            models_used = set(info["model"] for info in task_preds.values())
            print(f"  {clf_type}: {models_used}")
    else:
        print("  WARNING: No classifier predictions found.")

    # Load all agents on separate GPUs
    text_agent, mri_agent, eeg_agent, orchestrator = load_all_agents(config)

    # Generate or load agent reports
    if args.skip_agents and args.agent_reports and os.path.exists(args.agent_reports):
        print(f"\nLoading cached agent reports from {args.agent_reports}...")
        with open(args.agent_reports) as f:
            agent_reports = json.load(f)
    elif args.skip_agents:
        # Try default location
        default_path = os.path.join(config.baseline.save_dir, f"full_{args.tier}", "agent_reports.json")
        if os.path.exists(default_path):
            print(f"\nLoading cached agent reports from {default_path}...")
            with open(default_path) as f:
                agent_reports = json.load(f)
        else:
            print("\nNo cached reports found. Generating in parallel...")
            agent_reports = generate_reports_parallel(text_agent, mri_agent, eeg_agent, df)
    else:
        agent_reports = generate_reports_parallel(text_agent, mri_agent, eeg_agent, df)

    # Save agent reports
    with open(os.path.join(save_dir, "agent_reports.json"), "w") as f:
        json.dump(agent_reports, f, indent=2)

    # Set up multi-turn pipeline
    mt_pipeline = MultiTurnPipeline(
        text_agent=text_agent,
        mri_agent=mri_agent,
        eeg_agent=eeg_agent,
        orchestrator=orchestrator,
        max_rounds=0 if args.no_multiturn else config.multi_turn.max_rounds,
        max_questions_per_round=config.multi_turn.max_questions_per_round,
        followup_max_tokens=config.multi_turn.followup_max_tokens,
    )

    # Run pipeline
    results = {
        "patient_ids": [],
        "diagnoses": [],
        "parsed_labels": [],
        "ground_truth_labels": [],
        "conversation_logs": [],
        "num_rounds": [],
    }

    print(f"\nRunning {'multi-turn' if not args.no_multiturn else 'one-pass'} pipeline...")
    start_time = time.time()

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        pid = row["patient_id"]

        # Get discriminative predictions
        patient_preds = get_patient_predictions(all_predictions, idx, label_maps)

        # Get cached agent reports
        text_rpt = agent_reports.get("text_reports", {}).get(pid, "")
        mri_rpt = agent_reports.get("mri_reports", {}).get(pid, "")
        eeg_rpt = agent_reports.get("eeg_reports", {}).get(pid, "")

        # Build patient data for follow-up questions
        patient_data = {
            "semiology": str(row["semiology_text"]) if pd.notna(row.get("semiology_text")) else None,
            "mri_report_text": str(row["mri_report_text"]) if pd.notna(row.get("mri_report_text")) else None,
            "eeg_report_text": str(row["eeg_report_text"]) if pd.notna(row.get("eeg_report_text")) else None,
            "demographics_notes": str(row["demographics_notes"]) if pd.notna(row.get("demographics_notes")) else None,
            "raw_facts": str(row["raw_facts"]) if pd.notna(row.get("raw_facts")) else None,
            "mri_images": load_mri_images(row.get("mri_images")) if pd.notna(row.get("mri_images")) else [],
            "eeg_images": load_eeg_images(row.get("eeg_images")) if pd.notna(row.get("eeg_images")) else [],
        }

        # Run multi-turn pipeline
        try:
            result = mt_pipeline.run_patient(
                patient_data=patient_data,
                discriminative_predictions=patient_preds,
                text_report=text_rpt,
                mri_report=mri_rpt,
                eeg_report=eeg_rpt,
            )
            diagnosis = result["final_diagnosis"]
            conv_log = result["conversation_log"]
            n_rounds = result["num_rounds"]
        except Exception as e:
            print(f"\nError for {pid}: {e}")
            diagnosis = f"ERROR: {e}"
            conv_log = []
            n_rounds = 0

        parsed = parse_to_label_indices(diagnosis, label_maps)
        gt = get_ground_truth_labels(row, label_maps)

        results["patient_ids"].append(pid)
        results["diagnoses"].append(diagnosis)
        results["parsed_labels"].append(parsed)
        results["ground_truth_labels"].append(gt)
        results["conversation_logs"].append(conv_log)
        results["num_rounds"].append(n_rounds)

    elapsed = time.time() - start_time

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

    # Multi-turn stats
    round_counts = results["num_rounds"]
    mt_stats = {
        "avg_rounds": float(np.mean(round_counts)) if round_counts else 0,
        "max_rounds": int(max(round_counts)) if round_counts else 0,
        "min_rounds": int(min(round_counts)) if round_counts else 0,
        "total_time_seconds": elapsed,
        "time_per_patient": elapsed / n_patients if n_patients > 0 else 0,
    }

    # Print results
    print(f"\n{'=' * 60}")
    mode_name = "MULTI-TURN HYBRID" if not args.no_multiturn else "ONE-PASS HYBRID"
    print(f"{mode_name} PIPELINE RESULTS")
    print(f"{'=' * 60}")
    print(f"  Prediction format: {config.orchestrator.prediction_format}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Rounds: avg={mt_stats['avg_rounds']:.1f}, "
          f"min={mt_stats['min_rounds']}, max={mt_stats['max_rounds']}")
    print(f"{'-' * 60}")
    for task, m in metrics.items():
        print(f"  {task:25s}: {m['accuracy']:.1%} ({m['n_samples']} samples)")

    # Save
    output = {
        "config": {
            "multi_turn_enabled": not args.no_multiturn,
            "max_rounds": config.multi_turn.max_rounds,
            "prediction_format": config.orchestrator.prediction_format,
        },
        "metrics": metrics,
        "multi_turn_stats": mt_stats,
        **results,
    }
    with open(os.path.join(save_dir, "multiturn_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Save examples
    with open(os.path.join(save_dir, "multiturn_examples.txt"), "w") as f:
        for i in range(min(10, len(results["patient_ids"]))):
            f.write(f"{'=' * 60}\nPatient: {results['patient_ids'][i]}\n{'=' * 60}\n\n")
            f.write(f"ROUNDS: {results['num_rounds'][i]}\n")
            f.write(f"CONVERSATION LOG:\n{json.dumps(results['conversation_logs'][i], indent=2)}\n\n")
            f.write(f"FINAL DIAGNOSIS:\n{results['diagnoses'][i]}\n\n")
            f.write(f"PARSED: {results['parsed_labels'][i]}\n")
            f.write(f"GT:     {results['ground_truth_labels'][i]}\n\n")

    print(f"\nResults saved to {save_dir}")


if __name__ == "__main__":
    main()
