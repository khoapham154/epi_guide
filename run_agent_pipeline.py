"""
Multi-Agent Pipeline: Text + MRI + EEG agents -> Orchestrator -> Diagnosis.

Sequential model loading to manage GPU memory:
  1. Text Agent (MedGemma-27B, NO RAG) -> generates clinical summaries
  2. MRI Agent (MedGemma-1.5-4B) -> generates radiology reports
  3. EEG Agent (MedGemma-1.5-4B) -> generates EEG reports
  4. Orchestrator (Llama-70B + RAG) -> integrates all reports -> final diagnosis

All agents communicate to the orchestrator through TEXT only.

Usage:
    python run_agent_pipeline.py --tier gold
    python run_agent_pipeline.py --tier gold --max_patients 5
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from configs.default import Config
from models.text_agent import TextAgent
from models.mri_agent import MRIAgent
from models.orchestrator import OrchestratorAgent
from models.report_parser import parse_to_label_indices
from data.dataset import (
    load_label_maps, load_classification_csv, get_ground_truth_labels,
    load_mri_images, load_eeg_images, DOWNSTREAM_TASKS,
)


def run_text_agent_phase(config: Config, df: pd.DataFrame) -> dict:
    """Phase 1: Generate text reports for all patients."""
    print("\n" + "=" * 60)
    print("PHASE 1: Text Agent (MedGemma-27B-text-it)")
    print("=" * 60)

    agent = TextAgent(
        model_name=config.text_agent.model_name,
        torch_dtype=config.text_agent.torch_dtype,
        device_map=config.text_agent.device_map,
        max_new_tokens=config.text_agent.max_new_tokens,
        temperature=config.text_agent.temperature,
        do_sample=config.text_agent.do_sample,
        system_prompt=config.text_agent.system_prompt,
        hf_token=config.hf_token,
        repetition_penalty=config.text_agent.repetition_penalty,
    )
    agent.load_model()

    text_reports = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Text Agent"):
        pid = row["patient_id"]
        semiology = str(row["semiology_text"]) if pd.notna(row.get("semiology_text")) else None
        mri_text = str(row["mri_report_text"]) if pd.notna(row.get("mri_report_text")) else None
        eeg_text = str(row["eeg_report_text"]) if pd.notna(row.get("eeg_report_text")) else None
        demographics = str(row["demographics_notes"]) if pd.notna(row.get("demographics_notes")) else None
        raw_facts = str(row["raw_facts"]) if pd.notna(row.get("raw_facts")) else None

        has_any = semiology or mri_text or eeg_text
        if has_any:
            try:
                text_reports[pid] = agent.generate_summary(
                    demographics_notes=demographics,
                    raw_facts=raw_facts,
                    semiology=semiology,
                    mri_report=mri_text,
                    eeg_report=eeg_text,
                )
            except Exception as e:
                print(f"\n  Error for {pid}: {e}")
                text_reports[pid] = ""
        else:
            text_reports[pid] = ""

    agent.unload_model()
    del agent
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Generated {sum(1 for v in text_reports.values() if v)} text reports")
    return text_reports


def run_mri_agent_phase(config: Config, df: pd.DataFrame, device: str = "cuda:0") -> dict:
    """Phase 2: Generate MRI reports for patients with images."""
    print("\n" + "=" * 60)
    print("PHASE 2: MRI Agent (MedGemma-1.5-4B-it)")
    print("=" * 60)

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
    agent.load_model(device=device)

    mri_reports = {}
    has_mri = df[df["mri_images"].notna()]
    for _, row in tqdm(has_mri.iterrows(), total=len(has_mri), desc="MRI Agent"):
        pid = row["patient_id"]
        images = load_mri_images(row.get("mri_images"))
        if images:
            try:
                mri_reports[pid] = agent.generate_report(images)
            except Exception as e:
                print(f"\n  Error for {pid}: {e}")
                mri_reports[pid] = ""
        else:
            mri_reports[pid] = ""

    agent.unload_model()
    del agent
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Generated {sum(1 for v in mri_reports.values() if v)} MRI reports")
    return mri_reports


def run_eeg_agent_phase(config: Config, df: pd.DataFrame, device: str = "cuda:0") -> dict:
    """Phase 3: Generate EEG reports for patients with EEG images."""
    print("\n" + "=" * 60)
    print("PHASE 3: EEG Agent (MedGemma-1.5-4B-it)")
    print("=" * 60)

    agent = MRIAgent(  # Same model class, different prompt
        model_name=config.eeg_image.model_name,
        torch_dtype=config.eeg_image.torch_dtype,
        max_new_tokens=config.eeg_image.max_new_tokens,
        temperature=config.eeg_image.temperature,
        do_sample=config.eeg_image.do_sample,
        system_prompt=config.eeg_image.system_prompt,
        hf_token=config.hf_token,
        repetition_penalty=config.eeg_image.repetition_penalty,
        no_repeat_ngram_size=config.eeg_image.no_repeat_ngram_size,
    )
    agent.load_model(device=device)

    eeg_reports = {}
    has_eeg = df[df["eeg_images"].notna()]
    for _, row in tqdm(has_eeg.iterrows(), total=len(has_eeg), desc="EEG Agent"):
        pid = row["patient_id"]
        images = load_eeg_images(row.get("eeg_images"))
        if images:
            try:
                eeg_reports[pid] = agent.generate_report(images)
            except Exception as e:
                print(f"\n  Error for {pid}: {e}")
                eeg_reports[pid] = ""
        else:
            eeg_reports[pid] = ""

    agent.unload_model()
    del agent
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Generated {sum(1 for v in eeg_reports.values() if v)} EEG reports")
    return eeg_reports


def run_orchestrator_phase(
    config: Config,
    df: pd.DataFrame,
    text_reports: dict,
    mri_reports: dict,
    eeg_reports: dict,
    label_maps: dict,
) -> dict:
    """Phase 4: Orchestrator synthesizes all reports -> diagnosis."""
    print("\n" + "=" * 60)
    print("PHASE 4: Orchestrator (Llama-3.3-70B-Instruct + RAG)")
    print("=" * 60)

    orchestrator = OrchestratorAgent(
        model_name=config.orchestrator.model_name,
        torch_dtype=config.orchestrator.torch_dtype,
        device_map=config.orchestrator.device_map,
        max_new_tokens=config.orchestrator.max_new_tokens,
        temperature=config.orchestrator.temperature,
        do_sample=config.orchestrator.do_sample,
        system_prompt=config.orchestrator.system_prompt,
        rag_top_k=config.orchestrator.rag_top_k,
        rag_embedding_model=config.orchestrator.rag_embedding_model,
        hf_token=config.hf_token,
        repetition_penalty=config.orchestrator.repetition_penalty,
    )
    orchestrator.load_model()

    results = {
        "patient_ids": [],
        "text_reports": [],
        "mri_reports": [],
        "eeg_reports": [],
        "orchestrator_diagnoses": [],
        "parsed_labels": [],
        "ground_truth_labels": [],
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Orchestrator"):
        pid = row["patient_id"]

        text_rpt = text_reports.get(pid, "")
        mri_rpt = mri_reports.get(pid, "")
        eeg_rpt = eeg_reports.get(pid, "")

        try:
            diagnosis = orchestrator.generate_diagnosis(
                text_report=text_rpt,
                mri_report=mri_rpt,
                eeg_report=eeg_rpt,
            )
        except Exception as e:
            print(f"\n  Error for {pid}: {e}")
            diagnosis = f"ERROR: {e}"

        parsed = parse_to_label_indices(diagnosis, label_maps)
        gt = get_ground_truth_labels(row, label_maps)

        results["patient_ids"].append(pid)
        results["text_reports"].append(text_rpt)
        results["mri_reports"].append(mri_rpt)
        results["eeg_reports"].append(eeg_rpt)
        results["orchestrator_diagnoses"].append(diagnosis)
        results["parsed_labels"].append(parsed)
        results["ground_truth_labels"].append(gt)

    orchestrator.unload_model()
    del orchestrator
    gc.collect()
    torch.cuda.empty_cache()

    return results


def compute_metrics(results: dict, label_maps: dict) -> dict:
    """Compute per-task accuracy."""
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

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Pipeline")
    parser.add_argument("--tier", type=str, default="gold", choices=["gold", "silver", "bronze"])
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for MRI/EEG agents (text+orchestrator use device_map=auto)")
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

    if args.max_patients:
        df = df.head(args.max_patients)

    print(f"Pipeline: {len(df)} patients, {args.tier} tier")
    print(f"Tasks: {', '.join(f'{t}: {len(m)} classes' for t, m in label_maps.items())}")

    # Phase 1-3: Run modality agents (sequential to save GPU memory)
    text_reports = run_text_agent_phase(config, df)
    mri_reports = run_mri_agent_phase(config, df, device=args.device)
    eeg_reports = run_eeg_agent_phase(config, df, device=args.device)

    # Save intermediate reports
    intermediate = {
        "text_reports": text_reports,
        "mri_reports": mri_reports,
        "eeg_reports": eeg_reports,
    }
    with open(os.path.join(save_dir, "agent_reports.json"), "w") as f:
        json.dump(intermediate, f, indent=2)
    print(f"\nIntermediate reports saved to {save_dir}/agent_reports.json")

    # Phase 4: Orchestrator
    results = run_orchestrator_phase(
        config, df, text_reports, mri_reports, eeg_reports, label_maps
    )

    # Compute metrics
    metrics = compute_metrics(results, label_maps)

    # Print results
    print("\n" + "=" * 60)
    print("MULTI-AGENT PIPELINE RESULTS")
    print("=" * 60)
    print(f"  Models: Text=MedGemma-27B | MRI/EEG=MedGemma-1.5-4B | Orch=Llama-70B+RAG")
    print("-" * 60)
    for task, m in metrics.items():
        print(f"  {task:25s}: {m['accuracy']:.1%} ({m['n_samples']} samples)")

    # Save results
    output = {"metrics": metrics, **results}
    output_path = os.path.join(save_dir, "pipeline_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save examples
    examples_path = os.path.join(save_dir, "pipeline_examples.txt")
    with open(examples_path, "w") as f:
        for i in range(min(10, len(results["patient_ids"]))):
            f.write(f"{'=' * 60}\nPatient: {results['patient_ids'][i]}\n{'=' * 60}\n\n")
            f.write(f"TEXT REPORT:\n{results['text_reports'][i][:500]}\n\n")
            f.write(f"MRI REPORT:\n{results['mri_reports'][i][:500]}\n\n")
            f.write(f"EEG REPORT:\n{results['eeg_reports'][i][:500]}\n\n")
            f.write(f"ORCHESTRATOR DIAGNOSIS:\n{results['orchestrator_diagnoses'][i]}\n\n")
            f.write(f"PARSED: {results['parsed_labels'][i]}\n")
            f.write(f"GT:     {results['ground_truth_labels'][i]}\n\n")

    print(f"\nResults saved to {output_path}")
    print(f"Examples saved to {examples_path}")


if __name__ == "__main__":
    main()
