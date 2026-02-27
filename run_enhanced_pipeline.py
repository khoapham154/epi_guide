"""
Enhanced MEAF v2 Pipeline: Few-Shot + Meta-Ensemble + Improved Prompts.

Three layers of improvement over the original pipeline:
  Layer 1: Few-shot retrieval via PubMedBERT embeddings (zero-shot → few-shot)
  Layer 2: Enhanced prompts (confidence-tiered signals, JSON-only enforcement)
  Layer 3: Meta-ensemble post-processing (PubMedBERT + Agent + TF-IDF voting)

Supports both single-pass and multi-turn modes.
Uses cached agent reports + cached OOF predictions (no need to re-run baselines).

GPU Layout:
  Single-pass: Orchestrator on all available GPUs (agent reports cached)
  Multi-turn:  Agents on GPUs 0-3, Orchestrator on GPUs 4-7

Usage:
    # Single-pass (fast, ~2-3 hours)
    python run_enhanced_pipeline.py --mode single_pass --skip_agents \\
        --agent_reports logs/baselines/full_gold_20260218_030100/agent_reports.json \\
        --classifier_dir logs/baselines/classifiers_gold_20260218_030100

    # Multi-turn (slower, ~8-12 hours)
    python run_enhanced_pipeline.py --mode multi_turn \\
        --agent_reports logs/baselines/full_gold_20260218_030100/agent_reports.json \\
        --classifier_dir logs/baselines/classifiers_gold_20260218_030100

    # Both sequentially
    python run_enhanced_pipeline.py --mode both ...
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

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
from models.few_shot_retriever import FewShotRetriever
from models.meta_ensemble import MetaEnsemble


def parse_gpu_ids(gpu_str: str) -> dict:
    """Convert GPU string like '4,5,6,7' to max_memory dict."""
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


def prepare_texts(df: pd.DataFrame) -> list:
    """Prepare concatenated text features matching train_text_classifiers.py format."""
    texts = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row.get("demographics_notes")):
            parts.append(f"DEMOGRAPHICS: {row['demographics_notes']}")
        if pd.notna(row.get("raw_facts")):
            parts.append(f"FACTS: {row['raw_facts']}")
        if pd.notna(row.get("semiology_text")):
            parts.append(f"SEMIOLOGY: {row['semiology_text']}")
        if pd.notna(row.get("mri_report_text")):
            parts.append(f"MRI: {row['mri_report_text']}")
        if pd.notna(row.get("eeg_report_text")):
            parts.append(f"EEG: {row['eeg_report_text']}")
        texts.append(" ".join(parts) if parts else "")
    return texts


def load_classifier_predictions(results_dir, label_maps):
    """Load discriminative classifier OOF predictions (from run_multiturn_pipeline.py)."""
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


def build_oof_index_maps(df, label_maps):
    """
    Build mapping from CSV row index → OOF array index for each task.

    OOF arrays only contain entries for patients with valid labels (>= 0).
    So CSV row 5 might map to OOF index 4 if row 3 had label -1.
    """
    task_list = [t for t in label_maps if not t.endswith("_id2label")]
    oof_maps = {}
    for task in task_list:
        label_col = f"{task}_label_id"
        if label_col not in df.columns:
            label_col = f"{task}_label"

        csv_to_oof = {}
        oof_idx = 0
        for csv_idx in range(len(df)):
            val = df.iloc[csv_idx].get(label_col)
            if pd.notna(val) and float(val) >= 0:
                csv_to_oof[csv_idx] = oof_idx
                oof_idx += 1
        oof_maps[task] = csv_to_oof
    return oof_maps


def get_patient_predictions(all_predictions, patient_idx, label_maps, oof_maps=None):
    """Get discriminative predictions for a single patient.

    Uses oof_maps to correctly translate CSV row index → OOF array index.
    """
    patient_preds = {}
    for clf_type, task_preds in all_predictions.items():
        clf_patient_preds = {}
        for task, info in task_preds.items():
            probs = info["probabilities"]
            # Use oof_maps for correct index mapping
            if oof_maps and task in oof_maps:
                oof_idx = oof_maps[task].get(patient_idx)
            else:
                oof_idx = patient_idx

            if oof_idx is not None and oof_idx < len(probs):
                patient_probs = probs[oof_idx]
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


def load_agents(config: Config, agent_gpus: str):
    """Load modality agents for multi-turn mode."""
    pipeline_cfg = config.pipeline

    print("Loading agents for multi-turn follow-ups...")

    # Text Agent
    text_max_mem = parse_gpu_ids(agent_gpus.split(",")[0] + "," + agent_gpus.split(",")[1]
                                  if len(agent_gpus.split(",")) > 1
                                  else agent_gpus.split(",")[0])
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
    # Use first 2 GPUs from agent_gpus for text agent
    agent_gpu_list = [int(g.strip()) for g in agent_gpus.split(",")]
    text_gpu_ids = agent_gpu_list[:2] if len(agent_gpu_list) >= 2 else agent_gpu_list
    text_max_mem = parse_gpu_ids(",".join(str(g) for g in text_gpu_ids))
    text_agent.load_model(device_map="auto", max_memory=text_max_mem)

    # MRI Agent
    mri_gpu = agent_gpu_list[2] if len(agent_gpu_list) > 2 else agent_gpu_list[-1]
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
    mri_agent.load_model(device=f"cuda:{mri_gpu}")

    # EEG Agent
    eeg_gpu = agent_gpu_list[3] if len(agent_gpu_list) > 3 else agent_gpu_list[-1]
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
    eeg_agent.load_model(device=f"cuda:{eeg_gpu}")

    print("All agents loaded.")
    return text_agent, mri_agent, eeg_agent


def run_pipeline(
    mode: str,
    config: Config,
    df: pd.DataFrame,
    label_maps: dict,
    agent_reports: dict,
    all_predictions: dict,
    retriever: FewShotRetriever,
    ensemble: MetaEnsemble,
    orchestrator_gpus: str,
    agent_gpus: str = "0,1,2,3",
    save_dir: str = ".",
    oof_maps: dict = None,
):
    """Run enhanced pipeline in single_pass or multi_turn mode."""
    is_multiturn = (mode == "multi_turn")
    mode_name = "MULTI-TURN" if is_multiturn else "SINGLE-PASS"

    print(f"\n{'=' * 60}")
    print(f"ENHANCED {mode_name} PIPELINE")
    print(f"{'=' * 60}")
    print(f"  Patients: {len(df)}")
    print(f"  Few-shot: top-{config.enhanced.few_shot_top_k}")
    if ensemble.task_weights:
        print(f"  Ensemble: CALIBRATED per-task weights")
        for t, (wb, wt) in ensemble.task_weights.items():
            print(f"    {t}: bert={wb:.2f}, tfidf={wt:.2f}")
    else:
        print(f"  Ensemble: bert={config.enhanced.ensemble_weight_pubmedbert}, "
              f"tfidf={config.enhanced.ensemble_weight_tfidf}")

    # Load orchestrator
    orch_gpus = parse_gpu_ids(orchestrator_gpus)
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

    # Load agents for multi-turn
    mt_pipeline = None
    if is_multiturn:
        text_agent, mri_agent, eeg_agent = load_agents(config, agent_gpus)
        mt_pipeline = MultiTurnPipeline(
            text_agent=text_agent,
            mri_agent=mri_agent,
            eeg_agent=eeg_agent,
            orchestrator=orchestrator,
            max_rounds=config.multi_turn.max_rounds,
            max_questions_per_round=config.multi_turn.max_questions_per_round,
            followup_max_tokens=config.multi_turn.followup_max_tokens,
        )

    # Results storage
    results = {
        "patient_ids": [],
        "diagnoses": [],
        "parsed_labels": [],           # Raw LLM-parsed labels
        "ensemble_labels": [],          # Meta-ensemble final labels
        "ground_truth_labels": [],
        "num_rounds": [],
    }

    start_time = time.time()

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=mode_name)):
        pid = row["patient_id"]

        # 1. Get few-shot examples (leave-one-out)
        if config.enhanced.use_few_shot and retriever.index is not None:
            examples = retriever.retrieve_similar(idx, top_k=config.enhanced.few_shot_top_k)
            few_shot_text = retriever.format_few_shot_examples(examples)
        else:
            few_shot_text = ""

        # 2. Get cached reports
        text_rpt = agent_reports.get("text_reports", {}).get(pid, "")
        mri_rpt = agent_reports.get("mri_reports", {}).get(pid, "")
        eeg_rpt = agent_reports.get("eeg_reports", {}).get(pid, "")

        # 3. Get discriminative predictions
        patient_preds = get_patient_predictions(all_predictions, idx, label_maps, oof_maps=oof_maps)

        # 4. Run orchestrator
        n_rounds = 1
        try:
            if is_multiturn and mt_pipeline is not None:
                patient_data = {
                    "semiology": str(row["semiology_text"]) if pd.notna(row.get("semiology_text")) else None,
                    "mri_report_text": str(row["mri_report_text"]) if pd.notna(row.get("mri_report_text")) else None,
                    "eeg_report_text": str(row["eeg_report_text"]) if pd.notna(row.get("eeg_report_text")) else None,
                    "demographics_notes": str(row["demographics_notes"]) if pd.notna(row.get("demographics_notes")) else None,
                    "raw_facts": str(row["raw_facts"]) if pd.notna(row.get("raw_facts")) else None,
                    "mri_images": load_mri_images(row.get("mri_images")) if pd.notna(row.get("mri_images")) else [],
                    "eeg_images": load_eeg_images(row.get("eeg_images")) if pd.notna(row.get("eeg_images")) else [],
                }
                result = mt_pipeline.run_patient(
                    patient_data=patient_data,
                    discriminative_predictions=patient_preds,
                    text_report=text_rpt,
                    mri_report=mri_rpt,
                    eeg_report=eeg_rpt,
                    few_shot_examples=few_shot_text,
                )
                diagnosis = result["final_diagnosis"]
                n_rounds = result["num_rounds"]
            else:
                diagnosis = orchestrator.generate_hybrid_diagnosis(
                    text_report=text_rpt,
                    mri_report=mri_rpt,
                    eeg_report=eeg_rpt,
                    discriminative_predictions=patient_preds,
                    few_shot_examples=few_shot_text,
                )
        except Exception as e:
            print(f"\nError for {pid}: {e}")
            diagnosis = f"ERROR: {e}"

        # 5. Parse LLM output
        parsed = parse_to_label_indices(diagnosis, label_maps)
        gt = get_ground_truth_labels(row, label_maps)

        # 6. Meta-ensemble post-processing
        if config.enhanced.use_ensemble:
            ensemble_labels = ensemble.predict_all_tasks(
                patient_idx=idx,
                agent_labels=parsed,
                agent_text=diagnosis,
            )
        else:
            ensemble_labels = parsed

        results["patient_ids"].append(pid)
        results["diagnoses"].append(diagnosis)
        results["parsed_labels"].append(parsed)
        results["ensemble_labels"].append(ensemble_labels)
        results["ground_truth_labels"].append(gt)
        results["num_rounds"].append(n_rounds)

    elapsed = time.time() - start_time

    # Compute metrics for BOTH raw parsed and ensemble
    metrics_parsed = {}
    metrics_ensemble = {}

    for task in label_maps:
        y_true = [r[task] for r in results["ground_truth_labels"]]

        # Raw parsed (LLM only)
        y_pred_parsed = [r[task] for r in results["parsed_labels"]]
        valid_p = [(t, p) for t, p in zip(y_true, y_pred_parsed) if t >= 0]
        if valid_p:
            correct = sum(1 for t, p in valid_p if t == p)
            metrics_parsed[task] = {"accuracy": correct / len(valid_p), "n_samples": len(valid_p)}
        else:
            metrics_parsed[task] = {"accuracy": 0.0, "n_samples": 0}

        # Ensemble (final)
        y_pred_ens = [r[task] for r in results["ensemble_labels"]]
        valid_e = [(t, p) for t, p in zip(y_true, y_pred_ens) if t >= 0]
        if valid_e:
            correct = sum(1 for t, p in valid_e if t == p)
            metrics_ensemble[task] = {"accuracy": correct / len(valid_e), "n_samples": len(valid_e)}
        else:
            metrics_ensemble[task] = {"accuracy": 0.0, "n_samples": 0}

    # Print results
    print(f"\n{'=' * 60}")
    print(f"ENHANCED {mode_name} RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Time per patient: {elapsed/len(df):.1f}s")
    print(f"\n{'Task':<25s} | {'LLM Only':>10s} | {'+ Ensemble':>10s}")
    print("-" * 50)
    for task in label_maps:
        p_acc = metrics_parsed[task]["accuracy"]
        e_acc = metrics_ensemble[task]["accuracy"]
        n = metrics_ensemble[task]["n_samples"]
        delta = e_acc - p_acc
        print(f"  {task:<23s} | {p_acc:>9.1%} | {e_acc:>9.1%} ({'+' if delta >= 0 else ''}{delta:.1%})")

    avg_parsed = np.mean([m["accuracy"] for m in metrics_parsed.values()])
    avg_ensemble = np.mean([m["accuracy"] for m in metrics_ensemble.values()])
    print(f"  {'AVERAGE':<23s} | {avg_parsed:>9.1%} | {avg_ensemble:>9.1%}")

    # Save results
    output = {
        "config": {
            "mode": mode,
            "few_shot_enabled": config.enhanced.use_few_shot,
            "few_shot_top_k": config.enhanced.few_shot_top_k,
            "ensemble_enabled": config.enhanced.use_ensemble,
            "ensemble_weights": {
                "pubmedbert": config.enhanced.ensemble_weight_pubmedbert,
                "agent": config.enhanced.ensemble_weight_agent,
                "tfidf": config.enhanced.ensemble_weight_tfidf,
            },
            "high_confidence_threshold": config.enhanced.high_confidence_threshold,
            "prediction_format": config.orchestrator.prediction_format,
        },
        "metrics_llm_only": metrics_parsed,
        "metrics_ensemble": metrics_ensemble,
        "elapsed_seconds": elapsed,
        **results,
    }

    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"enhanced_{mode}_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Save examples
    examples_path = os.path.join(save_dir, f"enhanced_{mode}_examples.txt")
    with open(examples_path, "w") as f:
        for i in range(min(10, len(results["patient_ids"]))):
            f.write(f"{'=' * 60}\nPatient: {results['patient_ids'][i]}\n{'=' * 60}\n\n")
            f.write(f"DIAGNOSIS:\n{results['diagnoses'][i][:1000]}\n\n")
            f.write(f"LLM PARSED:  {results['parsed_labels'][i]}\n")
            f.write(f"ENSEMBLE:    {results['ensemble_labels'][i]}\n")
            f.write(f"GT:          {results['ground_truth_labels'][i]}\n\n")

    print(f"\nResults saved to {output_path}")

    # Cleanup
    orchestrator.unload_model()
    del orchestrator
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_ensemble


def main():
    parser = argparse.ArgumentParser(description="Enhanced MEAF v2 Pipeline")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--mode", type=str, default="single_pass",
                        choices=["single_pass", "multi_turn", "both"])
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--classifier_dir", type=str, default=None)
    parser.add_argument("--agent_reports", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--skip_agents", action="store_true",
                        help="Use cached agent reports (required for single_pass)")
    parser.add_argument("--orchestrator_gpus", type=str, default="4,5,6,7")
    parser.add_argument("--agent_gpus", type=str, default="0,1,2,3")
    parser.add_argument("--few_shot_device", type=str, default="cuda:0",
                        help="GPU for building few-shot FAISS index")
    # Ablation flags
    parser.add_argument("--no_few_shot", action="store_true")
    parser.add_argument("--no_ensemble", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.no_few_shot:
        config.enhanced.use_few_shot = False
    if args.no_ensemble:
        config.enhanced.use_ensemble = False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir or f"logs/baselines/enhanced_{args.tier}_{timestamp}"
    classifier_dir = args.classifier_dir or f"logs/baselines/classifiers_{args.tier}_20260218_030100"

    # Load data
    csv_path = os.path.join(config.data.quality_dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    label_maps = load_label_maps(config.data.quality_dataset_dir)

    if args.max_patients:
        df = df.head(args.max_patients)

    print(f"Enhanced Pipeline: {len(df)} patients, {args.tier} tier")
    print(f"Mode: {args.mode}")
    print(f"Few-shot: {'enabled' if config.enhanced.use_few_shot else 'DISABLED'}")
    print(f"Ensemble: {'enabled' if config.enhanced.use_ensemble else 'DISABLED'}")

    # Load cached agent reports
    agent_reports_path = args.agent_reports
    if not agent_reports_path:
        agent_reports_path = f"logs/baselines/full_{args.tier}_20260218_030100/agent_reports.json"
    print(f"\nLoading cached agent reports from {agent_reports_path}...")
    with open(agent_reports_path) as f:
        agent_reports = json.load(f)
    n_text = sum(1 for v in agent_reports.get("text_reports", {}).values() if v)
    n_mri = sum(1 for v in agent_reports.get("mri_reports", {}).values() if v)
    n_eeg = sum(1 for v in agent_reports.get("eeg_reports", {}).values() if v)
    print(f"  Loaded: {n_text} text, {n_mri} MRI, {n_eeg} EEG reports")

    # Load discriminative predictions (for orchestrator prompt injection)
    print(f"\nLoading classifier predictions from {classifier_dir}...")
    all_predictions = load_classifier_predictions(classifier_dir, label_maps)
    for clf_type, task_preds in all_predictions.items():
        models_used = set(info["model"] for info in task_preds.values())
        print(f"  {clf_type}: {models_used}")

    # Build few-shot retriever
    retriever = FewShotRetriever(
        embedding_model=config.enhanced.few_shot_embedding_model,
        device=args.few_shot_device,
    )
    if config.enhanced.use_few_shot:
        texts = prepare_texts(df)
        retriever.build_index(texts, df, label_maps)
    else:
        print("Few-shot retrieval DISABLED.")

    # Build meta-ensemble
    pubmedbert_path = os.path.join(classifier_dir, "pubmedbert_results.json")
    tfidf_path = os.path.join(classifier_dir, "tfidf_xgboost_results.json")

    # Load full CSV (before max_patients truncation) for OOF mapping
    full_csv_path = os.path.join(config.data.quality_dataset_dir, f"classification_{args.tier}.csv")
    full_df = load_classification_csv(full_csv_path)

    # Build OOF index maps (CSV row → OOF array index per task)
    oof_maps = build_oof_index_maps(full_df, label_maps)
    for task, mapping in oof_maps.items():
        print(f"  OOF mapping for {task}: {len(mapping)} valid patients")

    ensemble = MetaEnsemble(
        label_maps=label_maps,
        weight_pubmedbert=config.enhanced.ensemble_weight_pubmedbert,
        weight_tfidf=config.enhanced.ensemble_weight_tfidf,
    )
    if config.enhanced.use_ensemble:
        ensemble.load_oof_predictions(pubmedbert_path, tfidf_path)
        ensemble.set_oof_maps(oof_maps)
        ensemble.calibrate(full_df, label_maps)
    else:
        print("Meta-ensemble DISABLED.")

    # Run pipeline(s)
    modes = []
    if args.mode == "both":
        modes = ["single_pass", "multi_turn"]
    else:
        modes = [args.mode]

    for mode in modes:
        mode_save_dir = os.path.join(save_dir, mode)
        run_pipeline(
            mode=mode,
            config=config,
            df=df,
            label_maps=label_maps,
            agent_reports=agent_reports,
            all_predictions=all_predictions,
            retriever=retriever,
            ensemble=ensemble,
            orchestrator_gpus=args.orchestrator_gpus,
            agent_gpus=args.agent_gpus,
            save_dir=mode_save_dir,
            oof_maps=oof_maps,
        )

    print(f"\n{'=' * 60}")
    print("ALL ENHANCED PIPELINE RUNS COMPLETE")
    print(f"Results in: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
