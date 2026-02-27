#!/usr/bin/env python3
"""
MEAF Pipeline for Mobile-2 Public Dataset.

Applies the Modality-Expert Agent Fusion framework to the Mobile-2 HD-EEG
stimulation dataset using TimeOmni-1-7B as the EEG agent.

Two phases:
  --phase agents:       Generate text/MRI/EEG reports for all sessions
  --phase orchestrator: Run Llama-70B orchestrator with discriminative injection

Ablation flags:
  --no_discriminative:  Omit discriminative predictions from orchestrator
  --no_mri:             Omit MRI agent report
  --no_eeg:             Omit EEG agent report
  --no_ensemble:        Skip meta-ensemble post-processing
  --text_only:          Only use text agent (no MRI, no EEG, no disc)

Usage:
    # Generate agent reports (test on 2 subjects)
    python run_mobile2_meaf.py --phase agents --test_subjects sub-01,sub-02 \\
        --save_dir logs/mobile2_meaf_test

    # Run orchestrator (all subjects)
    python run_mobile2_meaf.py --phase orchestrator --mode single_pass \\
        --agent_reports logs/mobile2_meaf_test/agent_reports.json \\
        --orchestrator_gpus 0,1,2,3,4,5,6,7 \\
        --save_dir logs/mobile2_meaf_test
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from configs.default import Config
from data.mobile2_bids import (
    load_seeg_coords,
    load_hdeeg_electrode_info,
    mni_to_region,
    current_to_class,
    REGION_NAMES,
    INTENSITY_NAMES,
)
from models.mobile2_report_parser import (
    parse_mobile2_to_label_indices,
    parse_source_location,
    MOBILE2_LABEL_MAPS,
)


# ── Mobile-2 Orchestrator (subclass) ──────────────────────────────────────

MOBILE2_SYSTEM_PROMPT = (
    "You are a senior epileptologist integrating multimodal diagnostic reports for "
    "presurgical epilepsy evaluation using HD-EEG and SEEG stimulation data.\n\n"
    "You receive reports from three specialist agents:\n"
    "1) Clinical Text Agent — stimulation protocol, electrode placement, patient demographics\n"
    "2) MRI Agent — structural brain MRI findings (T1w axial slices)\n"
    "3) EEG Agent — HD-EEG spatial analysis during intracranial SEEG stimulation\n\n"
    "CRITICAL DECISION RULES:\n"
    "- For EZ REGION classification: The EEG Agent's SPATIAL ANALYSIS (center of mass, " 
    "estimated source region) is the most reliable evidence. MRI is structural and does NOT " 
    "indicate where the stimulation response is — do NOT use MRI for region classification.\n"
    "- For STIM INTENSITY: The stimulation current (in mA) is stated in the clinical text. "
    "Low = <=0.3mA, High = >=0.5mA. Use that value directly.\n"
    "- For SOURCE COORDINATES: Use the EEG spatial centroid as the estimate.\n"
    "- When a DISCRIMINATIVE MODEL shows STRONG confidence, follow it unless EEG spatial "
    "analysis clearly contradicts.\n"
)

MOBILE2_TASK_INSTRUCTION = (
    "\n=== TASK ===\n"
    "Integrate all evidence above and classify this stimulation session.\n\n"
    "Decision priority for EZ REGION:\n"
    "1. REVE discriminative model with STRONG confidence → follow it\n"
    "2. EEG Agent spatial analysis (center of mass, estimated region) → use as primary signal\n"
    "3. Clinical text → provides context but NOT region-specific information\n"
    "4. MRI Agent → IGNORE for region classification (structural, not functional)\n\n"
    "First briefly reason (2-3 sentences), then output your classification "
    "as a JSON code block:\n"
    "```json\n"
    "{\n"
    '  "ez_region": "Temporal | Frontal | Parieto-Occipital",\n'
    '  "stim_intensity": "Low | High",\n'
    '  "source_x": <estimated MNI x coordinate in mm>,\n'
    '  "source_y": <estimated MNI y coordinate in mm>,\n'
    '  "source_z": <estimated MNI z coordinate in mm>\n'
    "}\n"
    "```"
)


def build_mobile2_orchestrator_messages(
    text_report: str,
    mri_report: str,
    eeg_report: str,
    discriminative_text: str = "",
    rag_context: str = "",
) -> list:
    """Build orchestrator messages for Mobile-2."""
    user_parts = []

    # Discriminative predictions
    if discriminative_text:
        user_parts.append(discriminative_text)

    # Agent reports
    user_parts.append("=== SPECIALIST AGENT REPORTS ===")

    user_parts.append("--- Clinical Text Agent Report ---")
    user_parts.append(text_report.strip() if text_report else "No clinical text data available.")

    user_parts.append("\n--- MRI Agent Report ---")
    user_parts.append(mri_report.strip() if mri_report else "No MRI data available.")

    user_parts.append("\n--- EEG Agent Report ---")
    user_parts.append(eeg_report.strip() if eeg_report else "No EEG data available.")

    # RAG context
    if rag_context:
        user_parts.append(f"\n=== ILAE CLINICAL GUIDELINES ===\n{rag_context}")

    # Task instruction
    user_parts.append(MOBILE2_TASK_INSTRUCTION)

    messages = [
        {"role": "user", "content": MOBILE2_SYSTEM_PROMPT + "\n\n" + "\n".join(user_parts)},
    ]
    return messages


def format_discriminative_predictions(predictions: dict) -> str:
    """Format discriminative predictions as confidence-tiered text.

    Confidence tiers calibrated for 3-class problems (random = 0.33):
      - STRONG: max_prob > 0.60 (nearly 2x random)
      - Moderate: max_prob > 0.45
      - Uncertain: max_prob <= 0.45
    """
    if not predictions:
        return ""

    sections = []
    for clf_name, clf_preds in predictions.items():
        if not clf_preds:
            continue

        lines = [f"{clf_name}:"]
        for task, preds_list in clf_preds.items():
            if not preds_list:
                continue

            max_prob = max(p for _, p in preds_list)
            top_label = max(preds_list, key=lambda x: x[1])[0]

            if max_prob > 0.60:
                tier = f"STRONG RECOMMENDATION: {top_label} ({max_prob:.0%} confidence)"
            elif max_prob > 0.45:
                tier = f"Moderate signal: {top_label} ({max_prob:.0%})"
            else:
                tier = f"Uncertain — split across classes (best: {top_label} at {max_prob:.0%})"

            lines.append(f"  {task}: {tier}")
            for label, prob in sorted(preds_list, key=lambda x: x[1], reverse=True):
                lines.append(f"    P({label}) = {prob:.2f}")

        sections.append("\n".join(lines))

    if not sections:
        return ""

    header = (
        "=== DISCRIMINATIVE MODEL SIGNALS ===\n"
        "These are data-driven probability estimates from a trained EEG classifier (REVE).\n"
        "STRONG RECOMMENDATION (>60% confidence): Follow this unless EEG spatial analysis clearly contradicts.\n"
        "Moderate signal (45-60%): Consider alongside agent reports.\n"
        "Uncertain (<45%): Classifier unsure, rely on agent reports.\n\n"
    )
    return header + "\n\n".join(sections)


# ── GPU Utilities ─────────────────────────────────────────────────────────

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


# ── Data Loading ──────────────────────────────────────────────────────────

def load_mobile2_data(csv_path: str, test_subjects: str = None) -> pd.DataFrame:
    """Load Mobile-2 MEAF CSV, optionally filtering by subject."""
    df = pd.read_csv(csv_path)
    if test_subjects:
        subs = [s.strip() for s in test_subjects.split(",")]
        df = df[df["subject_id"].isin(subs)].reset_index(drop=True)
        print(f"Filtered to {len(df)} sessions from subjects: {subs}")
    print(f"Loaded {len(df)} sessions from {df['subject_id'].nunique()} subjects")
    return df


def load_electrode_positions(subject_id: str, bids_root: str) -> np.ndarray:
    """Load HD-EEG electrode positions for a subject."""
    tsv_path = os.path.join(
        bids_root, "derivatives", "epochs", subject_id, "eeg",
        f"{subject_id}_task-seegstim_electrodes.tsv",
    )
    if os.path.exists(tsv_path):
        names, positions = load_hdeeg_electrode_info(tsv_path)
        return positions
    # Fallback: return zeros (will use default region grouping)
    return np.zeros((256, 3), dtype=np.float32)


# ── Agent Phase ───────────────────────────────────────────────────────────

def run_text_agent_phase(df: pd.DataFrame, config: Config) -> dict:
    """Generate clinical text reports using MedGemma-27B."""
    from models.text_agent import TextAgent

    agent = TextAgent(
        model_name=config.text_agent.model_name,
        system_prompt=config.text_agent.system_prompt,
        max_new_tokens=config.text_agent.max_new_tokens,
        temperature=config.text_agent.temperature,
        do_sample=config.text_agent.do_sample,
        repetition_penalty=config.text_agent.repetition_penalty,
        torch_dtype="bfloat16",
    )
    agent.load_model(device_map="auto", max_memory=parse_gpu_ids("0,1"))

    reports = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Text Agent"):
        pid = row["patient_id"]
        semiology = row.get("semiology_text", "")
        mri_text = row.get("mri_report_text", "")
        eeg_text = row.get("eeg_report_text", "")
        try:
            report = agent.generate_summary(
                semiology=str(semiology) if pd.notna(semiology) else "",
                mri_report=str(mri_text) if pd.notna(mri_text) else "",
                eeg_report=str(eeg_text) if pd.notna(eeg_text) else "",
            )
        except Exception as e:
            print(f"  Text agent error for {pid}: {e}")
            report = f"Error generating report: {e}"
        reports[pid] = report

    # Unload
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def run_mri_agent_phase(df: pd.DataFrame, config: Config) -> dict:
    """Generate MRI reports using MedGemma-4B."""
    from models.mri_agent import MRIAgent
    from PIL import Image

    agent = MRIAgent(
        model_name=config.mri_agent.model_name,
        system_prompt=config.mri_agent.system_prompt,
        max_new_tokens=config.mri_agent.max_new_tokens,
        temperature=config.mri_agent.temperature,
        do_sample=config.mri_agent.do_sample,
        torch_dtype="bfloat16",
    )
    agent.load_model(device=f"cuda:2")

    reports = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MRI Agent"):
        pid = row["patient_id"]
        mri_images_str = row.get("mri_images", "[]")

        images = []
        try:
            img_list = json.loads(mri_images_str) if isinstance(mri_images_str, str) else []
            for img_info in img_list:
                path = img_info.get("path", "") if isinstance(img_info, dict) else str(img_info)
                if path and os.path.exists(path):
                    images.append(Image.open(path).convert("RGB"))
        except (json.JSONDecodeError, Exception):
            pass

        if images:
            try:
                report = agent.generate_report(images)
            except Exception as e:
                print(f"  MRI agent error for {pid}: {e}")
                report = f"MRI analysis error: {e}"
        else:
            report = "No MRI images available for analysis."
        reports[pid] = report

    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def run_eeg_agent_phase(df: pd.DataFrame, config: Config) -> dict:
    """Generate EEG reports using TimeOmni-1-7B."""
    from models.timeomni_eeg_agent import TimeOmniEEGAgent

    bids_root = config.mobile2_bids.bids_root
    target_sr = config.mobile2_bids.target_sample_rate
    original_sr = config.mobile2_bids.original_sample_rate

    agent = TimeOmniEEGAgent(
        model_name="anton-hugging/TimeOmni-1-7B",
        max_new_tokens=2048,
        temperature=0.1,
        target_sr=target_sr,
    )
    agent.load_model(device="cuda:3")

    # Cache electrode positions per subject
    positions_cache = {}

    reports = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="EEG Agent (TimeOmni)"):
        pid = row["patient_id"]
        subject_id = row["subject_id"]

        # Load electrode positions
        if subject_id not in positions_cache:
            positions_cache[subject_id] = load_electrode_positions(subject_id, bids_root)

        positions = positions_cache[subject_id]
        npy_path = row.get("npy_path", "")

        if not npy_path or not os.path.exists(npy_path):
            reports[pid] = "No EEG data available for analysis."
            continue

        try:
            # Load and preprocess EEG
            epochs = np.load(npy_path)  # (n_trials, 256, 2081) at 8kHz
            mean_erp = epochs.mean(axis=0)  # (256, 2081)

            # Downsample: 8kHz -> target_sr
            factor = original_sr // target_sr
            downsampled = decimate(mean_erp, factor, axis=-1, zero_phase=True)

            # Generate report
            report = agent.generate_report(
                raw_eeg=downsampled,
                electrode_positions=positions,
                current_mA=row.get("current_mA", 1.0),
                electrode_pair=f"{row.get('electrode1', '?')}-{row.get('electrode2', '?')}",
                subject_id=subject_id,
            )
        except Exception as e:
            print(f"  EEG agent error for {pid}: {e}")
            report = f"EEG analysis error: {e}"

        reports[pid] = report

    agent.unload_model()
    return reports


# ── Orchestrator Phase ────────────────────────────────────────────────────

def load_reve_oof_predictions(reve_dir: str) -> dict:
    """Load REVE OOF predictions from training results."""
    oof = {}
    for task in ["ez_region", "stim_intensity"]:
        fpath = os.path.join(reve_dir, f"reve_{task}_results.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        oof_preds = data.get("oof_predictions", {})
        if oof_preds:
            oof[task] = oof_preds
    return oof


def load_reve_source_oof(reve_dir: str) -> dict:
    """Load REVE source localization OOF predictions.

    Returns:
        dict mapping subject_id -> list of (x, y, z) predictions per run
    """
    fpath = os.path.join(reve_dir, "reve_source_localization_results.json")
    if not os.path.exists(fpath):
        return {}
    with open(fpath) as f:
        data = json.load(f)
    oof_preds = data.get("oof_predictions", {})
    source_oof = {}
    for subject_id, sub_data in oof_preds.items():
        preds = sub_data.get("preds", [])
        if preds:
            source_oof[subject_id] = preds  # list of [x, y, z] per run
    return source_oof


def get_reve_source_for_session(row: pd.Series, reve_source_oof: dict) -> tuple:
    """Get REVE source localization prediction for a session.

    Returns:
        (x, y, z) tuple or None
    """
    if not reve_source_oof:
        return None
    subject_id = row["subject_id"]
    preds = reve_source_oof.get(subject_id)
    if preds is None:
        return None

    run_id = row.get("run_id", "")
    run_idx = 0
    try:
        run_num = int(run_id.replace("run-", "")) - 1
        run_idx = min(run_num, len(preds) - 1)
    except (ValueError, AttributeError):
        pass

    if run_idx < len(preds) and len(preds[run_idx]) == 3:
        return tuple(float(v) for v in preds[run_idx])
    return None


def get_session_discriminative_predictions(
    row: pd.Series,
    reve_oof: dict,
    label_maps: dict,
) -> dict:
    """Get discriminative predictions for a single session."""
    subject_id = row["subject_id"]
    predictions = {}

    # REVE predictions (from LOSO OOF)
    reve_preds = {}
    for task in ["ez_region", "stim_intensity"]:
        if task not in reve_oof:
            continue
        sub_oof = reve_oof[task].get(subject_id, {})
        probs = sub_oof.get("probs")
        if probs is None:
            continue

        # probs is a list of arrays (one per sample in the held-out subject)
        # We need to find the right sample index within this subject's runs
        # For now, use the first available (run-averaged gives 1 prob per run)
        inv_map = {v: k for k, v in label_maps[task].items()}
        probs_arr = np.array(probs)
        if probs_arr.ndim == 2 and probs_arr.shape[0] > 0:
            # Get the correct run index within this subject
            run_id = row.get("run_id", "")
            # Simple: use the run index based on order
            run_idx = 0  # Will be refined below
            try:
                run_num = int(run_id.replace("run-", "")) - 1
                run_idx = min(run_num, probs_arr.shape[0] - 1)
            except (ValueError, AttributeError):
                pass

            if run_idx < probs_arr.shape[0]:
                run_probs = probs_arr[run_idx]
                task_preds = []
                for cls_idx, prob in enumerate(run_probs):
                    label = inv_map.get(cls_idx, f"class_{cls_idx}")
                    task_preds.append((label, float(prob)))
                reve_preds[task] = task_preds

    if reve_preds:
        predictions["REVE EEG Foundation Model"] = reve_preds

    return predictions


def apply_ensemble_override(
    parsed: dict,
    source_loc: tuple,
    disc_predictions: dict,
    reve_source_pred: tuple = None,
    confidence_threshold: float = 0.60,
) -> tuple:
    """Apply ensemble post-processing: override orchestrator when REVE is confident.

    For ez_region: if REVE probability > threshold and orchestrator disagrees, use REVE.
    For source_localization: always prefer REVE's predicted coordinates over LLM hallucination.

    Args:
        parsed: dict of {task: label_index} from orchestrator
        source_loc: (x, y, z) from orchestrator or None
        disc_predictions: dict from get_session_discriminative_predictions
        reve_source_pred: (x, y, z) from REVE source localization or None
        confidence_threshold: min probability to override orchestrator

    Returns:
        (parsed_updated, source_loc_updated, override_info)
    """
    override_info = {}
    parsed_updated = dict(parsed)

    reve_preds = disc_predictions.get("REVE EEG Foundation Model", {})

    for task in ["ez_region", "stim_intensity"]:
        if task not in reve_preds:
            continue

        preds_list = reve_preds[task]
        if not preds_list:
            continue

        # Find max probability prediction
        top_label, top_prob = max(preds_list, key=lambda x: x[1])

        if top_prob > confidence_threshold:
            # Convert label name to index
            from models.mobile2_report_parser import MOBILE2_LABEL_MAPS
            label_map = MOBILE2_LABEL_MAPS.get(task, {})
            reve_idx = label_map.get(top_label, -1)

            if reve_idx >= 0 and reve_idx != parsed.get(task, -1):
                override_info[task] = {
                    "action": "override",
                    "orchestrator_pred": parsed.get(task, -1),
                    "reve_pred": reve_idx,
                    "reve_label": top_label,
                    "reve_prob": top_prob,
                }
                parsed_updated[task] = reve_idx

    # Source localization: always prefer REVE if available
    source_loc_updated = source_loc
    if reve_source_pred is not None:
        source_loc_updated = reve_source_pred
        override_info["source_localization"] = {
            "action": "use_reve",
            "orchestrator_pred": source_loc,
            "reve_pred": reve_source_pred,
        }

    return parsed_updated, source_loc_updated, override_info


def run_orchestrator_phase(
    df: pd.DataFrame,
    agent_reports: dict,
    config: Config,
    args,
    save_dir: str,
) -> dict:
    """Run orchestrator on all sessions."""
    from models.hybrid_orchestrator import HybridOrchestrator
    from models.rag import ILAEKnowledgeBase

    label_maps = MOBILE2_LABEL_MAPS

    # Load REVE OOF (if available and not disabled)
    reve_oof = {}
    reve_source_oof = {}
    if not args.no_discriminative and args.reve_dir:
        reve_oof = load_reve_oof_predictions(args.reve_dir)
        reve_source_oof = load_reve_source_oof(args.reve_dir)
        print(f"Loaded REVE OOF for tasks: {list(reve_oof.keys())}")
        if reve_source_oof:
            print(f"Loaded REVE source OOF for {len(reve_source_oof)} subjects")

    # Load orchestrator
    print("Loading Orchestrator (Llama-3.3-70B)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    orch_model_name = config.orchestrator.model_name
    max_memory = parse_gpu_ids(args.orchestrator_gpus)

    tokenizer = AutoTokenizer.from_pretrained(orch_model_name, token=config.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        orch_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        token=config.hf_token,
    )
    model.eval()
    print(f"Orchestrator loaded: {orch_model_name}")

    # Build RAG index
    kb = ILAEKnowledgeBase(embedding_model=config.rag.embedding_model)
    kb.build_index()

    # Process each session
    results = []
    diagnoses = []
    parsed_labels_list = []
    gt_labels_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Orchestrator"):
        pid = row["patient_id"]

        # Get agent reports
        text_report = agent_reports.get("text", {}).get(pid, "")
        mri_report = "" if args.no_mri else agent_reports.get("mri", {}).get(pid, "")
        eeg_report = "" if args.no_eeg else agent_reports.get("eeg", {}).get(pid, "")

        if args.text_only:
            mri_report = ""
            eeg_report = ""

        # Truncate long reports
        text_report = text_report[:2000] if text_report else ""
        mri_report = mri_report[:1500] if mri_report else ""
        eeg_report = eeg_report[:2000] if eeg_report else ""

        # Get discriminative predictions
        disc_preds = {}
        if not args.no_discriminative and not args.text_only:
            disc_preds = get_session_discriminative_predictions(row, reve_oof, label_maps)

        disc_text = format_discriminative_predictions(disc_preds)

        # RAG context
        rag_query = f"Clinical: {text_report[:300]} MRI: {mri_report[:300]} EEG: {eeg_report[:300]}"
        rag_context = kb.retrieve_formatted(rag_query, top_k=3)

        # Build messages
        messages = build_mobile2_orchestrator_messages(
            text_report=text_report,
            mri_report=mri_report,
            eeg_report=eeg_report,
            discriminative_text=disc_text,
            rag_context=rag_context,
        )

        # Generate
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.orchestrator.max_new_tokens,
                do_sample=False,
                temperature=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        diagnosis = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse labels
        parsed = parse_mobile2_to_label_indices(diagnosis, label_maps)
        source_loc = parse_source_location(diagnosis)

        # Ensemble override: when REVE is confident, use its prediction
        override_info = {}
        if disc_preds and not args.no_discriminative:
            reve_source_pred = get_reve_source_for_session(row, reve_source_oof)
            parsed, source_loc, override_info = apply_ensemble_override(
                parsed, source_loc, disc_preds,
                reve_source_pred=reve_source_pred,
                confidence_threshold=0.40,
            )
            if override_info:
                print(f"  {pid}: Ensemble override — {override_info}")

        # Ground truth
        gt = {
            "ez_region": int(row.get("ez_region_label", -1)),
            "stim_intensity": int(row.get("stim_intensity_label", -1)),
        }
        gt_source = None
        if "source_x" in row and pd.notna(row["source_x"]):
            gt_source = (float(row["source_x"]), float(row["source_y"]), float(row["source_z"]))

        diagnoses.append(diagnosis)
        parsed_labels_list.append(parsed)
        gt_labels_list.append(gt)

        results.append({
            "patient_id": pid,
            "subject_id": row["subject_id"],
            "parsed": parsed,
            "ground_truth": gt,
            "source_pred": source_loc,
            "source_gt": gt_source,
            "ensemble_override": override_info if override_info else None,
        })

    # Compute metrics
    metrics = compute_metrics(results, label_maps)

    # Save
    output = {
        "config": {
            "mode": args.mode,
            "text_only": args.text_only,
            "no_discriminative": args.no_discriminative,
            "no_mri": args.no_mri,
            "no_eeg": args.no_eeg,
        },
        "metrics": metrics,
        "results": results,
        "diagnoses": diagnoses,
    }

    os.makedirs(save_dir, exist_ok=True)
    suffix = ""
    if args.text_only:
        suffix = "_text_only"
    elif args.no_discriminative:
        suffix = "_no_disc"
    else:
        suffix = f"_{args.mode}"

    out_path = os.path.join(save_dir, f"mobile2_meaf{suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results: {out_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output


def compute_metrics(results: list, label_maps: dict) -> dict:
    """Compute per-task accuracy and source localization error."""
    metrics = {}

    # Classification tasks
    for task in ["ez_region", "stim_intensity"]:
        correct = 0
        total = 0
        for r in results:
            pred = r["parsed"].get(task, -1)
            gt = r["ground_truth"].get(task, -1)
            if pred >= 0 and gt >= 0:
                total += 1
                if pred == gt:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        metrics[task] = {
            "accuracy": round(acc * 100, 1),
            "correct": correct,
            "total": total,
        }

    # Source localization
    errors = []
    for r in results:
        pred = r.get("source_pred")
        gt = r.get("source_gt")
        if pred is not None and gt is not None:
            error = np.sqrt(sum((p - g) ** 2 for p, g in zip(pred, gt)))
            errors.append(error)

    if errors:
        metrics["source_localization"] = {
            "mean_error_mm": round(float(np.mean(errors)), 1),
            "median_error_mm": round(float(np.median(errors)), 1),
            "within_20mm": round(float(np.mean([e < 20 for e in errors]) * 100), 1),
            "n_samples": len(errors),
        }

    # Mean accuracy (classification only)
    class_accs = [metrics[t]["accuracy"] for t in ["ez_region", "stim_intensity"]
                  if t in metrics and metrics[t]["total"] > 0]
    metrics["mean_accuracy"] = round(np.mean(class_accs), 1) if class_accs else 0.0

    return metrics


# ── Multi-Turn Self-Critique ──────────────────────────────────────────────

SELF_CRITIQUE_PROMPT = (
    "\n=== SELF-REVIEW ===\n"
    "You previously generated this diagnosis:\n{prev_diagnosis}\n\n"
    "Review your diagnosis against the evidence. Check for:\n"
    "1. Discordance between your classification and the discriminative model signals\n"
    "2. Whether your source coordinates are plausible given the EEG topography\n"
    "3. Whether the stimulation intensity matches the reported current\n\n"
    "If you are confident, re-output the SAME classification. If you find errors, "
    "correct them. Output your final classification as a JSON code block:\n"
    "```json\n"
    "{\n"
    '  "ez_region": "Temporal | Frontal | Parieto-Occipital",\n'
    '  "stim_intensity": "Low | High",\n'
    '  "source_x": <MNI x mm>,\n'
    '  "source_y": <MNI y mm>,\n'
    '  "source_z": <MNI z mm>\n'
    "}\n"
    "```"
)


def run_multi_turn_orchestrator_phase(
    df, agent_reports, config, args, save_dir,
):
    """Multi-turn: first pass → self-critique → final pass."""
    from models.rag import ILAEKnowledgeBase
    from transformers import AutoModelForCausalLM, AutoTokenizer

    label_maps = MOBILE2_LABEL_MAPS

    reve_oof = {}
    if not args.no_discriminative and args.reve_dir:
        reve_oof = load_reve_oof_predictions(args.reve_dir)
        print(f"Loaded REVE OOF for tasks: {list(reve_oof.keys())}")

    print("Loading Orchestrator (Llama-3.3-70B)...")
    orch_model_name = config.orchestrator.model_name
    max_memory = parse_gpu_ids(args.orchestrator_gpus)

    tokenizer = AutoTokenizer.from_pretrained(orch_model_name, token=config.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        orch_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        token=config.hf_token,
    )
    model.eval()

    kb = ILAEKnowledgeBase(embedding_model=config.rag.embedding_model)
    kb.build_index()

    results = []
    diagnoses = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Multi-Turn Orchestrator"):
        pid = row["patient_id"]

        text_report = agent_reports.get("text", {}).get(pid, "")[:2000]
        mri_report = agent_reports.get("mri", {}).get(pid, "")[:1500]
        eeg_report = agent_reports.get("eeg", {}).get(pid, "")[:2000]

        disc_preds = {}
        if not args.no_discriminative:
            disc_preds = get_session_discriminative_predictions(row, reve_oof, label_maps)
        disc_text = format_discriminative_predictions(disc_preds)

        rag_query = f"Clinical: {text_report[:300]} EEG: {eeg_report[:300]}"
        rag_context = kb.retrieve_formatted(rag_query, top_k=3)

        # ── Round 1: Initial diagnosis ──
        messages_r1 = build_mobile2_orchestrator_messages(
            text_report=text_report, mri_report=mri_report, eeg_report=eeg_report,
            discriminative_text=disc_text, rag_context=rag_context,
        )
        inputs_r1 = tokenizer.apply_chat_template(
            messages_r1, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        )
        inputs_r1 = {k: v.to(model.device) for k, v in inputs_r1.items()}
        with torch.no_grad():
            out_r1 = model.generate(
                **inputs_r1, max_new_tokens=config.orchestrator.max_new_tokens,
                do_sample=False, temperature=None, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_r1 = out_r1[0, inputs_r1["input_ids"].shape[1]:]
        diag_r1 = tokenizer.decode(gen_r1, skip_special_tokens=True).strip()

        # ── Round 2: Self-critique ──
        critique_content = (
            messages_r1[0]["content"] + "\n\n"
            + SELF_CRITIQUE_PROMPT.replace("{prev_diagnosis}", diag_r1[:1500])
        )
        messages_r2 = [{"role": "user", "content": critique_content}]
        inputs_r2 = tokenizer.apply_chat_template(
            messages_r2, return_tensors="pt", add_generation_prompt=True, return_dict=True,
        )
        inputs_r2 = {k: v.to(model.device) for k, v in inputs_r2.items()}
        with torch.no_grad():
            out_r2 = model.generate(
                **inputs_r2, max_new_tokens=config.orchestrator.max_new_tokens,
                do_sample=False, temperature=None, use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_r2 = out_r2[0, inputs_r2["input_ids"].shape[1]:]
        diagnosis = tokenizer.decode(gen_r2, skip_special_tokens=True).strip()

        parsed = parse_mobile2_to_label_indices(diagnosis, label_maps)
        source_loc = parse_source_location(diagnosis)

        gt = {
            "ez_region": int(row.get("ez_region_label", -1)),
            "stim_intensity": int(row.get("stim_intensity_label", -1)),
        }
        gt_source = None
        if "source_x" in row and pd.notna(row["source_x"]):
            gt_source = (float(row["source_x"]), float(row["source_y"]), float(row["source_z"]))

        diagnoses.append(diagnosis)
        results.append({
            "patient_id": pid, "subject_id": row["subject_id"],
            "parsed": parsed, "ground_truth": gt,
            "source_pred": source_loc, "source_gt": gt_source,
        })

    metrics = compute_metrics(results, label_maps)

    output = {
        "config": {"mode": "multi_turn", "text_only": args.text_only,
                    "no_discriminative": args.no_discriminative},
        "metrics": metrics, "results": results, "diagnoses": diagnoses,
    }
    out_path = os.path.join(save_dir, "mobile2_meaf_multi_turn_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved multi-turn results: {out_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output


# ── MEAF Full: Meta-Ensemble ─────────────────────────────────────────────

MEAF_ENSEMBLE_WEIGHTS = {
    "ez_region": {"disc": 0.75, "gen": 0.25},
    "stim_intensity": {"disc": 0.35, "gen": 0.65},
}


def run_meaf_full_phase(df, agent_reports, config, args, save_dir):
    """MEAF Full: run orchestrator + meta-ensemble with REVE probabilities.

    1. Run multi-prompt orchestrator (3 prompt variants, majority vote)
    2. Meta-ensemble: combine REVE OOF probs + orchestrator prediction
    """
    from models.rag import ILAEKnowledgeBase
    from transformers import AutoModelForCausalLM, AutoTokenizer

    label_maps = MOBILE2_LABEL_MAPS

    reve_oof = {}
    reve_source_oof = {}
    if args.reve_dir:
        reve_oof = load_reve_oof_predictions(args.reve_dir)
        reve_source_oof = load_reve_source_oof(args.reve_dir)
        print(f"Loaded REVE OOF for tasks: {list(reve_oof.keys())}")
        if reve_source_oof:
            print(f"Loaded REVE source OOF for {len(reve_source_oof)} subjects")

    print("Loading Orchestrator (Llama-3.3-70B)...")
    orch_model_name = config.orchestrator.model_name
    max_memory = parse_gpu_ids(args.orchestrator_gpus)

    tokenizer = AutoTokenizer.from_pretrained(orch_model_name, token=config.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        orch_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        token=config.hf_token,
    )
    model.eval()

    kb = ILAEKnowledgeBase(embedding_model=config.rag.embedding_model)
    kb.build_index()

    # Prompt variants for multi-prompt self-consistency
    PROMPT_VARIANTS = [
        MOBILE2_TASK_INSTRUCTION,  # Default
        (
            "\n=== TASK ===\n"
            "Integrate all evidence. Pay special attention to the discriminative model's "
            "confidence levels. If STRONG RECOMMENDATION, follow it unless there is "
            "overwhelming contradicting evidence from agents.\n\n"
            "First reason briefly, then output JSON:\n"
            "```json\n"
            '{"ez_region": "...", "stim_intensity": "...", '
            '"source_x": ..., "source_y": ..., "source_z": ...}\n'
            "```"
        ),
        (
            "\n=== TASK ===\n"
            "You are making a clinical classification. Consider:\n"
            "- EEG spatial topography for region localization\n"
            "- Stimulation current amplitude for intensity classification\n"
            "- MRI findings for structural context\n"
            "- Discriminative model probabilities as Bayesian priors\n\n"
            "Output JSON classification:\n"
            "```json\n"
            '{"ez_region": "Temporal|Frontal|Parieto-Occipital", '
            '"stim_intensity": "Low|High", '
            '"source_x": 0, "source_y": 0, "source_z": 0}\n'
            "```"
        ),
    ]

    results = []
    diagnoses = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MEAF Full"):
        pid = row["patient_id"]
        subject_id = row["subject_id"]

        text_report = agent_reports.get("text", {}).get(pid, "")[:2000]
        mri_report = agent_reports.get("mri", {}).get(pid, "")[:1500]
        eeg_report = agent_reports.get("eeg", {}).get(pid, "")[:2000]

        disc_preds = get_session_discriminative_predictions(row, reve_oof, label_maps)
        disc_text = format_discriminative_predictions(disc_preds)

        rag_query = f"Clinical: {text_report[:300]} EEG: {eeg_report[:300]}"
        rag_context = kb.retrieve_formatted(rag_query, top_k=3)

        # ── Multi-prompt: run 3 variants ──
        all_parsed = []
        all_sources = []
        best_diagnosis = ""

        for vi, task_instr in enumerate(PROMPT_VARIANTS):
            user_parts = []
            if disc_text:
                user_parts.append(disc_text)
            user_parts.append("=== SPECIALIST AGENT REPORTS ===")
            user_parts.append("--- Clinical Text Agent Report ---")
            user_parts.append(text_report.strip() if text_report else "No data.")
            user_parts.append("\n--- MRI Agent Report ---")
            user_parts.append(mri_report.strip() if mri_report else "No data.")
            user_parts.append("\n--- EEG Agent Report ---")
            user_parts.append(eeg_report.strip() if eeg_report else "No data.")
            if rag_context:
                user_parts.append(f"\n=== ILAE CLINICAL GUIDELINES ===\n{rag_context}")
            user_parts.append(task_instr)

            messages = [{"role": "user",
                         "content": MOBILE2_SYSTEM_PROMPT + "\n\n" + "\n".join(user_parts)}]

            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt",
                add_generation_prompt=True, return_dict=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=config.orchestrator.max_new_tokens,
                    do_sample=(vi > 0), temperature=0.3 if vi > 0 else None,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen = out[0, inputs["input_ids"].shape[1]:]
            diag = tokenizer.decode(gen, skip_special_tokens=True).strip()
            if vi == 0:
                best_diagnosis = diag

            p = parse_mobile2_to_label_indices(diag, label_maps)
            s = parse_source_location(diag)
            all_parsed.append(p)
            all_sources.append(s)

        # ── Majority vote from multi-prompt ──
        orch_parsed = {}
        for task in ["ez_region", "stim_intensity"]:
            votes = [p.get(task, -1) for p in all_parsed]
            valid = [v for v in votes if v >= 0]
            if valid:
                orch_parsed[task] = Counter(valid).most_common(1)[0][0]
            else:
                orch_parsed[task] = -1

        # Source: average of valid predictions
        valid_sources = [s for s in all_sources if s is not None]
        orch_source = None
        if valid_sources:
            orch_source = tuple(
                float(np.mean([s[d] for s in valid_sources])) for d in range(3)
            )

        # ── Meta-ensemble: combine REVE + orchestrator ──
        final_parsed = {}
        for task in ["ez_region", "stim_intensity"]:
            reve_probs = _get_reve_probs_for_session(row, reve_oof, task, label_maps)

            if reve_probs is not None and orch_parsed.get(task, -1) >= 0:
                n_classes = len(reve_probs)
                w = MEAF_ENSEMBLE_WEIGHTS.get(task, {"disc": 0.5, "gen": 0.5})

                # Build orchestrator soft label from multi-prompt votes
                vote_counts = Counter(
                    p.get(task, -1) for p in all_parsed if p.get(task, -1) >= 0
                )
                total_votes = sum(vote_counts.values())
                orch_soft = np.zeros(n_classes)
                for cls, cnt in vote_counts.items():
                    if 0 <= cls < n_classes:
                        orch_soft[cls] = cnt / total_votes

                combined = w["disc"] * np.array(reve_probs) + w["gen"] * orch_soft
                final_parsed[task] = int(np.argmax(combined))
            elif reve_probs is not None:
                final_parsed[task] = int(np.argmax(reve_probs))
            else:
                final_parsed[task] = orch_parsed.get(task, -1)

        # Source: prefer REVE source prediction over LLM-hallucinated coordinates
        reve_source_pred = get_reve_source_for_session(row, reve_source_oof)
        final_source = reve_source_pred if reve_source_pred is not None else orch_source

        gt = {
            "ez_region": int(row.get("ez_region_label", -1)),
            "stim_intensity": int(row.get("stim_intensity_label", -1)),
        }
        gt_source = None
        if "source_x" in row and pd.notna(row["source_x"]):
            gt_source = (float(row["source_x"]), float(row["source_y"]),
                         float(row["source_z"]))

        diagnoses.append(best_diagnosis)
        results.append({
            "patient_id": pid, "subject_id": subject_id,
            "parsed": final_parsed, "ground_truth": gt,
            "source_pred": final_source, "source_gt": gt_source,
        })

    metrics = compute_metrics(results, label_maps)

    output = {
        "config": {"mode": "meaf_full", "meta_ensemble": True,
                    "multi_prompt": True, "n_prompts": len(PROMPT_VARIANTS)},
        "metrics": metrics, "results": results, "diagnoses": diagnoses,
    }
    out_path = os.path.join(save_dir, "mobile2_meaf_meaf_full_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved MEAF Full results: {out_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _get_reve_probs_for_session(row, reve_oof, task, label_maps):
    """Get REVE OOF probabilities for a single session."""
    subject_id = row["subject_id"]
    if task not in reve_oof:
        return None
    sub_oof = reve_oof[task].get(subject_id, {})
    probs = sub_oof.get("probs")
    if probs is None:
        return None
    probs_arr = np.array(probs)
    if probs_arr.ndim < 2 or probs_arr.shape[0] == 0:
        return None

    run_id = row.get("run_id", "")
    run_idx = 0
    try:
        run_num = int(run_id.replace("run-", "")) - 1
        run_idx = min(run_num, probs_arr.shape[0] - 1)
    except (ValueError, AttributeError):
        pass

    if run_idx < probs_arr.shape[0]:
        return probs_arr[run_idx].tolist()
    return None


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mobile-2 MEAF Pipeline")
    parser.add_argument("--phase", choices=["agents", "orchestrator", "both"], default="both")
    parser.add_argument("--mode", choices=["single_pass", "meaf_full"],
                        default="single_pass")
    parser.add_argument("--test_subjects", type=str, default=None,
                        help="Comma-separated subject IDs for testing (e.g., sub-01,sub-02)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--agent_reports", type=str, default=None,
                        help="Path to cached agent_reports.json (skip agent phase)")
    parser.add_argument("--reve_dir", type=str, default=None,
                        help="Path to REVE results dir with OOF predictions")
    parser.add_argument("--orchestrator_gpus", type=str, default="0,1,2,3,4,5,6,7")

    # Ablation flags
    parser.add_argument("--no_discriminative", action="store_true")
    parser.add_argument("--no_mri", action="store_true")
    parser.add_argument("--no_eeg", action="store_true")
    parser.add_argument("--no_ensemble", action="store_true")
    parser.add_argument("--text_only", action="store_true",
                        help="Text agent only (no MRI, EEG, or discriminative)")

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    os.makedirs(args.save_dir, exist_ok=True)

    csv_path = os.path.join(
        str(Path(__file__).parent), "data", "mobile2_classification.csv"
    )

    print("=" * 60)
    print("Mobile-2 MEAF Pipeline with TimeOmni EEG Agent")
    print("=" * 60)
    print(f"Phase:    {args.phase}")
    print(f"Mode:     {args.mode}")
    print(f"Save dir: {args.save_dir}")
    if args.test_subjects:
        print(f"Test:     {args.test_subjects}")
    print(f"Flags:    text_only={args.text_only} no_disc={args.no_discriminative} "
          f"no_mri={args.no_mri} no_eeg={args.no_eeg}")
    print("=" * 60)

    # Load data
    df = load_mobile2_data(csv_path, args.test_subjects)

    # ── Agent Phase ──
    if args.phase in ("agents", "both"):
        print("\n>>> AGENT PHASE <<<")
        t0 = time.time()

        print("\n--- Text Agent (MedGemma-27B) ---")
        text_reports = run_text_agent_phase(df, config)

        print("\n--- MRI Agent (MedGemma-4B) ---")
        mri_reports = run_mri_agent_phase(df, config)

        print("\n--- EEG Agent (TimeOmni-7B) ---")
        eeg_reports = run_eeg_agent_phase(df, config)

        # Save agent reports
        agent_reports = {
            "text": text_reports,
            "mri": mri_reports,
            "eeg": eeg_reports,
        }
        reports_path = os.path.join(args.save_dir, "agent_reports.json")
        with open(reports_path, "w") as f:
            json.dump(agent_reports, f, indent=2)
        print(f"\nAgent reports saved: {reports_path}")
        print(f"Agent phase elapsed: {time.time() - t0:.0f}s")
    else:
        agent_reports = None

    # ── Orchestrator Phase ──
    if args.phase in ("orchestrator", "both"):
        print("\n>>> ORCHESTRATOR PHASE <<<")

        # Load agent reports: use in-memory if available (from "both" mode),
        # otherwise load from file
        if agent_reports is None:
            if args.agent_reports and os.path.exists(args.agent_reports):
                with open(args.agent_reports) as f:
                    agent_reports = json.load(f)
                print(f"Loaded cached agent reports from {args.agent_reports}")
            else:
                print("ERROR: No agent reports. Run --phase agents first or provide --agent_reports.")
                sys.exit(1)

        t0 = time.time()

        if args.mode == "meaf_full":
            # MEAF Full: multi-prompt + meta-ensemble
            run_meaf_full_phase(
                df=df,
                agent_reports=agent_reports,
                config=config,
                args=args,
                save_dir=args.save_dir,
            )
        elif args.phase == "both" and not args.text_only and not args.no_discriminative:
            # Run all ablation configs sequentially
            import copy
            configs = [
                ("text_only", {"text_only": True, "no_discriminative": True, "no_mri": True, "no_eeg": True}),
                ("multi_agent_no_disc", {"text_only": False, "no_discriminative": True, "no_mri": False, "no_eeg": False}),
                ("meaf_single_pass", {"text_only": False, "no_discriminative": False, "no_mri": False, "no_eeg": False}),
            ]
            for config_name, flags in configs:
                print(f"\n--- Running ablation: {config_name} ---")
                abl_args = copy.copy(args)
                for k, v in flags.items():
                    setattr(abl_args, k, v)
                # Bug fix: always use "single_pass" as mode for orchestrator
                abl_args.mode = "single_pass"

                run_orchestrator_phase(
                    df=df,
                    agent_reports=agent_reports,
                    config=config,
                    args=abl_args,
                    save_dir=args.save_dir,
                )
        else:
            run_orchestrator_phase(
                df=df,
                agent_reports=agent_reports,
                config=config,
                args=args,
                save_dir=args.save_dir,
            )

        print(f"\nOrchestrator phase elapsed: {time.time() - t0:.0f}s")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results in: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
