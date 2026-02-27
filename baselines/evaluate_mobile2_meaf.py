#!/usr/bin/env python3
"""
Mobile-2 MEAF Results Evaluation & Table Generation.

Aggregates results from discriminative baselines and MEAF pipeline runs,
then generates Table 2 (CSV + LaTeX) for the paper.

Usage:
    python baselines/evaluate_mobile2_meaf.py --results_dir logs/mobile2_meaf_XXXXXXXX/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_all_results(results_dir: str) -> dict:
    """Scan results_dir for all result files and organize by method."""
    all_methods = {}

    # ── Discriminative baselines ──

    # REVE results
    reve_dir = os.path.join(results_dir, "reve")
    if os.path.isdir(reve_dir):
        reve_metrics = {"method": "REVE (frozen)", "modality": "E"}
        for task in ["ez_region", "stim_intensity", "source_localization"]:
            fpath = os.path.join(reve_dir, f"reve_{task}_results.json")
            if os.path.exists(fpath):
                data = load_json(fpath)
                agg = data.get("aggregate", {})
                if task == "source_localization":
                    reve_metrics[task] = agg.get("mean_error_mm", "N/A")
                else:
                    reve_metrics[task] = agg.get("accuracy", "N/A")
        all_methods["reve"] = reve_metrics

    # Classical baselines (GFP, BandPower)
    baselines_dir = os.path.join(results_dir, "baselines")
    if os.path.isdir(baselines_dir):
        for fname in sorted(os.listdir(baselines_dir)):
            if not fname.endswith("_results.json"):
                continue
            fpath = os.path.join(baselines_dir, fname)
            try:
                data = load_json(fpath)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

            method = data.get("method", "unknown")
            task = data.get("task", "unknown")
            agg = data.get("aggregate", {})

            if method not in all_methods:
                all_methods[method] = {"method": method, "modality": "E"}
            all_methods[method][task] = agg.get("accuracy", "N/A")

    # ── MEAF pipeline results ──
    meaf_files = {
        "text_only": "mobile2_meaf_text_only_results.json",
        "no_disc": "mobile2_meaf_no_disc_results.json",
        "single_pass": "mobile2_meaf_single_pass_results.json",
        # Alternative naming from the --mode flag
        "multi_agent_no_disc": "mobile2_meaf_multi_agent_no_disc_results.json",
        "meaf_single_pass": "mobile2_meaf_meaf_single_pass_results.json",
    }

    # Also scan for any mobile2_meaf_*_results.json files
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("mobile2_meaf_") and fname.endswith("_results.json"):
            key = fname.replace("mobile2_meaf_", "").replace("_results.json", "")
            meaf_files[key] = fname

    meaf_display = {
        "text_only": ("MedGemma Text-Only", "T"),
        "no_disc": ("Multi-Agent LLM", "T+I"),
        "multi_agent_no_disc": ("Multi-Agent LLM", "T+I"),
        "single_pass": ("MEAF Single-Pass", "T+I"),
        "meaf_single_pass": ("MEAF Single-Pass", "T+I"),
        "multi_turn": ("MEAF Multi-Turn", "T+I"),
        "meaf_multi_turn": ("MEAF Multi-Turn", "T+I"),
        "meaf_full": ("MEAF Full", "T+I"),
        "meaf_meaf_full": ("MEAF Full", "T+I"),
    }

    for key, fname in meaf_files.items():
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            continue

        try:
            data = load_json(fpath)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        metrics = data.get("metrics", {})
        display_name, modality = meaf_display.get(key, (key, "T+I"))

        entry = {"method": display_name, "modality": modality}
        for task in ["ez_region", "stim_intensity"]:
            if task in metrics:
                entry[task] = f"{metrics[task]['accuracy']:.1f}"
        if "source_localization" in metrics:
            entry["source_localization"] = f"{metrics['source_localization']['mean_error_mm']:.1f}"
        if "mean_accuracy" in metrics:
            entry["mean_accuracy"] = f"{metrics['mean_accuracy']:.1f}"

        all_methods[display_name] = entry

    return all_methods


def create_comparison_table(methods: dict) -> str:
    """Create a formatted comparison table."""
    # Define display order
    order = [
        "GFP_XGBoost", "BandPower_XGBoost", "reve",
        "MedGemma Text-Only", "Multi-Agent LLM",
        "MEAF Single-Pass", "MEAF Multi-Turn", "MEAF Full",
    ]

    sorted_keys = []
    for key in order:
        if key in methods:
            sorted_keys.append(key)
    for key in methods:
        if key not in sorted_keys:
            sorted_keys.append(key)

    # Header
    lines = []
    header = f"{'Method':<25} {'Mod.':<6} {'EZ Region':>10} {'Stim Int.':>10} {'Src Loc(mm↓)':>13} {'Mean Acc':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for key in sorted_keys:
        m = methods[key]
        name = m.get("method", key)
        mod = m.get("modality", "?")
        ez = m.get("ez_region", "—")
        si = m.get("stim_intensity", "—")
        sl = m.get("source_localization", "—")
        ma = m.get("mean_accuracy", "—")

        # Compute mean accuracy if not set
        if ma == "—":
            accs = []
            for t in ["ez_region", "stim_intensity"]:
                val = m.get(t, "—")
                if val != "—" and val != "N/A":
                    try:
                        # Handle "0.XXX ± Y.YYY" format from LOSO
                        val_str = str(val).split("±")[0].strip()
                        v = float(val_str)
                        if v <= 1.0:
                            v *= 100  # Convert from fraction to %
                        accs.append(v)
                    except (ValueError, IndexError):
                        pass
            if accs:
                ma = f"{np.mean(accs):.1f}"

        lines.append(f"{name:<25} {mod:<6} {str(ez):>10} {str(si):>10} {str(sl):>13} {str(ma):>10}")

    return "\n".join(lines)


def create_latex_table(methods: dict) -> str:
    """Generate LaTeX table for MICCAI paper (Table 2)."""
    order = [
        "GFP_XGBoost", "BandPower_XGBoost", "reve",
        "MedGemma Text-Only", "Multi-Agent LLM",
        "MEAF Single-Pass", "MEAF Multi-Turn", "MEAF Full",
    ]

    sorted_keys = []
    for key in order:
        if key in methods:
            sorted_keys.append(key)
    for key in methods:
        if key not in sorted_keys:
            sorted_keys.append(key)

    method_display = {
        "GFP_XGBoost": "GFP + XGBoost",
        "BandPower_XGBoost": "Band Power + XGBoost",
        "reve": r"REVE~\cite{reve} (frozen)",
        "MedGemma Text-Only": "MedGemma-27B (text only)",
        "Multi-Agent LLM": "Multi-Agent Pipeline",
        "MEAF Single-Pass": "MEAF Single-Pass",
        "MEAF Multi-Turn": "MEAF + Self-Critique",
        "MEAF Full": r"\textbf{MEAF (Ours)}",
    }

    modality_display = {
        "GFP_XGBoost": r"$\text{I}_\text{EEG}$",
        "BandPower_XGBoost": r"$\text{I}_\text{EEG}$",
        "reve": r"$\text{I}_\text{EEG}$",
        "MedGemma Text-Only": "T",
        "Multi-Agent LLM": "T+I",
        "MEAF Single-Pass": "T+I",
        "MEAF Multi-Turn": "T+I",
        "MEAF Full": "T+I",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy (\%) on Mobile-2 external validation (7 subjects, 61 sessions). "
        r"$\text{I}_\text{EEG}$ = EEG signal features; T = clinical text; I = images. "
        r"EEG discriminative models report LOSO CV accuracy. "
        r"Generative methods are zero-shot. \textbf{Bold} = best per column.}",
        r"\label{tab:mobile2}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Mod. & EZ Region & Stim. Int. & Mean \\",
        r"\midrule",
    ]

    for key in sorted_keys:
        m = methods[key]
        display = method_display.get(key, m.get("method", key))
        mod = modality_display.get(key, m.get("modality", "?"))

        ez = m.get("ez_region", "---")
        si = m.get("stim_intensity", "---")

        # Compute mean
        accs = []
        for t in ["ez_region", "stim_intensity"]:
            val = m.get(t, "---")
            if val not in ("---", "N/A", "—"):
                try:
                    val_str = str(val).split("±")[0].strip()
                    v = float(val_str)
                    if v <= 1.0:
                        v *= 100
                    accs.append(v)
                except (ValueError, IndexError):
                    pass
        ma = f"{np.mean(accs):.1f}" if accs else "---"

        lines.append(f"{display} & {mod} & {ez} & {si} & {ma} " + r"\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Mobile-2 MEAF evaluation")
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Mobile-2 MEAF Results Evaluation")
    print("=" * 60)
    print(f"Results dir: {args.results_dir}")

    if not os.path.isdir(args.results_dir):
        print(f"  Error: {args.results_dir} does not exist.")
        sys.exit(1)

    methods = collect_all_results(args.results_dir)

    if not methods:
        print("  No results found.")
        sys.exit(1)

    print(f"\n  Found {len(methods)} methods: {list(methods.keys())}")

    # Print comparison table
    table = create_comparison_table(methods)
    print(f"\n{table}\n")

    # Save CSV
    csv_path = os.path.join(args.results_dir, "mobile2_meaf_comparison.csv")
    with open(csv_path, "w") as f:
        f.write(table)
    print(f"Saved: {csv_path}")

    # Save LaTeX
    latex = create_latex_table(methods)
    latex_path = os.path.join(args.results_dir, "mobile2_meaf_comparison.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved: {latex_path}")

    print(f"\n{'='*60}")
    print("EVALUATION DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
