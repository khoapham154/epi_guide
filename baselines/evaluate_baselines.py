"""
Baseline Comparison Script.

Loads results from all baseline evaluations (generative agents + discriminative
classifiers + hybrid) and generates comparison tables for the paper.

Usage:
    python baselines/evaluate_baselines.py --results_dir logs/baselines/full_gold
    python baselines/evaluate_baselines.py --results_dir logs/baselines/full_gold --classifier_dir logs/baselines/classifiers_gold
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_baseline_results(results_dir: str) -> dict:
    """Load all baseline result JSON files."""
    results = {}

    for name, fname in [
        ("text", "text_baseline_results.json"),
        ("mri", "mri_baseline_results.json"),
        ("eeg_image", "eeg_image_baseline_results.json"),
        ("pipeline", "pipeline_results.json"),
    ]:
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                results[name] = json.load(f)
            print(f"Loaded {name}: {fpath}")

    return results


def load_classifier_results(classifier_dir: str) -> dict:
    """Load discriminative classifier results."""
    results = {}

    for name, fname in [
        ("tfidf_xgb", "tfidf_xgboost_results.json"),
        ("pubmedbert", "pubmedbert_results.json"),
        ("mri_resnet", "mri_resnet_results.json"),
        ("eeg_resnet", "eeg_resnet_results.json"),
        ("mri_medsiglip", "medsiglip_mri_results.json"),
        ("eeg_medsiglip", "medsiglip_eeg_results.json"),
        ("ensemble", "ensemble_results.json"),
    ]:
        fpath = os.path.join(classifier_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                results[name] = json.load(f)
            print(f"Loaded {name}: {fpath}")

    return results


def load_hybrid_results(hybrid_dir: str) -> dict:
    """Load hybrid pipeline results."""
    fpath = os.path.join(hybrid_dir, "hybrid_results.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        print(f"Loaded hybrid: {fpath}")
        return data
    return None


def load_multiturn_results(multiturn_dir: str) -> dict:
    """Load multi-turn pipeline results."""
    fpath = os.path.join(multiturn_dir, "multiturn_results.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        print(f"Loaded multi-turn: {fpath}")
        return data
    return None


def create_full_comparison(baseline_results, classifier_results, hybrid_results, multiturn_results=None):
    """Create comprehensive comparison table across all approaches."""
    all_tasks = set()

    # Collect all tasks
    for data in baseline_results.values():
        if "metrics" in data:
            all_tasks.update(data["metrics"].keys())
    for data in classifier_results.values():
        all_tasks.update(data.keys())
    all_tasks = sorted(all_tasks)

    rows = []
    for task in all_tasks:
        row = {"task": task}

        # Generative baselines
        for name in ["text", "mri", "eeg_image", "pipeline"]:
            if name in baseline_results and task in baseline_results[name].get("metrics", {}):
                m = baseline_results[name]["metrics"][task]
                row[f"gen_{name}"] = f"{m['accuracy']:.1%}"
                row[f"gen_{name}_n"] = m["n_samples"]
            else:
                row[f"gen_{name}"] = "-"
                row[f"gen_{name}_n"] = 0

        # Discriminative baselines
        for name in ["tfidf_xgb", "pubmedbert", "mri_resnet", "eeg_resnet", "mri_medsiglip", "eeg_medsiglip"]:
            if name in classifier_results and task in classifier_results[name]:
                r = classifier_results[name][task]
                acc = r.get("mean_accuracy", 0)
                std = r.get("std_accuracy", 0)
                n = r.get("n_samples", 0)
                row[f"disc_{name}"] = f"{acc:.1%}±{std:.1%}"
                row[f"disc_{name}_n"] = n
            else:
                row[f"disc_{name}"] = "-"
                row[f"disc_{name}_n"] = 0

        # Ensemble
        if "ensemble" in classifier_results and task in classifier_results["ensemble"]:
            e = classifier_results["ensemble"][task]
            wa = e.get("weighted_average_accuracy", 0)
            st = e.get("stacking_accuracy", 0)
            row["disc_ensemble_wa"] = f"{wa:.1%}"
            row["disc_ensemble_stack"] = f"{st:.1%}"
        else:
            row["disc_ensemble_wa"] = "-"
            row["disc_ensemble_stack"] = "-"

        # Hybrid
        if hybrid_results and task in hybrid_results.get("metrics", {}):
            m = hybrid_results["metrics"][task]
            row["hybrid"] = f"{m['accuracy']:.1%}"
            row["hybrid_n"] = m["n_samples"]
        else:
            row["hybrid"] = "-"
            row["hybrid_n"] = 0

        # Multi-turn
        if multiturn_results and task in multiturn_results.get("metrics", {}):
            m = multiturn_results["metrics"][task]
            row["multiturn"] = f"{m['accuracy']:.1%}"
            row["multiturn_n"] = m["n_samples"]
        else:
            row["multiturn"] = "-"
            row["multiturn_n"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def generate_paper_table(df):
    """Generate LaTeX table for paper."""
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Comprehensive comparison of discriminative, generative, and hybrid approaches for epilepsy classification.}")
    lines.append("\\label{tab:full_comparison}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{l|cccc|cccc|cc|c}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{4}{c|}{\\textbf{Generative (Zero-shot)}} & \\multicolumn{4}{c|}{\\textbf{Discriminative (Fine-tuned)}} & \\multicolumn{2}{c|}{\\textbf{Ensemble}} & \\textbf{Hybrid} \\\\")
    lines.append("\\textbf{Task} & Text & MRI & EEG & Pipeline & TF-IDF & BERT & MRI-R50 & EEG-R50 & WA & Stack & \\textbf{Ours} \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        task = row["task"].replace("_", " ").title()
        vals = [
            row.get("gen_text", "-"),
            row.get("gen_mri", "-"),
            row.get("gen_eeg_image", "-"),
            row.get("gen_pipeline", "-"),
            row.get("disc_tfidf_xgb", "-"),
            row.get("disc_pubmedbert", "-"),
            row.get("disc_mri_resnet", "-"),
            row.get("disc_eeg_resnet", "-"),
            row.get("disc_ensemble_wa", "-"),
            row.get("disc_ensemble_stack", "-"),
            row.get("hybrid", "-"),
        ]
        lines.append(f"{task} & {' & '.join(vals)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Baseline Comparison")
    parser.add_argument("--results_dir", type=str,
                       default="logs/baselines/full_gold")
    parser.add_argument("--classifier_dir", type=str,
                       default="logs/baselines/classifiers_gold")
    parser.add_argument("--hybrid_dir", type=str,
                       default="logs/baselines/hybrid_gold")
    parser.add_argument("--multiturn_dir", type=str, default=None,
                       help="Directory with multi-turn pipeline results")
    args = parser.parse_args()

    # Load all results
    print("Loading results...")
    baseline_results = load_baseline_results(args.results_dir)
    classifier_results = load_classifier_results(args.classifier_dir)
    hybrid_results = load_hybrid_results(args.hybrid_dir)
    multiturn_results = load_multiturn_results(args.multiturn_dir) if args.multiturn_dir else None

    if not baseline_results and not classifier_results:
        print("No results found.")
        return

    # Create comparison table
    df = create_full_comparison(baseline_results, classifier_results, hybrid_results, multiturn_results)

    # Print console table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 120)

    # Generative baselines
    print("\n--- Generative (Zero-shot) ---")
    print(f"{'Task':<25s} | {'Text':>8s} | {'MRI':>8s} | {'EEG':>8s} | {'Pipeline':>8s}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['task']:<25s} | {row.get('gen_text', '-'):>8s} | "
              f"{row.get('gen_mri', '-'):>8s} | {row.get('gen_eeg_image', '-'):>8s} | "
              f"{row.get('gen_pipeline', '-'):>8s}")

    # Discriminative baselines
    print("\n--- Discriminative (Fine-tuned) ---")
    print(f"{'Task':<25s} | {'TF-IDF+XGB':>14s} | {'PubMedBERT':>14s} | {'MRI-R50':>14s} | {'EEG-R50':>14s}")
    print("-" * 90)
    for _, row in df.iterrows():
        print(f"{row['task']:<25s} | {row.get('disc_tfidf_xgb', '-'):>14s} | "
              f"{row.get('disc_pubmedbert', '-'):>14s} | {row.get('disc_mri_resnet', '-'):>14s} | "
              f"{row.get('disc_eeg_resnet', '-'):>14s}")

    # Ensemble + Hybrid
    print("\n--- Ensemble & Hybrid ---")
    print(f"{'Task':<25s} | {'Ensemble WA':>12s} | {'Ensemble Stack':>14s} | {'Hybrid':>8s}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['task']:<25s} | {row.get('disc_ensemble_wa', '-'):>12s} | "
              f"{row.get('disc_ensemble_stack', '-'):>14s} | {row.get('hybrid', '-'):>8s}")

    # Save CSV
    csv_path = os.path.join(args.results_dir, "full_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save LaTeX
    tex_path = os.path.join(args.results_dir, "full_comparison.tex")
    tex_content = generate_paper_table(df)
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"Saved LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
