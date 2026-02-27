#!/usr/bin/env python3
"""
Mobile-2 Results Evaluation & Comparison Tables.

Aggregates results from REVE, classical baselines, and LaBraM,
then generates comparison tables (CSV + LaTeX) for the paper.

Usage:
    python baselines/evaluate_mobile2.py --results_dir logs/baselines/mobile2_*/
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


def extract_metric(result: dict, task: str) -> str:
    """Extract the primary metric string from a result dict."""
    agg = result.get("aggregate", {})

    if task == "source_localization":
        return agg.get("mean_error_mm", "N/A")
    else:
        return agg.get("accuracy", "N/A")


def collect_results(results_dir: str) -> dict:
    """Scan results_dir for all result JSON files and organize by method × task."""
    results = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_results.json"):
            continue

        fpath = os.path.join(results_dir, fname)
        try:
            data = load_json(fpath)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        method = data.get("method") or data.get("model", "unknown")
        task = data.get("task", "unknown")

        if method not in results:
            results[method] = {}
        results[method][task] = data

    # Also check subdirectories (e.g., reve/, labram/)
    for subdir in sorted(os.listdir(results_dir)):
        subpath = os.path.join(results_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        for fname in sorted(os.listdir(subpath)):
            if not fname.endswith("_results.json"):
                continue

            fpath = os.path.join(subpath, fname)
            try:
                data = load_json(fpath)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

            # Handle combined results file
            if fname == "reve_all_results.json":
                for task_name, task_data in data.items():
                    method = task_data.get("model", subdir)
                    if method not in results:
                        results[method] = {}
                    results[method][task_name] = task_data
                continue

            method = data.get("method") or data.get("model", subdir)
            task = data.get("task", "unknown")

            if method not in results:
                results[method] = {}
            results[method][task] = data

    return results


def create_comparison_table(results: dict) -> str:
    """Create a formatted comparison table."""
    tasks = ["source_localization", "ez_region", "stim_intensity"]
    task_headers = {
        "source_localization": "Source Loc (mm↓)",
        "ez_region": "EZ Region (Acc↑)",
        "stim_intensity": "Intensity (Acc↑)",
    }

    # Determine column widths
    method_width = max(len(m) for m in results.keys()) if results else 20
    method_width = max(method_width, 20)

    header = f"{'Method':<{method_width}}"
    for task in tasks:
        header += f" | {task_headers[task]:>20}"
    separator = "-" * len(header)

    rows = [header, separator]

    # Sort methods: classical first, then deep learning
    method_order = ["GFP_XGBoost", "BandPower_XGBoost", "labram", "reve"]
    sorted_methods = []
    for m in method_order:
        if m in results:
            sorted_methods.append(m)
    for m in results:
        if m not in sorted_methods:
            sorted_methods.append(m)

    for method in sorted_methods:
        row = f"{method:<{method_width}}"
        for task in tasks:
            if task in results[method]:
                metric = extract_metric(results[method][task], task)
                row += f" | {str(metric):>20}"
            else:
                row += f" | {'—':>20}"
        rows.append(row)

    return "\n".join(rows)


def create_latex_table(results: dict) -> str:
    """Generate LaTeX table for MICCAI paper."""
    tasks = ["source_localization", "ez_region", "stim_intensity"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mobile-2 external validation results. "
        r"Source localization reports mean localization error (mm, $\downarrow$). "
        r"EZ region and intensity classification report accuracy ($\uparrow$). "
        r"All results are 7-fold leave-one-subject-out cross-validation.}",
        r"\label{tab:mobile2}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Source Loc. (mm $\downarrow$) & EZ Region (Acc $\uparrow$) & Intensity (Acc $\uparrow$) \\",
        r"\midrule",
    ]

    method_display = {
        "GFP_XGBoost": "GFP + XGBoost",
        "BandPower_XGBoost": "Band Power + XGBoost",
        "labram": "LaBraM (scratch)",
        "reve": r"\textbf{REVE (frozen)}",
    }

    method_order = ["GFP_XGBoost", "BandPower_XGBoost", "labram", "reve"]
    sorted_methods = [m for m in method_order if m in results]
    for m in results:
        if m not in sorted_methods:
            sorted_methods.append(m)

    for method in sorted_methods:
        display = method_display.get(method, method)
        cells = [display]
        for task in tasks:
            if task in results[method]:
                metric = extract_metric(results[method][task], task)
                cells.append(str(metric))
            else:
                cells.append("—")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Mobile-2 evaluation")
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = args.results_dir

    print("=" * 60)
    print("Mobile-2 Results Evaluation")
    print("=" * 60)
    print(f"Results dir: {results_dir}")

    if not os.path.isdir(results_dir):
        print(f"  Error: {results_dir} does not exist.")
        sys.exit(1)

    results = collect_results(results_dir)

    if not results:
        print("  No results found.")
        sys.exit(1)

    print(f"\n  Found {len(results)} methods: {list(results.keys())}")

    # Print comparison table
    table = create_comparison_table(results)
    print(f"\n{table}\n")

    # Save CSV
    csv_path = os.path.join(results_dir, "mobile2_comparison.csv")
    with open(csv_path, "w") as f:
        f.write(table)
    print(f"Saved: {csv_path}")

    # Save LaTeX
    latex = create_latex_table(results)
    latex_path = os.path.join(results_dir, "mobile2_comparison.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved: {latex_path}")

    print(f"\n{'='*60}")
    print("EVALUATION DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
