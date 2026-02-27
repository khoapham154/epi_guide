"""
Drug Information Extraction from Clinical Text.

Mines semiology_text, mri_report_text, and eeg_report_text for:
1. AED (anti-epileptic drug) mentions
2. Dosage information (when available)
3. Treatment outcomes associated with specific drugs

Usage:
    python data/extract_drug_info.py --tier gold
    python data/extract_drug_info.py --tier gold --verbose
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import load_classification_csv

# Comprehensive AED name dictionary with aliases
AED_PATTERNS = {
    # Sodium channel blockers
    "carbamazepine": [r"\bcarbamazepine\b", r"\bCBZ\b", r"\btegretol\b"],
    "oxcarbazepine": [r"\boxcarbazepine\b", r"\bOXC\b", r"\btrileptal\b"],
    "phenytoin": [r"\bphenytoin\b", r"\bPHT\b", r"\bdilantin\b", r"\bepanutin\b"],
    "lacosamide": [r"\blacosamide\b", r"\bLCM\b", r"\bvimpat\b"],
    "lamotrigine": [r"\blamotrigine\b", r"\bLTG\b", r"\blamictal\b"],
    "eslicarbazepine": [r"\beslicarbazepine\b", r"\bESL\b", r"\bzebinix\b"],

    # Broad spectrum
    "valproate": [r"\bvalproate\b", r"\bvalproic\s+acid\b", r"\bVPA\b", r"\bdepakote\b",
                  r"\bdepakene\b", r"\bsodium\s+valproate\b", r"\bdivalproex\b"],
    "levetiracetam": [r"\blevetiracetam\b", r"\bLEV\b", r"\bkeppra\b"],
    "topiramate": [r"\btopiramate\b", r"\bTPM\b", r"\btopamax\b"],
    "zonisamide": [r"\bzonisamide\b", r"\bZNS\b", r"\bzonegran\b"],

    # GABA modulators
    "phenobarbital": [r"\bphenobarbital\b", r"\bphenobarbitone\b", r"\bPB\b", r"\bluminal\b"],
    "clobazam": [r"\bclobazam\b", r"\bCLB\b", r"\bfrisium\b", r"\bonfi\b"],
    "clonazepam": [r"\bclonazepam\b", r"\bCZP\b", r"\bklonopin\b", r"\brivotril\b"],
    "vigabatrin": [r"\bvigabatrin\b", r"\bVGB\b", r"\bsabril\b"],

    # Others
    "ethosuximide": [r"\bethosuximide\b", r"\bESM\b", r"\bzarontin\b"],
    "perampanel": [r"\bperampanel\b", r"\bPER\b", r"\bfycompa\b"],
    "brivaracetam": [r"\bbrivaracetam\b", r"\bBRV\b", r"\bbriviact\b"],
    "gabapentin": [r"\bgabapentin\b", r"\bGBP\b", r"\bneurontin\b"],
    "pregabalin": [r"\bpregabalin\b", r"\bPGB\b", r"\blyrica\b"],
    "rufinamide": [r"\brufinamide\b", r"\bRUF\b", r"\bbanzel\b"],
    "felbamate": [r"\bfelbamate\b", r"\bFBM\b", r"\bfelbatol\b"],
    "stiripentol": [r"\bstiripentol\b", r"\bSTP\b", r"\bdiacomit\b"],
    "cannabidiol": [r"\bcannabidiol\b", r"\bCBD\b", r"\bepidiolex\b"],
    "fenfluramine": [r"\bfenfluramine\b", r"\bfintepla\b"],
    "cenobamate": [r"\bcenobamate\b", r"\bxcopri\b"],

    # Older drugs
    "primidone": [r"\bprimidone\b", r"\bPRM\b", r"\bmysoline\b"],
    "acetazolamide": [r"\bacetazolamide\b", r"\bACZ\b", r"\bdiamox\b"],

    # ACTH / steroids (for infantile spasms)
    "ACTH": [r"\bACTH\b", r"\badrenocorticotropic\b", r"\bacthar\b"],
    "prednisolone": [r"\bprednisolone\b", r"\bprednisone\b"],
}

# Dosage pattern: number + unit (mg, mg/kg, g, mcg, ml)
DOSAGE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(mg(?:/kg)?(?:/day)?|g(?:/day)?|mcg(?:/day)?|ml(?:/day)?|mg/d|mg\s*(?:bid|tid|qid|daily|qd|twice|three))',
    re.IGNORECASE
)

# Extended dosage context: drug name followed by dosage within ~50 chars
def find_drug_dosage_pairs(text, drug_name, patterns):
    """Find dosage information near drug mentions."""
    pairs = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 80)
            context = text[start:end]
            dosage_match = DOSAGE_PATTERN.search(context)
            if dosage_match:
                pairs.append({
                    "drug": drug_name,
                    "dosage": dosage_match.group(0),
                    "value": float(dosage_match.group(1)),
                    "unit": dosage_match.group(2),
                    "context": context.strip(),
                })
    return pairs


def extract_drugs_from_text(text):
    """Extract all drug mentions and dosages from text."""
    if not text or not isinstance(text, str):
        return [], []

    text_lower = text.lower()
    found_drugs = set()
    dosage_pairs = []

    for drug_name, patterns in AED_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_drugs.add(drug_name)
                # Also search for dosage nearby
                pairs = find_drug_dosage_pairs(text, drug_name, [pattern])
                dosage_pairs.extend(pairs)
                break

    return list(found_drugs), dosage_pairs


def main():
    parser = argparse.ArgumentParser(description="Extract Drug Information from Clinical Text")
    parser.add_argument("--tier", type=str, default="gold")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    dataset_dir = "external_data/MME"
    save_dir = args.save_dir or f"logs/drug_analysis"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(dataset_dir, f"classification_{args.tier}.csv")
    df = load_classification_csv(csv_path)
    print(f"Loaded {len(df)} patients from {args.tier} tier")

    # Extract drug info for each patient
    patient_drugs = {}
    all_drug_counts = Counter()
    patients_with_drugs = 0
    patients_with_dosage = 0
    all_dosage_pairs = []

    text_columns = ["semiology_text", "mri_report_text", "eeg_report_text"]

    for _, row in df.iterrows():
        pid = row["patient_id"]
        patient_found_drugs = set()
        patient_dosages = []

        for col in text_columns:
            if pd.notna(row.get(col)):
                drugs, dosages = extract_drugs_from_text(str(row[col]))
                patient_found_drugs.update(drugs)
                patient_dosages.extend(dosages)

        patient_drugs[pid] = {
            "drugs": list(patient_found_drugs),
            "dosages": patient_dosages,
            "aed_response_label": str(row.get("aed_response_label", "")) if pd.notna(row.get("aed_response_label")) else None,
        }

        if patient_found_drugs:
            patients_with_drugs += 1
            for drug in patient_found_drugs:
                all_drug_counts[drug] += 1

        if patient_dosages:
            patients_with_dosage += 1
            all_dosage_pairs.extend(patient_dosages)

    # Print results
    print("\n" + "=" * 60)
    print("DRUG INFORMATION EXTRACTION RESULTS")
    print("=" * 60)
    print(f"\nTotal patients: {len(df)}")
    print(f"Patients with any drug mention: {patients_with_drugs} ({patients_with_drugs/len(df)*100:.1f}%)")
    print(f"Patients with dosage info: {patients_with_dosage} ({patients_with_dosage/len(df)*100:.1f}%)")

    print(f"\nDrug mention counts (top 20):")
    for drug, count in all_drug_counts.most_common(20):
        print(f"  {drug:20s}: {count:4d} patients ({count/len(df)*100:.1f}%)")

    print(f"\nTotal dosage pairs found: {len(all_dosage_pairs)}")
    if all_dosage_pairs:
        print("Example dosage pairs:")
        for pair in all_dosage_pairs[:10]:
            print(f"  {pair['drug']}: {pair['dosage']} (context: ...{pair['context'][:60]}...)")

    # Drug category analysis
    drug_categories = {
        "Sodium Channel Blockers": ["carbamazepine", "oxcarbazepine", "phenytoin", "lacosamide", "lamotrigine", "eslicarbazepine"],
        "Broad Spectrum": ["valproate", "levetiracetam", "topiramate", "zonisamide"],
        "GABA Modulators": ["phenobarbital", "clobazam", "clonazepam", "vigabatrin"],
        "Others": ["ethosuximide", "perampanel", "brivaracetam", "gabapentin", "pregabalin",
                    "rufinamide", "felbamate", "stiripentol", "cannabidiol", "fenfluramine", "cenobamate"],
    }

    print(f"\nDrug category distribution (patients with mentions):")
    category_counts = Counter()
    for pid, info in patient_drugs.items():
        for cat, drugs in drug_categories.items():
            if any(d in info["drugs"] for d in drugs):
                category_counts[cat] += 1

    for cat, count in category_counts.most_common():
        print(f"  {cat:30s}: {count:4d} patients ({count/len(df)*100:.1f}%)")

    # Cross-reference with AED response labels
    print(f"\nDrug mentions vs AED response labels:")
    for response in ["Drug-Resistant", "Responsive", "On Treatment (Unspecified)"]:
        patients = [pid for pid, info in patient_drugs.items()
                    if info["aed_response_label"] == response]
        with_drugs = [pid for pid in patients if patient_drugs[pid]["drugs"]]
        avg_drugs = np.mean([len(patient_drugs[pid]["drugs"]) for pid in patients]) if patients else 0
        print(f"  {response:30s}: {len(patients):4d} patients, "
              f"{len(with_drugs)} with drug mentions ({len(with_drugs)/max(len(patients),1)*100:.1f}%), "
              f"avg {avg_drugs:.1f} drugs/patient")

    # Feasibility assessment
    print("\n" + "=" * 60)
    print("FEASIBILITY ASSESSMENT")
    print("=" * 60)

    if patients_with_drugs >= 50:
        print(f"  Drug NAME prediction: FEASIBLE ({patients_with_drugs} patients with drug mentions)")
        print(f"  Recommended: Group into {len(drug_categories)} categories as classification task")
    else:
        print(f"  Drug NAME prediction: NOT FEASIBLE (only {patients_with_drugs} patients)")

    if patients_with_dosage >= 30:
        print(f"  Drug DOSAGE prediction: POSSIBLE ({patients_with_dosage} patients with dosage info)")
    else:
        print(f"  Drug DOSAGE prediction: NOT FEASIBLE (only {patients_with_dosage} patients)")

    # Save results
    output = {
        "summary": {
            "total_patients": len(df),
            "patients_with_drugs": patients_with_drugs,
            "patients_with_dosage": patients_with_dosage,
            "drug_counts": dict(all_drug_counts.most_common()),
            "category_counts": dict(category_counts),
            "n_dosage_pairs": len(all_dosage_pairs),
        },
        "patient_drugs": patient_drugs,
        "dosage_pairs": all_dosage_pairs,
    }

    with open(os.path.join(save_dir, f"drug_extraction_{args.tier}.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to {save_dir}")


# Need numpy for cross-reference stats
import numpy as np

if __name__ == "__main__":
    main()
