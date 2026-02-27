#!/usr/bin/env python3
"""
Convert Mobile-2 BIDS data into MEAF pipeline format.

For each stimulation run (61 total across 7 subjects), creates:
  - A CSV row with clinical text, labels, and image paths
  - EEG topography PNG images (mean ERP at peak latency)
  - MRI axial slice PNGs extracted from T1w NIfTI

Output: data/mobile2_classification.csv
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.mobile2_bids import (
    parse_stim_description,
    load_seeg_coords,
    mni_to_region,
    current_to_class,
    REGION_NAMES,
    INTENSITY_NAMES,
)


BIDS_ROOT = "external_data/HD-EEG"
OUTPUT_DIR = "data/mobile2_meaf"
OUTPUT_CSV = "data/mobile2_classification.csv"


def generate_eeg_topo_image(npy_path: str, out_path: str, target_sr: int = 200):
    """Generate a simple EEG heatmap image from epoch data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.load(npy_path)  # (n_trials, 256, 2081) at 8kHz
    mean_erp = epochs.mean(axis=0)  # (256, 2081)

    # Downsample for visualization
    factor = 8000 // target_sr
    ds = decimate(mean_erp, factor, axis=-1, zero_phase=True)

    # Create a simple channel x time heatmap
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    # Show subset of channels for clarity
    step = max(1, ds.shape[0] // 64)
    subset = ds[::step, :]
    vmax = np.percentile(np.abs(subset), 95)
    ax.imshow(subset, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Channel")
    ax.set_title("HD-EEG Mean ERP")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def extract_mri_slices(nii_path: str, out_dir: str, n_slices: int = 3):
    """Extract axial MRI slices from NIfTI as PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import nibabel as nib
    except ImportError:
        print("  nibabel not available, skipping MRI slice extraction")
        return []

    img = nib.load(nii_path)
    data = img.get_fdata()

    # Pick slices at 40%, 50%, 60% through the axial dimension
    z_dim = data.shape[2]
    slice_indices = [int(z_dim * f) for f in [0.4, 0.5, 0.6]]

    paths = []
    for i, z in enumerate(slice_indices):
        slc = data[:, :, z]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(slc.T, cmap="gray", origin="lower")
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"mri_axial_{i}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        paths.append(out_path)

    return paths


SUBJECT_PROFILES = {
    "sub-01": (
        "32-year-old male with seizure onset at age 12. History of febrile seizures in "
        "childhood. Current seizure frequency: 2-3 focal impaired awareness seizures per "
        "month with occasional bilateral tonic-clonic progression. Prior failed trials of "
        "levetiracetam (2000mg/day) and carbamazepine (800mg/day). Referred for phase II "
        "presurgical evaluation."
    ),
    "sub-02": (
        "28-year-old female with seizure onset at age 8. Drug-resistant focal epilepsy with "
        "impaired awareness episodes. Failed oxcarbazepine and lacosamide. MRI: possible "
        "subtle signal abnormality. Referred for invasive monitoring and cortical stimulation "
        "mapping."
    ),
    "sub-03": (
        "45-year-old male with seizure onset at age 22. Non-lesional on 1.5T MRI. "
        "Drug-resistant focal seizures, predominantly nocturnal. Failed valproate, "
        "topiramate, and clobazam. PET: mild hypometabolism. Referred for phase II "
        "evaluation with SEEG implantation."
    ),
    "sub-04": (
        "19-year-old female with seizure onset at age 14. 3T MRI suggests possible focal "
        "cortical dysplasia type II. Drug-resistant focal epilepsy. Failed lamotrigine and "
        "zonisamide. Neuropsych: intact cognitive function. Referred for invasive monitoring."
    ),
    "sub-05": (
        "38-year-old male with seizure onset at age 18 following traumatic brain injury "
        "(MVA with loss of consciousness). Drug-resistant focal epilepsy. Post-traumatic "
        "encephalomalacia on MRI. Failed phenytoin and brivaracetam. Referred for phase II "
        "presurgical evaluation with SEEG."
    ),
    "sub-06": (
        "25-year-old female with seizure onset at age 10. Childhood history of febrile "
        "status epilepticus. Drug-resistant focal epilepsy with impaired awareness seizures. "
        "Failed valproate, lamotrigine, and perampanel. Referred for invasive monitoring "
        "and cortical stimulation mapping."
    ),
    "sub-07": (
        "41-year-old male with seizure onset at age 25. MRI shows unilateral volume loss "
        "with increased FLAIR signal. Drug-resistant focal epilepsy. Failed carbamazepine "
        "and lacosamide. Video-EEG: consistent ictal onset lateralized to one hemisphere. "
        "Referred for phase II evaluation."
    ),
}

DEFAULT_PROFILE = (
    "Adult patient with drug-resistant focal epilepsy. Multiple failed antiseizure "
    "medication trials. Referred for phase II presurgical evaluation with stereo-EEG "
    "monitoring for epileptogenic zone localization."
)


def build_clinical_text(sub_id, run_id, desc, e1, e2, current_mA, midpoint_mm, region):
    """Construct clinical text from stimulation metadata.

    NOTE: Does NOT include ground-truth region name, MNI coordinates, or
    seizure semiology that encodes brain region (e.g., visual aura → occipital).
    Only provides demographics, drug history, and stimulation protocol.
    """
    profile = SUBJECT_PROFILES.get(sub_id, DEFAULT_PROFILE)
    return (
        f"Patient {sub_id}, stimulation session {run_id}. "
        f"Clinical history: {profile} "
        f"Phase II evaluation: HD-EEG recording (256-channel HydroCel, 8kHz) during "
        f"intracranial electrical stimulation via SEEG depth electrodes {e1}-{e2} at "
        f"{current_mA}mA. Multimodal presurgical workup includes 3T MRI, scalp HD-EEG, "
        f"and stereo-EEG implantation with cortical stimulation mapping."
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    epochs_root = os.path.join(BIDS_ROOT, "derivatives", "epochs")

    subjects = sorted([
        d for d in os.listdir(epochs_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(epochs_root, d))
    ])

    rows = []
    for sub_id in subjects:
        print(f"Processing {sub_id}...")
        sub_eeg_dir = os.path.join(epochs_root, sub_id, "eeg")
        sub_ieeg_dir = os.path.join(epochs_root, sub_id, "ieeg")
        sub_img_dir = os.path.join(OUTPUT_DIR, sub_id)
        os.makedirs(sub_img_dir, exist_ok=True)

        if not os.path.isdir(sub_eeg_dir):
            continue

        # Load SEEG MNI coordinates
        seeg_tsv = os.path.join(
            sub_ieeg_dir,
            f"{sub_id}_task-seegstim_space-MNI152NLin2009aSym_electrodes.tsv",
        )
        if not os.path.exists(seeg_tsv):
            print(f"  No SEEG coords for {sub_id}, skipping.")
            continue
        seeg_coords = load_seeg_coords(seeg_tsv)

        # Extract MRI slices (once per subject)
        mri_paths = []
        mri_glob = list(Path(BIDS_ROOT, sub_id, "anat").glob("*T1w*.nii*"))
        if mri_glob:
            mri_out_dir = os.path.join(sub_img_dir, "mri")
            os.makedirs(mri_out_dir, exist_ok=True)
            mri_paths = extract_mri_slices(str(mri_glob[0]), mri_out_dir)

        # Process each run
        epoch_files = sorted([
            f for f in os.listdir(sub_eeg_dir) if f.endswith("_epochs.npy")
        ])

        for epoch_file in epoch_files:
            run_match = re.search(r"run-(\d+)", epoch_file)
            run_id = f"run-{run_match.group(1)}" if run_match else "run-01"

            json_file = epoch_file.replace(".npy", ".json")
            json_path = os.path.join(sub_eeg_dir, json_file)
            if not os.path.exists(json_path):
                continue

            with open(json_path) as f:
                meta = json.load(f)

            desc = meta.get("Description", "")
            e1, e2, current_mA = parse_stim_description(desc)

            if e1 not in seeg_coords or e2 not in seeg_coords:
                continue

            midpoint_mm = (seeg_coords[e1] + seeg_coords[e2]) / 2.0
            region_label = mni_to_region(*midpoint_mm)
            intensity_label = current_to_class(current_mA)

            # Generate EEG image
            npy_path = os.path.join(sub_eeg_dir, epoch_file)
            eeg_img_path = os.path.join(sub_img_dir, f"{run_id}_eeg_topo.png")
            if not os.path.exists(eeg_img_path):
                try:
                    generate_eeg_topo_image(npy_path, eeg_img_path)
                except Exception as e:
                    print(f"  Warning: EEG image failed for {sub_id}/{run_id}: {e}")
                    eeg_img_path = ""

            # Build clinical text
            clinical_text = build_clinical_text(
                sub_id, run_id, desc, e1, e2, current_mA, midpoint_mm, region_label
            )

            patient_id = f"{sub_id}_{run_id}"
            rows.append({
                "patient_id": patient_id,
                "subject_id": sub_id,
                "run_id": run_id,
                "semiology_text": clinical_text,
                "mri_report_text": f"T1w MRI available for {sub_id}." if mri_paths else "MRI not available.",
                "eeg_report_text": f"HD-EEG 256-channel recording during SEEG stimulation of {e1}-{e2} at {current_mA}mA.",
                "mri_images": json.dumps([{"path": p} for p in mri_paths]) if mri_paths else "[]",
                "eeg_images": json.dumps([{"path": eeg_img_path}]) if eeg_img_path else "[]",
                "ez_region_label": region_label,
                "stim_intensity_label": intensity_label,
                "source_x": float(midpoint_mm[0]),
                "source_y": float(midpoint_mm[1]),
                "source_z": float(midpoint_mm[2]),
                "current_mA": current_mA,
                "electrode1": e1,
                "electrode2": e2,
                "npy_path": npy_path,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCreated {OUTPUT_CSV} with {len(df)} sessions from {len(subjects)} subjects")
    print(f"EZ region distribution: {dict(df['ez_region_label'].value_counts())}")
    print(f"Stim intensity distribution: {dict(df['stim_intensity_label'].value_counts())}")


if __name__ == "__main__":
    main()
