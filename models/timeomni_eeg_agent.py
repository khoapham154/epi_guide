"""
TimeOmni EEG Agent for Mobile-2 Dataset.

Uses anton-hugging/TimeOmni-1-7B (Qwen2.5-7B fine-tuned for time-series reasoning)
to analyze HD-EEG signals and generate structured clinical reports.

The 256-channel HD-EEG is reduced to compact text-serialized features:
  1. GFP time series (std across channels per time point)
  2. Regional average time series (8 spatial regions)
  3. Band power profile per region

These are embedded as comma-separated values in the prompt.
"""

import gc
import numpy as np
import torch
from typing import Optional
from scipy.signal import welch


# 8 scalp regions for 256-channel HydroCel net (approximate grouping by position)
REGION_NAMES_8 = [
    "Frontal", "Central", "Left-Temporal", "Right-Temporal",
    "Left-Parietal", "Right-Parietal", "Occipital", "Vertex",
]


def assign_channels_to_regions(positions: np.ndarray) -> dict:
    """Assign 256 HD-EEG channels to 8 scalp regions using electrode positions.

    Args:
        positions: (C, 3) array of electrode (x, y, z) positions in meters.

    Returns:
        Dict mapping region_index -> list of channel indices.
    """
    C = positions.shape[0]
    # Convert to mm for easier thresholds
    pos_mm = positions * 1000.0 if np.max(np.abs(positions)) < 1.0 else positions

    x, y, z = pos_mm[:, 0], pos_mm[:, 1], pos_mm[:, 2]
    regions = {i: [] for i in range(8)}

    for ch in range(C):
        xi, yi, zi = x[ch], y[ch], z[ch]
        if zi > 80:
            regions[7].append(ch)       # Vertex (top of head)
        elif yi > 30:
            regions[0].append(ch)       # Frontal (anterior)
        elif yi < -60:
            regions[6].append(ch)       # Occipital (posterior)
        elif xi < -40:
            regions[2].append(ch)       # Left-Temporal
        elif xi > 40:
            regions[3].append(ch)       # Right-Temporal
        elif yi < -20:
            if xi < 0:
                regions[4].append(ch)   # Left-Parietal
            else:
                regions[5].append(ch)   # Right-Parietal
        else:
            regions[1].append(ch)       # Central

    # Ensure no empty regions (assign to nearest non-empty)
    for i in range(8):
        if not regions[i]:
            regions[i] = list(range(C))  # fallback: all channels

    return regions


def compute_regional_timeseries(eeg: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Compute mean time series for each of 8 scalp regions.

    Args:
        eeg: (C, T) at target sample rate
        positions: (C, 3) electrode positions

    Returns:
        (8, T) regional average time series
    """
    regions = assign_channels_to_regions(positions)
    T = eeg.shape[1]
    regional = np.zeros((8, T), dtype=np.float32)
    for i in range(8):
        ch_indices = regions[i]
        regional[i] = eeg[ch_indices].mean(axis=0)
    return regional


def compute_band_powers(eeg: np.ndarray, sr: int, positions: np.ndarray) -> np.ndarray:
    """Compute relative band power per region.

    Args:
        eeg: (C, T) at sr Hz
        sr: sample rate
        positions: (C, 3) electrode positions

    Returns:
        (8, 5) relative band powers [delta, theta, alpha, beta, gamma] per region
    """
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, min(45, sr / 2 - 1))]
    regions = assign_channels_to_regions(positions)
    T = eeg.shape[1]
    nperseg = min(T, sr)

    result = np.zeros((8, 5), dtype=np.float32)
    for i in range(8):
        ch_indices = regions[i]
        region_eeg = eeg[ch_indices].mean(axis=0)  # (T,)
        freqs, psd = welch(region_eeg, fs=sr, nperseg=nperseg)
        total = np.trapz(psd, freqs) + 1e-10
        for b, (lo, hi) in enumerate(bands):
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.any():
                result[i, b] = np.trapz(psd[mask], freqs[mask]) / total
    return result


def compute_spatial_center_of_mass(eeg: np.ndarray, positions: np.ndarray) -> dict:
    """Compute spatial center-of-mass of the evoked response at GFP peak.

    Uses amplitude-weighted channel positions to find the spatial centroid
    of the evoked response. This is a strong indicator of source region.

    Args:
        eeg: (C, T) at target sample rate
        positions: (C, 3) electrode positions

    Returns:
        dict with centroid coordinates, peak amplitudes per region, lateralization index
    """
    C, T = eeg.shape
    gfp = eeg.std(axis=0)
    peak_idx = np.argmax(gfp)

    # Amplitude at GFP peak per channel
    amp_at_peak = np.abs(eeg[:, peak_idx])
    amp_total = amp_at_peak.sum() + 1e-10

    # Convert positions to mm if in meters
    pos_mm = positions * 1000.0 if np.max(np.abs(positions)) < 1.0 else positions.copy()

    # Weighted center of mass
    centroid = np.zeros(3, dtype=np.float64)
    for ch in range(C):
        centroid += amp_at_peak[ch] * pos_mm[ch]
    centroid /= amp_total

    # Peak-to-peak amplitude per region
    regions = assign_channels_to_regions(positions)
    region_p2p = {}
    for i, name in enumerate(REGION_NAMES_8):
        ch_indices = regions[i]
        region_ts = eeg[ch_indices].mean(axis=0)
        region_p2p[name] = float(np.ptp(region_ts))

    # Lateralization index: (right - left) / (right + left)
    left_regions = [2, 4]   # Left-Temporal, Left-Parietal
    right_regions = [3, 5]  # Right-Temporal, Right-Parietal
    left_amp = sum(region_p2p[REGION_NAMES_8[i]] for i in left_regions)
    right_amp = sum(region_p2p[REGION_NAMES_8[i]] for i in right_regions)
    lat_index = (right_amp - left_amp) / (right_amp + left_amp + 1e-10)

    # Estimate brain region from centroid
    cx, cy, cz = centroid
    if cy < -50:
        est_region = "Parieto-Occipital"
    elif cy < -30 and cz > 30:
        est_region = "Parieto-Occipital"
    elif cz < 0:
        est_region = "Temporal"
    elif cy < -20 and cz < 25:
        est_region = "Temporal"
    else:
        est_region = "Frontal"

    return {
        "centroid_mm": centroid.tolist(),
        "region_p2p": region_p2p,
        "lateralization_index": float(lat_index),
        "estimated_region": est_region,
        "max_region": max(region_p2p, key=region_p2p.get),
    }


def format_array_compact(arr: np.ndarray, precision: int = 3) -> str:
    """Format a 1D numpy array as compact comma-separated string."""
    return ", ".join(f"{v:.{precision}f}" for v in arr)


TIMEOMNI_SYSTEM_PROMPT = (
    "You are a clinical neurophysiologist analyzing HD-EEG data recorded during "
    "intracranial electrical stimulation (SEEG). You will receive processed EEG "
    "features from a 256-channel HydroCel recording at 200Hz.\n\n"
    "Your analysis should determine:\n"
    "1. The brain region of the primary evoked response (Temporal / Frontal / Parieto-Occipital)\n"
    "2. Whether the stimulation intensity is Low (<=0.3mA) or High (>=0.5mA)\n"
    "3. The approximate MNI coordinates (x, y, z in mm) of the source\n\n"
    "Output Format: <think>Your step-by-step reasoning analyzing the EEG patterns, "
    "spatial distribution, and amplitude features</think> "
    "<answer>Provide a structured JSON report with your findings</answer>"
)


class TimeOmniEEGAgent:
    """EEG Agent using TimeOmni-1-7B for time-series reasoning on HD-EEG data."""

    def __init__(
        self,
        model_name: str = "anton-hugging/TimeOmni-1-7B",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.001,
        repetition_penalty: float = 1.05,
        target_sr: int = 200,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.target_sr = target_sr

        self.model = None
        self.tokenizer = None

    def load_model(self, device: str = "cuda:3"):
        """Load TimeOmni-1-7B model."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading TimeOmni EEG Agent ({self.model_name}) on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self._device = device
        print(f"TimeOmni loaded on {device}")

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            print("TimeOmni EEG Agent unloaded")

    def _build_eeg_prompt(
        self,
        raw_eeg: np.ndarray,
        electrode_positions: np.ndarray,
        current_mA: float,
        electrode_pair: str,
        subject_id: str = "",
    ) -> str:
        """Build text prompt with serialized EEG features.

        Args:
            raw_eeg: (C, T) at target_sr Hz (already downsampled)
            electrode_positions: (C, 3) electrode positions
            current_mA: stimulation current in mA
            electrode_pair: e.g. "K13-K14"
            subject_id: subject identifier
        """
        C, T = raw_eeg.shape
        sr = self.target_sr

        # Convert to microvolts if data appears to be in volts (values < 1.0)
        if np.max(np.abs(raw_eeg)) < 0.1:
            eeg_uv = raw_eeg * 1e6  # V -> µV
            unit = "µV"
        else:
            eeg_uv = raw_eeg
            unit = "µV"

        # 1. GFP time series
        gfp = eeg_uv.std(axis=0)  # (T,)
        gfp_str = format_array_compact(gfp, precision=2)

        # 2. Regional average time series
        regional = compute_regional_timeseries(raw_eeg, electrode_positions)  # (8, T)

        regional_lines = []
        for i, name in enumerate(REGION_NAMES_8):
            ts_str = format_array_compact(regional[i], precision=4)
            regional_lines.append(f"  {name}: [{ts_str}]")
        regional_str = "\n".join(regional_lines)

        # 3. Band power profile
        band_powers = compute_band_powers(raw_eeg, sr, electrode_positions)  # (8, 5)
        band_names = ["delta(0.5-4Hz)", "theta(4-8Hz)", "alpha(8-13Hz)",
                       "beta(13-30Hz)", "gamma(30-45Hz)"]
        bp_lines = []
        for i, name in enumerate(REGION_NAMES_8):
            bp_vals = ", ".join(f"{band_names[b]}={band_powers[i, b]:.3f}"
                                for b in range(5))
            bp_lines.append(f"  {name}: {bp_vals}")
        bp_str = "\n".join(bp_lines)

        # 4. Spatial analysis — center of mass and region estimation
        spatial = compute_spatial_center_of_mass(raw_eeg, electrode_positions)
        centroid = spatial["centroid_mm"]
        est_region = spatial["estimated_region"]
        max_p2p_region = spatial["max_region"]
        lat_idx = spatial["lateralization_index"]
        region_p2p = spatial["region_p2p"]

        # 5. GFP summary
        gfp_peak_idx = np.argmax(gfp)
        gfp_peak_latency_ms = gfp_peak_idx / sr * 1000
        gfp_peak_amp = gfp[gfp_peak_idx]

        # Region peak-to-peak amplitude summary
        p2p_lines = []
        for name, p2p in sorted(region_p2p.items(), key=lambda x: x[1], reverse=True):
            p2p_lines.append(f"  {name}: {p2p:.6f}")
        p2p_str = "\n".join(p2p_lines)

        lat_side = "right" if lat_idx > 0.1 else ("left" if lat_idx < -0.1 else "bilateral/midline")

        prompt = (
            f"HD-EEG Recording Analysis\n"
            f"Patient: {subject_id}\n"
            f"Stimulation: SEEG electrodes {electrode_pair} at {current_mA}mA\n"
            f"Recording: 256 channels, {sr}Hz, {T} samples ({T/sr*1000:.0f}ms epoch)\n\n"
            f"=== SPATIAL ANALYSIS (amplitude-weighted center of mass) ===\n"
            f"Evoked response centroid: x={centroid[0]:.1f}, y={centroid[1]:.1f}, z={centroid[2]:.1f} mm\n"
            f"Estimated source region: {est_region}\n"
            f"Region with strongest evoked response: {max_p2p_region}\n"
            f"Lateralization: {lat_side} (index={lat_idx:.2f})\n\n"
            f"=== REGION PEAK-TO-PEAK AMPLITUDE (ranked) ===\n"
            f"{p2p_str}\n\n"
            f"=== GFP SUMMARY ===\n"
            f"GFP peak at {gfp_peak_latency_ms:.1f}ms, amplitude {gfp_peak_amp:.4f}\n\n"
            f"Regional average time series ({T} points each):\n"
            f"{regional_str}\n\n"
            f"Relative band power per region:\n"
            f"{bp_str}\n\n"
            f"Based on the spatial distribution and temporal pattern of the evoked response, "
            f"determine:\n"
            f"1. Which brain region (Temporal / Frontal / Parieto-Occipital) is the primary source?\n"
            f"   HINT: The spatial center of mass analysis above provides a strong estimate.\n"
            f"2. Is the stimulation intensity Low (<=0.3mA) or High (>=0.5mA)?\n"
            f"   HINT: The stimulation current is {current_mA}mA. Low means <=0.3mA, High means >=0.5mA.\n"
            f"3. Estimate the approximate MNI coordinates (x, y, z in mm) of the source.\n"
            f"   HINT: Use the evoked response centroid ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) as a starting point.\n\n"
            f"Provide your analysis as a structured EEG report."
        )
        return prompt

    @torch.no_grad()
    def generate_report(
        self,
        raw_eeg: np.ndarray,
        electrode_positions: np.ndarray,
        current_mA: float,
        electrode_pair: str,
        subject_id: str = "",
    ) -> str:
        """Generate EEG analysis report using TimeOmni.

        Args:
            raw_eeg: (C, T) at target_sr Hz (already downsampled and run-averaged)
            electrode_positions: (C, 3) electrode positions
            current_mA: stimulation current in mA
            electrode_pair: e.g. "K13-K14"
            subject_id: e.g. "sub-01"

        Returns:
            Structured EEG report text
        """
        assert self.model is not None, "Call load_model() first"

        question = self._build_eeg_prompt(
            raw_eeg, electrode_positions, current_mA, electrode_pair, subject_id
        )

        # Build chat prompt using Qwen/TimeOmni template
        prompt = (
            f"<|im_start|>system\n{TIMEOMNI_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
        )

        # Decode only generated tokens
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        report = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return report

    @torch.no_grad()
    def answer_question(
        self,
        original_report: str,
        question: str,
        raw_eeg: Optional[np.ndarray] = None,
        electrode_positions: Optional[np.ndarray] = None,
        current_mA: float = 1.0,
        electrode_pair: str = "",
        max_new_tokens: int = 1024,
    ) -> str:
        """Answer a follow-up question from the orchestrator."""
        assert self.model is not None, "Call load_model() first"

        followup = (
            f"You previously generated this EEG report:\n---\n{original_report[:1000]}\n---\n\n"
            f"The integrating physician asks:\nQ: {question}\n\n"
            f"Provide a focused answer based on the EEG data."
        )

        prompt = (
            f"<|im_start|>system\n{TIMEOMNI_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{followup}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
        )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
