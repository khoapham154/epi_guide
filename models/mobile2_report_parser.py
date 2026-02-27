"""
Mobile-2 Report Parser: Extracts structured labels from generated reports.

Handles 3 tasks specific to the Mobile-2 dataset:
  - ez_region: Temporal / Frontal / Parieto-Occipital (3-class)
  - stim_intensity: Low / High (2-class)
  - source_localization: x, y, z MNI coordinates (regression, parsed from text)

Follows the same 4-priority parsing cascade as the in-house report_parser.py.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


MOBILE2_LABEL_SCHEMA = {
    "ez_region": ["Temporal", "Frontal", "Parieto-Occipital"],
    "stim_intensity": ["Low", "High"],
}

MOBILE2_ALIASES = {
    "ez_region": {
        # Temporal
        "temporal": "Temporal",
        "temporal lobe": "Temporal",
        "mesial temporal": "Temporal",
        "lateral temporal": "Temporal",
        "left temporal": "Temporal",
        "right temporal": "Temporal",
        "hippocampal": "Temporal",
        "amygdala": "Temporal",
        "inferotemporal": "Temporal",
        # Frontal
        "frontal": "Frontal",
        "frontal lobe": "Frontal",
        "left frontal": "Frontal",
        "right frontal": "Frontal",
        "prefrontal": "Frontal",
        "premotor": "Frontal",
        "motor": "Frontal",
        "supplementary motor": "Frontal",
        "orbitofrontal": "Frontal",
        "central": "Frontal",
        "rolandic": "Frontal",
        # Parieto-Occipital
        "parieto-occipital": "Parieto-Occipital",
        "parietal": "Parieto-Occipital",
        "parietal lobe": "Parieto-Occipital",
        "occipital": "Parieto-Occipital",
        "occipital lobe": "Parieto-Occipital",
        "posterior": "Parieto-Occipital",
        "parieto occipital": "Parieto-Occipital",
        "parietooccipital": "Parieto-Occipital",
        "visual cortex": "Parieto-Occipital",
    },
    "stim_intensity": {
        "low": "Low",
        "low intensity": "Low",
        "weak": "Low",
        "subthreshold": "Low",
        "low current": "Low",
        "high": "High",
        "high intensity": "High",
        "strong": "High",
        "suprathreshold": "High",
        "high current": "High",
    },
}

# Label maps for index conversion
MOBILE2_LABEL_MAPS = {
    "ez_region": {"Temporal": 0, "Frontal": 1, "Parieto-Occipital": 2},
    "stim_intensity": {"Low": 0, "High": 1},
}


def _normalize_mobile2_label(field: str, raw_value: str) -> Optional[str]:
    """Normalize a raw extracted value to canonical label."""
    raw_lower = raw_value.lower().strip()

    # Direct match against canonical labels
    for label in MOBILE2_LABEL_SCHEMA[field]:
        if label.lower() == raw_lower or label.lower() in raw_lower or raw_lower in label.lower():
            return label

    # Alias match (longest first)
    if field in MOBILE2_ALIASES:
        sorted_aliases = sorted(MOBILE2_ALIASES[field].items(), key=lambda x: len(x[0]), reverse=True)
        for alias, canonical in sorted_aliases:
            if alias in raw_lower:
                return canonical

    return None


def _fuzzy_extract_mobile2(field: str, full_text: str) -> Optional[str]:
    """Extract label from full report via fuzzy matching."""
    text_lower = full_text.lower()

    # Check canonical labels
    sorted_labels = sorted(MOBILE2_LABEL_SCHEMA[field], key=len, reverse=True)
    for label in sorted_labels:
        if label.lower() in text_lower:
            return label

    # Check aliases
    if field in MOBILE2_ALIASES:
        sorted_aliases = sorted(MOBILE2_ALIASES[field].items(), key=lambda x: len(x[0]), reverse=True)
        for alias, canonical in sorted_aliases:
            if alias in text_lower:
                return canonical

    return None


def parse_mobile2_diagnosis(report_text: str) -> Dict[str, Optional[str]]:
    """Parse a Mobile-2 diagnosis report into classification labels.

    Priority: JSON block > raw JSON > regex fields > fuzzy matching.
    """
    results = {}
    text = report_text.strip()

    # Strip TimeOmni <think>...</think> wrapper if present
    think_match = re.search(r'<answer>(.*?)(?:</answer>|$)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    # PRIMARY: JSON from markdown code block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            json_results = _extract_from_dict(data)
            if sum(1 for v in json_results.values() if v is not None) >= 1:
                return json_results
        except (json.JSONDecodeError, AttributeError):
            pass

    # SECONDARY: Raw JSON parse
    # Try to find any JSON object in the text
    json_pattern = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_pattern:
        try:
            data = json.loads(json_pattern.group(0))
            if isinstance(data, dict):
                json_results = _extract_from_dict(data)
                if sum(1 for v in json_results.values() if v is not None) >= 1:
                    return json_results
        except (json.JSONDecodeError, ValueError):
            pass

    # TERTIARY: Regex "field": "value" patterns
    regex_results = {}
    for field in MOBILE2_LABEL_SCHEMA:
        pattern = rf'"{field}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            canonical = _normalize_mobile2_label(field, match.group(1))
            if canonical:
                regex_results[field] = canonical
    if regex_results:
        for field in MOBILE2_LABEL_SCHEMA:
            if field not in regex_results:
                regex_results[field] = None
        return regex_results

    # FALLBACK: Pattern matching and fuzzy extraction
    for field in MOBILE2_LABEL_SCHEMA:
        field_pattern = field.upper().replace("_", r"[\s_]")
        pattern = rf'(?:\*\*)?{field_pattern}(?:\*\*)?[:\s]+(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            raw_value = match.group(1).strip().rstrip('.').strip('*').strip()
            canonical = _normalize_mobile2_label(field, raw_value)
            results[field] = canonical
        else:
            results[field] = _fuzzy_extract_mobile2(field, text)

    return results


def _extract_from_dict(data: dict) -> Dict[str, Optional[str]]:
    """Extract labels from a parsed JSON dict."""
    results = {}
    for field in MOBILE2_LABEL_SCHEMA:
        raw = data.get(field, "")
        if raw and str(raw).strip():
            results[field] = _normalize_mobile2_label(field, str(raw))
        else:
            results[field] = None
    return results


def parse_source_location(report_text: str) -> Optional[Tuple[float, float, float]]:
    """Try to extract MNI coordinates from report text.

    Looks for patterns like:
      - "MNI coordinates: [x, y, z]"
      - "source_x": 45.6, "source_y": -17.7, "source_z": 52.8
      - x=45.6mm, y=-17.7mm, z=52.8mm
    """
    text = report_text.strip()

    # Strip TimeOmni wrapper
    think_match = re.search(r'<answer>(.*?)(?:</answer>|$)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    # Pattern 1: JSON fields
    x_match = re.search(r'"source_?x"\s*:\s*([-\d.]+)', text)
    y_match = re.search(r'"source_?y"\s*:\s*([-\d.]+)', text)
    z_match = re.search(r'"source_?z"\s*:\s*([-\d.]+)', text)
    if x_match and y_match and z_match:
        try:
            return (float(x_match.group(1)), float(y_match.group(1)), float(z_match.group(1)))
        except ValueError:
            pass

    # Pattern 2: Bracketed coordinates [x, y, z]
    bracket_match = re.search(r'MNI[^[]*\[([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\]', text, re.IGNORECASE)
    if bracket_match:
        try:
            return (float(bracket_match.group(1)), float(bracket_match.group(2)),
                    float(bracket_match.group(3)))
        except ValueError:
            pass

    # Pattern 3: x=..., y=..., z=...
    xyz_match = re.search(
        r'x\s*[=:]\s*([-\d.]+)\s*(?:mm)?\s*[,;]?\s*'
        r'y\s*[=:]\s*([-\d.]+)\s*(?:mm)?\s*[,;]?\s*'
        r'z\s*[=:]\s*([-\d.]+)',
        text, re.IGNORECASE
    )
    if xyz_match:
        try:
            return (float(xyz_match.group(1)), float(xyz_match.group(2)),
                    float(xyz_match.group(3)))
        except ValueError:
            pass

    # Pattern 4: three consecutive numbers after "coordinates" or "location"
    coord_match = re.search(
        r'(?:coordinates?|location|source)\s*[:\s]+\(?\s*([-\d.]+)\s*[,\s]+'
        r'([-\d.]+)\s*[,\s]+([-\d.]+)',
        text, re.IGNORECASE
    )
    if coord_match:
        try:
            return (float(coord_match.group(1)), float(coord_match.group(2)),
                    float(coord_match.group(3)))
        except ValueError:
            pass

    return None


def parse_mobile2_to_label_indices(
    report_text: str,
    label_maps: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, int]:
    """Parse report and convert to label indices.

    Returns dict mapping task name to predicted label index (-1 if not found).
    """
    if label_maps is None:
        label_maps = MOBILE2_LABEL_MAPS

    parsed = parse_mobile2_diagnosis(report_text)
    indices = {}

    for task, label_map in label_maps.items():
        extracted = parsed.get(task)
        if extracted is None:
            indices[task] = -1
            continue

        if extracted in label_map:
            indices[task] = label_map[extracted]
            continue

        # Case-insensitive match
        label_map_lower = {k.lower(): v for k, v in label_map.items()}
        if extracted.lower() in label_map_lower:
            indices[task] = label_map_lower[extracted.lower()]
            continue

        # Partial match
        matched = False
        for map_key, idx in label_map.items():
            if extracted.lower() in map_key.lower() or map_key.lower() in extracted.lower():
                indices[task] = idx
                matched = True
                break

        if not matched:
            indices[task] = -1

    return indices
