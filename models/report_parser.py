"""
Report Parser: Extracts structured classification labels from generated reports.

Maps free-text diagnoses to simplified 3-class categories matching
the pre-consolidated label_maps.json schema.
"""

import json
import re
from typing import Dict, Optional


# Simplified label schema matching label_maps.json (3 classes per task)
LABEL_SCHEMA = {
    "epilepsy_type": ["Focal", "Generalized", "Other"],
    "seizure_type": ["Focal Onset", "Generalized Onset", "Unknown/Other"],
    "ez_localization": ["Temporal", "Extratemporal", "Multifocal/Hemispheric"],
    "aed_response": ["Drug-Resistant", "Responsive", "On Treatment (Unspecified)"],
    "surgery_outcome": ["Seizure-Free", "Improved", "No Improvement"],
}

# Aliases mapping common LLM output phrases -> canonical labels
ALIASES = {
    "epilepsy_type": {
        # Focal
        "focal": "Focal",
        "focal epilepsy": "Focal",
        "temporal lobe epilepsy": "Focal",
        "tle": "Focal",
        "mtle": "Focal",
        "mesial temporal lobe epilepsy": "Focal",
        "frontal lobe epilepsy": "Focal",
        "fle": "Focal",
        "insular epilepsy": "Focal",
        "parietal lobe epilepsy": "Focal",
        "occipital lobe epilepsy": "Focal",
        "focal cortical dysplasia": "Focal",
        # Generalized
        "generalized": "Generalized",
        "generalized epilepsy": "Generalized",
        "jme": "Generalized",
        "juvenile myoclonic epilepsy": "Generalized",
        "ige": "Generalized",
        "gge": "Generalized",
        "idiopathic generalized epilepsy": "Generalized",
        "childhood absence epilepsy": "Generalized",
        "genetic generalized epilepsy": "Generalized",
        # Other
        "other": "Other",
        "dee": "Other",
        "epileptic encephalopathy": "Other",
        "developmental and epileptic encephalopathy": "Other",
        "dravet": "Other",
        "dravet syndrome": "Other",
        "lennox-gastaut": "Other",
        "lennox-gastaut syndrome": "Other",
        "west syndrome": "Other",
        "autoimmune epilepsy": "Other",
        "immune epilepsy": "Other",
        "drug-resistant epilepsy": "Other",
        "status epilepticus": "Other",
        "unknown": "Other",
        "combined": "Other",
        "combined generalized and focal": "Other",
        "unclassified": "Other",
    },
    "seizure_type": {
        # Focal Onset
        "focal onset": "Focal Onset",
        "focal": "Focal Onset",
        "focal seizure": "Focal Onset",
        "focal seizures": "Focal Onset",
        "focal aware": "Focal Onset",
        "focal impaired awareness": "Focal Onset",
        "focal with impaired awareness": "Focal Onset",
        "focal aware seizure": "Focal Onset",
        "simple partial": "Focal Onset",
        "complex partial": "Focal Onset",
        "focal to bilateral tonic-clonic": "Focal Onset",
        "fbtc": "Focal Onset",
        "secondarily generalized": "Focal Onset",
        "temporal lobe seizure": "Focal Onset",
        "frontal lobe seizure": "Focal Onset",
        # Generalized Onset
        "generalized onset": "Generalized Onset",
        "generalized": "Generalized Onset",
        "generalized seizure": "Generalized Onset",
        "generalized seizures": "Generalized Onset",
        "generalized tonic-clonic": "Generalized Onset",
        "gtc": "Generalized Onset",
        "gtcs": "Generalized Onset",
        "tonic-clonic": "Generalized Onset",
        "absence": "Generalized Onset",
        "absence seizures": "Generalized Onset",
        "typical absence": "Generalized Onset",
        "myoclonic": "Generalized Onset",
        "myoclonic seizures": "Generalized Onset",
        "myoclonic jerks": "Generalized Onset",
        "epileptic spasms": "Generalized Onset",
        # Unknown/Other
        "unknown": "Unknown/Other",
        "unknown/other": "Unknown/Other",
        "other": "Unknown/Other",
        "mixed": "Unknown/Other",
        "multiple seizure types": "Unknown/Other",
        "unclassified": "Unknown/Other",
        "status epilepticus": "Unknown/Other",
    },
    "ez_localization": {
        # Temporal
        "temporal": "Temporal",
        "temporal lobe": "Temporal",
        "left temporal": "Temporal",
        "right temporal": "Temporal",
        "bilateral temporal": "Temporal",
        "mesial temporal": "Temporal",
        "lateral temporal": "Temporal",
        "hippocampal": "Temporal",
        "amygdala": "Temporal",
        # Extratemporal
        "extratemporal": "Extratemporal",
        "frontal": "Extratemporal",
        "frontal lobe": "Extratemporal",
        "left frontal": "Extratemporal",
        "right frontal": "Extratemporal",
        "parietal": "Extratemporal",
        "parietal lobe": "Extratemporal",
        "occipital": "Extratemporal",
        "occipital lobe": "Extratemporal",
        "insular": "Extratemporal",
        "insula": "Extratemporal",
        "opercular": "Extratemporal",
        "central": "Extratemporal",
        "rolandic": "Extratemporal",
        # Multifocal/Hemispheric
        "multifocal": "Multifocal/Hemispheric",
        "multifocal/hemispheric": "Multifocal/Hemispheric",
        "hemispheric": "Multifocal/Hemispheric",
        "bilateral": "Multifocal/Hemispheric",
        "diffuse": "Multifocal/Hemispheric",
        "multilobar": "Multifocal/Hemispheric",
        "widespread": "Multifocal/Hemispheric",
        "generalized": "Multifocal/Hemispheric",
    },
    "aed_response": {
        # Drug-Resistant
        "drug-resistant": "Drug-Resistant",
        "drug resistant": "Drug-Resistant",
        "refractory": "Drug-Resistant",
        "intractable": "Drug-Resistant",
        "pharmacoresistant": "Drug-Resistant",
        "medically refractory": "Drug-Resistant",
        "medically intractable": "Drug-Resistant",
        # Responsive
        "responsive": "Responsive",
        "seizure-free": "Responsive",
        "seizure free": "Responsive",
        "well controlled": "Responsive",
        "well-controlled": "Responsive",
        "controlled": "Responsive",
        "pharmacoresponsive": "Responsive",
        # On Treatment
        "on treatment": "On Treatment (Unspecified)",
        "on treatment (unspecified)": "On Treatment (Unspecified)",
        "partial response": "On Treatment (Unspecified)",
        "on medication": "On Treatment (Unspecified)",
        "partially controlled": "On Treatment (Unspecified)",
        "reduced seizures": "On Treatment (Unspecified)",
    },
    "surgery_outcome": {
        # Seizure-Free
        "seizure-free": "Seizure-Free",
        "seizure free": "Seizure-Free",
        "engel i": "Seizure-Free",
        "engel class i": "Seizure-Free",
        "engel 1": "Seizure-Free",
        "engel ia": "Seizure-Free",
        "free of seizures": "Seizure-Free",
        # Improved
        "improved": "Improved",
        "engel ii": "Improved",
        "engel iii": "Improved",
        "engel class ii": "Improved",
        "engel class iii": "Improved",
        "engel 2": "Improved",
        "engel 3": "Improved",
        "worthwhile improvement": "Improved",
        "reduced frequency": "Improved",
        # No Improvement
        "no improvement": "No Improvement",
        "engel iv": "No Improvement",
        "engel class iv": "No Improvement",
        "engel 4": "No Improvement",
        "no change": "No Improvement",
        "poor outcome": "No Improvement",
        "no surgery": "No Improvement",
        "not applicable": "No Improvement",
    },
}


def parse_diagnosis(report_text: str) -> Dict[str, Optional[str]]:
    """
    Parse a structured diagnosis report into classification labels.

    Priority order:
    1. JSON extraction from ```json ... ``` code block
    2. Raw JSON parse of entire text
    3. Regex field extraction (EPILEPSY_TYPE: Focal)
    4. Fuzzy text matching
    """
    results = {}
    text = report_text.strip()

    # PRIMARY: Extract JSON from markdown code block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            json_results = {}
            for field in LABEL_SCHEMA:
                raw = data.get(field, "")
                if raw and str(raw).strip():
                    json_results[field] = _normalize_label(field, str(raw))
                else:
                    json_results[field] = None
            # Trust JSON parse if we got at least 3 valid fields
            if sum(1 for v in json_results.values() if v is not None) >= 3:
                return json_results
        except (json.JSONDecodeError, AttributeError):
            pass

    # SECONDARY: Try raw JSON parse (no markdown wrapper)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            json_results = {}
            for field in LABEL_SCHEMA:
                raw = data.get(field, "")
                if raw and str(raw).strip():
                    json_results[field] = _normalize_label(field, str(raw))
                else:
                    json_results[field] = None
            if sum(1 for v in json_results.values() if v is not None) >= 3:
                return json_results
    except (json.JSONDecodeError, ValueError):
        pass

    # TERTIARY: Extract "field": "value" patterns from raw text (handles partial/broken JSON)
    json_field_results = {}
    for field in LABEL_SCHEMA:
        pattern = rf'"{field}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            canonical = _normalize_label(field, match.group(1))
            if canonical:
                json_field_results[field] = canonical
    if sum(1 for v in json_field_results.values() if v is not None) >= 3:
        # Fill in any missing fields
        for field in LABEL_SCHEMA:
            if field not in json_field_results:
                json_field_results[field] = None
        return json_field_results

    # FALLBACK: Regex field extraction + fuzzy matching (original method)
    for field in LABEL_SCHEMA:
        # Build regex pattern for both underscored and spaced field names
        field_pattern = field.upper().replace("_", r"[\s_]")
        pattern = rf'(?:\*\*)?{field_pattern}(?:\*\*)?[:\s]+(.+?)(?:\n|$)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            raw_value = match.group(1).strip().rstrip('.').strip('*').strip()
            canonical = _normalize_label(field, raw_value)
            results[field] = canonical
        else:
            results[field] = _fuzzy_extract(field, text)

    return results


def _normalize_label(field: str, raw_value: str) -> Optional[str]:
    """Normalize a raw extracted value to canonical label."""
    raw_lower = raw_value.lower().strip()

    # Direct match against canonical labels (case-insensitive)
    for label in LABEL_SCHEMA[field]:
        if label.lower() == raw_lower or label.lower() in raw_lower or raw_lower in label.lower():
            return label

    # Alias match (longest first for specificity)
    if field in ALIASES:
        sorted_aliases = sorted(ALIASES[field].items(), key=lambda x: len(x[0]), reverse=True)
        for alias, canonical in sorted_aliases:
            if alias in raw_lower:
                return canonical

    return None


def _fuzzy_extract(field: str, full_text: str) -> Optional[str]:
    """Try to extract a label from the full report text via fuzzy matching."""
    text_lower = full_text.lower()

    # Check for direct mentions of canonical labels (longest first)
    sorted_labels = sorted(LABEL_SCHEMA[field], key=len, reverse=True)
    for label in sorted_labels:
        if label.lower() in text_lower:
            return label

    # Check aliases (longest first for specificity)
    if field in ALIASES:
        sorted_aliases = sorted(ALIASES[field].items(), key=lambda x: len(x[0]), reverse=True)
        for alias, canonical in sorted_aliases:
            if alias in text_lower:
                return canonical

    return None


def parse_to_label_indices(
    report_text: str,
    label_maps: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    """
    Parse report and convert to label indices matching label_maps.json.

    Args:
        report_text: Generated diagnosis text.
        label_maps: {task_name: {label_string: index}} from label_maps.json.

    Returns:
        Dict mapping task name to predicted label index (-1 if not found).
    """
    parsed = parse_diagnosis(report_text)
    indices = {}

    for task, label_map in label_maps.items():
        extracted = parsed.get(task)
        if extracted is None:
            indices[task] = -1
            continue

        # Try direct match
        if extracted in label_map:
            indices[task] = label_map[extracted]
            continue

        # Try case-insensitive match
        label_map_lower = {k.lower(): v for k, v in label_map.items()}
        if extracted.lower() in label_map_lower:
            indices[task] = label_map_lower[extracted.lower()]
            continue

        # Try partial match
        matched = False
        for map_key, idx in label_map.items():
            if extracted.lower() in map_key.lower() or map_key.lower() in extracted.lower():
                indices[task] = idx
                matched = True
                break

        if not matched:
            indices[task] = -1

    return indices


def format_classification_prompt() -> str:
    """Return the classification output format instruction for system prompts."""
    lines = [
        "\nAt the END of your report, provide a STRUCTURED CLASSIFICATION with these fields:\n"
    ]
    for task, labels in LABEL_SCHEMA.items():
        options = ", ".join(labels)
        lines.append(f"{task.upper()}: <one of: {options}>")
    lines.append("REASONING: <brief clinical reasoning summary>")
    return "\n".join(lines)
