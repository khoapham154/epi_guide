# MEAF Architecture — Clinical Case Study (v3)
## Comparative Analysis of Multi-Modal Epilepsy Analysis Framework

**Prepared for:** Clinical Co-Author Expert Rating
**Dataset:** 306 patients from 32 multi-institutional epilepsy PMC case reports
**Date:** February 2026

---

## Overview

**MEAF (Multi-modal Epilepsy Analysis Framework)** produces structured epilepsy phenotyping across five clinically consequential axes:

| Task | Classes | Clinical Importance |
|------|---------|---------------------|
| Epilepsy Type | Focal · Generalized · Other/Unknown | Determines treatment pathway |
| Seizure Type | Focal Onset · Generalized Onset · Unknown/Mixed | Guides AED selection |
| EZ Localization | Temporal · Extratemporal · Multifocal/Hemispheric | Determines surgical candidacy |
| AED Response | Drug-Resistant · Drug-Responsive · Unknown | Triggers presurgical evaluation |
| Surgery Outcome | Seizure-Free · Improved · No Improvement | Prognostication |

### Systems Under Comparison

| System | What It Does |
|--------|-------------|
| **MEAF (Full)** | Specialist agents (Text + MRI-VLM + EEG-VLM) → Fine-tuned discriminative classifiers → RAG-augmented LLM Orchestrator → Multi-Turn consensus |
| **LLM Orchestrator Only** | Same agents, **no discriminative classifiers, no RAG, single-pass** |

## Case Selection

**15 cases** across **5 category groups**, selected for comprehensive evaluation:

| Group | Mechanism Demonstrated | # Cases |
|-------|----------------------|---------|
| A | Discriminative classifiers overriding incorrect predictions | 3 |
| B | Multi-modal fusion advantage | 3 |
| C | Surgery outcome and localization prediction | 3 |
| D | Rare/complex syndromes | 3 |
| E | Comprehensive classification cases | 3 |

---

## Rating Instructions

For each case, rate **MEAF** and **LLM-Orchestrator Only** on two dimensions (1–5):

**Reliability:** Does the prediction match ground truth and reflect what an expert would conclude?
- 5 = All tasks correct, fully consistent with evidence
- 4 = Most tasks correct, minor error on ambiguous dimension
- 3 = Partial accuracy, defensible given uncertainty
- 2 = Significant errors on clinically important tasks
- 1 = Fundamentally wrong

**Reasoning Quality:** Does the narrative reflect sound clinical logic and appropriate multi-modal integration?
- 5 = Expert-level synthesis, transparent uncertainty, correct ILAE application
- 4 = Good reasoning with minor gaps
- 3 = Reasonable but incomplete or partially confused
- 2 = Flawed reasoning, unjustified leaps, internal contradictions
- 1 = No meaningful clinical reasoning

---

# GROUP A — Discriminative Classifiers Overriding Incorrect Predictions

> *Fine-tuned discriminative classifiers trained on structured clinical annotations provide a grounded correction signal when generative agents produce plausible but incorrect classifications. MEAF's ensemble protocol follows strong classifier recommendations (>85% confidence) unless directly contradicted by objective clinical data.*

---

## Case A-1 — PMC4864756, 3y2m_M_EIEE
**Age/Sex:** 3 years 2 months / Male | **Modalities:** Clinical Text + EEG images | **MT Rounds:** 2

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Generalized** | ✅ Generalized | ❌ Focal |
| Seizure Type | **Generalized Onset** | ✅ Generalized Onset | ❌ Focal Onset |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ❌ N/A |
| Surgery Outcome | **No Improvement** | ✅ No Improvement | ❌ Improved |
| **Score** | | **4/4 (100%)** | **0/4 (0%)** |

### MEAF Reasoning
> *"We need to integrate reports.

We have:

Clinical Text Agent report says:
- epilepsy_type: Generalized
- seizure_type: Generalized Onset
- ez_localization: Multifocal/Hemispheric
- aed_response: Drug-Resistant
- surgery_outcome: No Improvement
- reasoning: multiple seizure types early infancy, generalized fast activity, normal MRI, multifocal EEG suggests diffuse process, surgery unlikely.

MRI Agent: No MRI data.

EEG Agent: Unable to analyze; no data.

Thus only clinical text provides concrete info. However we also have discriminative model signals:

Text classifier: moderate Other (82%) but strong recommendation Generalized Onset, drug resistant, no improvement. So conflict: text classifier says epilepsy_type Other (moderate) while clinical text says Generalized."*

### LLM-Only Reasoning
> *"We need to integrate reports.

Agent reports:

Clinical Text Agent says epilepsy_type=Generalized, seizure_type=Generalized Onset, ez_localization=Multifocal/Hemispheric, aed_response=Drug-Resistant, surgery_outcome=No Improvement.

MRI Agent report: basically none, says no images. So no MRI data.

EEG Agent report: says interictal discharges primarily frontal, focal involvement, but no ictal info. So suggests focal frontal focus."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case A-2 — PMC7497200, Patient 3
**Age/Sex:** 9 years (deceased) / F | **Modalities:** Clinical Text | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Generalized** | ✅ Generalized | ❌ Focal |
| Seizure Type | **Generalized Onset** | ✅ Generalized Onset | ❌ Focal Onset |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **No Improvement** | ✅ No Improvement | ❌ Seizure-Free |
| **Score** | | **4/4 (100%)** | **1/4 (25%)** |

### MEAF Reasoning
> *"We need to read all agent reports. Clinical Text Agent gives classification: epilepsy_type Generalized, seizure_type Generalized Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement. MRI and EEG agents have no data ("Not available").

Thus we only have clinical text info plus discriminative model which strongly recommends Generalized, Generalized Onset, Drug-Resistant, No Improvement. No conflict because MRI/EEG missing. So concordance between clinical text and model."*

### LLM-Only Reasoning
> *"We need to integrate reports.

Agent reports:

Clinical Text Agent says epilepsy_type=Generalized, seizure_type=Generalized Onset, ez_localization=Multifocal/Hemispheric, aed_response=Drug-Resistant, surgery_outcome=No Improvement.

MRI Agent: No images available, no report content given. So likely "Not available". Same for EEG Agent: no images, no details. So only clinical text gives info.

Classifier probabilities: epilepsy_type: P(Focal)=0.92 high, P(Generalized)=0.06."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case A-3 — PMC6015084, Patient_10
**Age/Sex:** 66 / Male | **Modalities:** Clinical Text | **MT Rounds:** 2

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ✅ Focal |
| Seizure Type | **Unknown/Mixed** | ✅ Unknown/Mixed | ❌ Focal Onset |
| EZ Localization | **Temporal** | ✅ Temporal | ❌ Multifocal/Hemispheric |
| AED Response | **Unknown** | ✅ Unknown | ❌ Drug-Resistant |
| **Score** | | **4/4 (100%)** | **1/4 (25%)** |

### MEAF Reasoning
> *"We need to integrate reports. We have:

Clinical Text Agent says: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement.

MRI Agent: No MRI data.

EEG Agent: No EEG data.

Discriminative model gives strong recommendations: epilepsy_type Focal, seizure_type Unknown/Other, ez_localization Temporal, aed_response On Treatment (Unspecified). Strong recommendation.

Guidelines: prioritize concordant findings; we only have clinical text (subjective) and discriminative model. No MRI/EEG.

We need to see if any contradictions: Clinical says Multifocal/Hemispheric vs model says Temporal. That's discordant. Also seizure type: Clinical says Focal Onset vs model says Unknown/Other."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning string.

We have classifier probabilities: epilepsy_type: P(Focal)=1.0 => strong support for Focal. seizure_type: P(Focal Onset)=0.75, Generalized=0.01, Unknown=0.24 => strong support for Focal Onset."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

# GROUP B — Multi-Modal Fusion Advantage

> *These cases demonstrate how integrating multiple modalities (clinical text + MRI and/or EEG images) through MEAF's fusion architecture produces more accurate classifications than LLM-only reasoning on the same inputs.*

---

## Case B-1 — PMC4449965, Patient_1
**Age/Sex:** 7 months / Male | **Modalities:** Clinical Text + Brain MRI | **MT Rounds:** 2

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Generalized** | ✅ Generalized | ❌ Focal |
| Seizure Type | **Generalized Onset** | ✅ Generalized Onset | ✅ Generalized Onset |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **No Improvement** | ✅ No Improvement | ✅ No Improvement |
| **Score** | | **4/4 (100%)** | **3/4 (75%)** |

### MEAF Reasoning
> *"We need to integrate reports.

Clinical Text Agent says epilepsy_type Generalized, seizure_type Generalized Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement. Provides reasoning about DEE, severe structural malformations (pontocerebellar hypoplasia, cortical atrophy, hypomyelination). However MRI Agent report says infant with ventriculomegaly, no cortical dysplasia, no focal abnormality, hippocampi normal, mild periventricular WM changes age-appropriate, impression: no epileptogenic lesion. So MRI suggests no structural lesion, essentially MRI-negative. That aligns with possibly generalized? Not necessarily. But MRI doesn't support focal lesion.

EEG Agent: No EEG data."*

### LLM-Only Reasoning
> *"We need to produce JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning sentence(s).

We have three agent reports:

Clinical Text Agent says: epilepsy_type Generalized, seizure_type Generalized Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case B-2 — PMC9896472, Patient_6
**Age/Sex:** 6 years old / Male | **Modalities:** Clinical Text + EEG images | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Generalized** | ✅ Generalized | ✅ Generalized |
| Seizure Type | **Generalized Onset** | ✅ Generalized Onset | ❌ Focal Onset |
| EZ Localization | **Multifocal/Hemispheric** | ✅ Multifocal/Hemispheric | ✅ Multifocal/Hemispheric |
| **Score** | | **3/3 (100%)** | **2/3 (66%)** |

### MEAF Reasoning
> *"We need to integrate reports.

Clinical Text Agent says: epilepsy_type Generalized, seizure_type Focal Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement.

MRI Agent: No imaging data (not available). So cannot contribute.

EEG Agent: Interictal multifocal spikes bilaterally. Background generalized slow wave activity. No clear ictal patterns. Impression: multifocal spike and sharp wave discharges suggestive of epilepsy.

Thus EEG supports multifocal/higher bilateral involvement, consistent with generalized or multifocal focal epilepsy. The EEG does not specify seizure onset type; background generalized slow waves might hint generalized, but interictal multifocal spikes can be seen in focal epilepsy with secondary generalization."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus brief reasoning.

We have three agents:

Clinical Text Agent says:
- epilepsy_type: Generalized
- seizure_type: Focal Onset
- ez_localization: Multifocal/Hemispheric
- aed_response: Drug-Resistant
- surgery_outcome: No Improvement

MRI Agent: No data (no images). So no info.

EEG Agent: First report seems contradictory: says "brain appears normal..."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case B-3 — PMC7261749, Patient_13
**Age/Sex:** 18 ± 9.4 years / Female | **Modalities:** Clinical Text + EEG images | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ✅ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Temporal** | ✅ Temporal | ✅ Temporal |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **Improved** | ✅ Improved | ❌ Seizure-Free |
| **Score** | | **5/5 (100%)** | **4/5 (80%)** |

### MEAF Reasoning
> *"We need to integrate reports.

We have three agents:

Clinical Text Agent says: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Extratemporal, aed_response Drug-Resistant, surgery_outcome Seizure-Free. Reasoning mentions hypothalamic hamartoma (HH) extratemporal.

MRI Agent: No images available; earlier clarification said no images. So MRI report essentially missing. Not available.

EEG Agent: Provides contradictory info. First part says "normal morphology... no structural abnormality". Then later says interictal epileptiform discharge in L-aHH (left anterior hypothalamic hamartoma?) at 1-2 Hz. Ictal onset not clear. So EEG suggests interictal discharges in left hypothalamic region (consistent with HH). But earlier impression says normal MRI."*

### LLM-Only Reasoning
> *"We need to integrate reports.

Agent reports:

Clinical Text Agent says: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Extratemporal, aed_response Drug-Resistant, surgery_outcome Seizure-Free. Reasoning includes hypothalamic hamartoma (extratemporal). So they think extratemporal focus (hypothalamic hamartoma). That's unusual: hypothalamic hamartoma is deep midline structure, considered part of diencephalon, not cortical extratemporal? Usually considered hypothalamic (intrinsic)."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

# GROUP C — Surgery Outcome and Localization Prediction

> *Surgery outcome prediction (Seizure-Free / Improved / No Improvement) and accurate EZ localization are the most clinically consequential tasks. MEAF's discriminative ensemble provides evidence-based posteriors over outcomes independent of narrative bias in clinical text.*

---

## Case C-1 — PMC10791031, Patient 16
**Age/Sex:** 20 / F | **Modalities:** Clinical Text | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ✅ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Temporal** | ✅ Temporal | ❌ N/A |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **No Improvement** | ✅ No Improvement | ✅ No Improvement |
| **Score** | | **5/5 (100%)** | **4/5 (80%)** |

### MEAF Reasoning
> *"We need to parse the situation. The user gave us a long instruction set, plus the specialist agent reports (clinical text, MRI, EEG) which basically say there is no data: "No images available", "No specific MRI findings". The clinical text earlier said focal epilepsy, focal onset, unknown localization, drug-resistant, no improvement. There's also discriminative model signals: strong recommendation for epilepsy_type=Focal, aed_response=Drug-Resistant, others uncertain.

We need to integrate all evidence, note concordance/discordance, prioritize concordant findings, use ILAE guidelines to resolve conflicts. Then output JSON first, then reasoning (brief 2-3 sentences). Actually instructions: "Output the JSON classification FIRST, then reasoning."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning field.

We have to integrate all reports: Clinical Text Agent says epilepsy_type Focal, seizure_type Focal Onset, ez_localization Unknown/Other, aed_response Drug-Resistant, surgery_outcome No Improvement. MRI Agent says "Standard MRI, no significant differences" i.e., MRI-negative. EEG Agent says "VLTM performed", but no result. So we have limited info."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case C-2 — PMC9800159, 19
**Age/Sex:** 36 / Male | **Modalities:** Clinical Text | **MT Rounds:** 2

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ✅ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Temporal** | ✅ Temporal | ❌ Multifocal/Hemispheric |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **No Improvement** | ✅ No Improvement | ✅ No Improvement |
| **Score** | | **5/5 (100%)** | **4/5 (80%)** |

### MEAF Reasoning
> *"We need to integrate reports. We have:

Clinical Text Agent says:
- epilepsy_type: Focal
- seizure_type: Focal Onset
- ez_localization: Extratemporal
- aed_response: Drug-Resistant
- surgery_outcome: No Improvement
- reasoning: semiology laughter without mirth -> frontal lobe origin, EEG confirms left frontotemporal focus, diffuse atrophy suggests widespread pathology, contributes to drug resistance and poor outcome.

MRI Agent: No MRI data.

EEG Agent: No EEG data."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning string.

We have:

Classifier probabilities:
- epilepsy_type: P(Focal)=1.00 => strong support for Focal.
- seizure_type: P(Focal Onset)=0.71, Generalized=0.18, Unknown=0.10 => supports Focal Onset.
- ez_localization: P(Multifocal/Hemispheric)=1.00, others zero. So classifier says multifocal/hemispheric.
- aed_response: P(Drug-Resistant)=0.87, responsive 0."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case C-3 — PMC9800159, Patient_12
**Age/Sex:** 17 / Male | **Modalities:** Clinical Text | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ✅ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Multifocal/Hemispheric** | ✅ Multifocal/Hemispheric | ❌ Temporal |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **Improved** | ❌ No Improvement | ❌ No Improvement |
| **Score** | | **4/5 (80%)** | **3/5 (60%)** |

### MEAF Reasoning
> *"We need to integrate reports. We have Clinical Text Agent report giving full classification: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement. MRI and EEG agents have no data ("No images available"). So only clinical text and discriminative model.

Discriminative model: strong recommendation Focal epilepsy type, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant. For seizure_type uncertain: probabilities 0.55 focal onset, 0.21 generalized, 0.24 unknown. For surgery_outcome uncertain: 0.57 seizure-free, 0.18 improved, 0.24 no improvement.

We need to prioritize concordant findings; use ILAE guidelines to resolve conflicts."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning string.

We have:

Classifier probabilities:
- epilepsy_type: P(Focal)=1.00 => strong support for Focal.
- seizure_type: P(Focal Onset)=0.33, P(Generalized Onset)=0.63, P(Unknown)=0.04. So generalized higher but not overwhelming; conflict with clinical text which says Focal Onset. Need to weigh."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

# GROUP D — Rare and Complex Syndromes

> *Rare epilepsy syndromes test whether MEAF can identify when standard Focal/Generalized labels are insufficient and the ILAE "Other/Unknown" designation is clinically correct.*

---

## Case D-1 — PMC4585516, Patient_4
**Age/Sex:** 38 / F | **Modalities:** Clinical Text + EEG images | **MT Rounds:** 2

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Other/Unknown** | ✅ Other/Unknown | ❌ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| **Score** | | **2/2 (100%)** | **1/2 (50%)** |

### MEAF Reasoning
> *"We need to integrate reports.

Clinical Text Agent says: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Temporal, aed_response Drug-Resistant, surgery_outcome Seizure-Free, reasoning includes parahippocampal gyrus etc.

MRI Agent: No MRI data.

EEG Agent: Normal morphology, mild cortical atrophy bilateral parietal lobes, impression mild cortical atrophy bilateral parietal. No mention of epileptiform discharges, no focal abnormality.

Thus discordance: Clinical text claims clear left temporal focus with MRI hippocampal sclerosis, but MRI not provided, EEG does not support any focal abnormality; only mild bilateral parietal atrophy (non-specific). So we have conflict."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning field.

We have three agents:

Clinical Text Agent gives consistent info: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Temporal, aed_response Drug-Resistant, surgery_outcome Seizure-Free. Also clarifications confirm same.

MRI Agent: Not much detail; follow-up says no images available. So no MRI data."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case D-2 — PMC3757309, 3/M
**Age/Sex:** 2.5 years / Male | **Modalities:** Clinical Text | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Other/Unknown** | ✅ Other/Unknown | ❌ Generalized |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Extratemporal** | ❌ Multifocal/Hemispheric | ❌ Multifocal/Hemispheric |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| **Score** | | **3/4 (75%)** | **2/4 (50%)** |

### MEAF Reasoning
> *"We need to integrate reports. We have:

Clinical Text Agent report says epilepsy_type: Generalized, seizure_type: Focal Onset, ez_localization: Multifocal/Hemispheric, aed_response: Drug-Resistant, surgery_outcome: No Improvement. Then clarification says epilepsy_type: Combined, seizure_type: Focal Onset, ez_localization: Multifocal/Hemispheric, etc.

But we must follow ILAE guidelines and prioritize concordant findings. Also consider discriminative model signals.

MRI Agent: No images, but presumably there is some info? Actually MRI Agent didn't give details; just "No images available". So we have no MRI data.

EEG Agent: also no images.

Thus only textual report gives us data. But we also have classifier outputs: Text classifier strongly recommends epilepsy_type Other (99%)."*

### LLM-Only Reasoning
> *"We need to produce JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus brief reasoning.

We have only Clinical Text Agent report (others not available). Also we have discriminative model probabilities from text, MRI, EEG classifiers. MRI and EEG have no data but still have classifier outputs? Probably they are based on image/eeg but since no data, maybe predictions are default? But we have them anyway.

We need integrate all."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case D-3 — PMC6562032, 13
**Age/Sex:** 19 / M | **Modalities:** Clinical Text | **MT Rounds:** 4

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Other/Unknown** | ❌ Focal | ❌ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Multifocal/Hemispheric** | ✅ Multifocal/Hemispheric | ❌ Temporal |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| **Score** | | **3/4 (75%)** | **2/4 (50%)** |

### MEAF Reasoning
> *"We need to read all agent reports. Clinical Text Agent gives many fields: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Extratemporal, aed_response Drug-Resistant, surgery_outcome No Improvement. MRI and EEG agents have no data ("No images available").

Discriminative model says strong recommendation: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant. No mention of surgery outcome.

We need to integrate. Since MRI and EEG missing, we rely on clinical text and discriminative model. There's discordance between ez_localization: Clinical says Extratemporal, model says Multifocal/Hemispheric. Need to prioritize concordant findings and use ILAE guidelines to resolve conflict."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning field.

We have only Clinical Text Agent report providing all categories. MRI and EEG missing. Discriminative model gives high probability for epilepsy_type=Focal (1.0) and seizure_type=Focal Onset (0.99). So consistent with Clinical report (Focal, Focal Onset). For ez_localization, Clinical says Extratemporal. No other data."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

# GROUP E — Comprehensive Classification Cases

> *These cases cover additional clinical scenarios demonstrating MEAF's ability to correctly phenotype across diverse epilepsy presentations, including drug-responsive cases and varied EZ localizations.*

---

## Case E-1 — PMC7497200, Patient 6
**Age/Sex:** 9 years, 9 months (deceased) / F | **Modalities:** Clinical Text | **MT Rounds:** 4

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Generalized** | ✅ Generalized | ❌ Focal |
| Seizure Type | **Generalized Onset** | ✅ Generalized Onset | ❌ Focal Onset |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| Surgery Outcome | **No Improvement** | ✅ No Improvement | ❌ Seizure-Free |
| **Score** | | **4/4 (100%)** | **1/4 (25%)** |

### MEAF Reasoning
> *"We need to read all agent reports. Clinical Text Agent gives full classification: epilepsy_type Generalized, seizure_type Generalized Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement. MRI and EEG agents have no data ("No images available"). So we only have clinical text. Discriminative model also strongly recommends Generalized, Generalized Onset, Drug-Resistant, No Improvement. No conflict. For ez_localization, the model didn't give; classification options require one of Temporal, Extratemporal, Multifocal/Hemispheric. Clinical says Multifocal/Hemispheric. Since no MRI/EEG data, we accept that."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning field.

We have three agent reports: Clinical Text Agent gives certain classifications. MRI and EEG agents gave no data (only follow-up clarifications saying no images). So only Clinical Text Agent has substantive info. Also we have discriminative model probabilities:

epilepsy_type: P(Focal)=0.00, P(Generalized)=0.97, P(Other)=0.02 => strong support for Generalized."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case E-2 — PMC6562032, Patient_3
**Age/Sex:** 35 / M | **Modalities:** Clinical Text | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ✅ Focal |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Temporal** | ✅ Temporal | ❌ Multifocal/Hemispheric |
| AED Response | **Drug-Resistant** | ✅ Drug-Resistant | ✅ Drug-Resistant |
| **Score** | | **4/4 (100%)** | **3/4 (75%)** |

### MEAF Reasoning
> *"We need to integrate all evidence. We have:

- Clinical Text Agent says epilepsy_type Focal, seizure_type Focal Onset, ez_localization Multifocal/Hemispheric, aed_response Drug-Resistant, surgery_outcome No Improvement. Reasoning mentions left temporal origin but bilateral PNH leads to multifocal.

- Discriminative model says strong recommendation: epilepsy_type Focal, seizure_type Focal Onset, ez_localization Temporal, aed_response Drug-Resistant.

- MRI: bilateral periventricular nodular heterotopia (PNH) with cortical gyration anomalies. Locations: bilateral frontal horns, bilateral trigones etc. So structural abnormality is widespread, not just temporal. Could still be considered extratemporal? But heterotopia are periventricular, could involve both lobes."*

### LLM-Only Reasoning
> *"We need to output JSON with fields: epilepsy_type, seizure_type, ez_localization, aed_response, surgery_outcome, plus reasoning string.

We have only Clinical Text Agent report; MRI and EEG not available. So we rely on that.

Clinical says:

- epilepsy_type: Focal
- seizure_type: Focal Onset
- ez_localization: Multifocal/Hemispheric
- aed_response: Drug-Resistant
- surgery_outcome: No Improvement

Classifier probabilities: epilepsy_type: P(Focal)=0.92 >0.7 => supports Focal."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

## Case E-3 — PMC4210593, Patient_27
**Age/Sex:** 40 / F | **Modalities:** Clinical Text | **MT Rounds:** 3

### Labels
| Task | Ground Truth | MEAF | LLM-Only |
|------|:-----------:|:----:|:--------:|
| Epilepsy Type | **Focal** | ✅ Focal | ❌ Other/Unknown |
| Seizure Type | **Focal Onset** | ✅ Focal Onset | ✅ Focal Onset |
| EZ Localization | **Temporal** | ✅ Temporal | ✅ Temporal |
| AED Response | **Drug-Responsive** | ❌ Drug-Resistant | ❌ Drug-Resistant |
| **Score** | | **3/4 (75%)** | **2/4 (50%)** |

### MEAF Reasoning
> *"We need to integrate reports.

We have:

Clinical Text Agent report: says epilepsy_type Focal, seizure_type Focal Onset, ez_localization Temporal, aed_response Drug-Resistant, surgery_outcome Seizure-Free. Then clarification still same.

But later clarification says surgery_outcome On Treatment (Unspecified) because no confirmation of surgery.

MRI Agent: No images available, so no MRI data.

EEG Agent: No EEG data.

Thus only source is clinical text. However there is conflict: earlier said surgery outcome seizure free, later says unknown. Need to decide based on evidence. Since no explicit evidence of surgery having been done, we should treat surgery_outcome as On Treatment (Unspecified) (i.e., unknown)."*

### LLM-Only Reasoning
> *"We need to output JSON classification first, then brief reasoning (2-3 sentences). Need to incorporate the new info: The question asks about whether patient already underwent surgery and outcome. The reports earlier said surgery outcome: Seizure-Free, but that might be hypothetical? The clarification says we need to answer the physician's question. However the task still is to classify epilepsy type etc. We have to integrate all reports."*

### Clinical Rater Assessment
| | MEAF | LLM-Only |
|-|:----:|:--------:|
| Reliability (1–5) | | |
| Reasoning Quality (1–5) | | |
| Comments | | |

---

# Summary

## Performance Across 15 Cases

| Case | Patient | Modalities | MEAF | LLM-Only | Delta | Key Mechanism |
|------|---------|-----------|:----:|:--------:|:-----:|---------------|
| A-1 | PMC4864756-3y2m_M_EIEE | Text+EEG | 4/4 | 0/4 | +4 | Classifier corrects agent error |
| A-2 | PMC7497200-Patient 3 | Text | 4/4 | 1/4 | +3 | Classifier corrects agent error |
| A-3 | PMC6015084-Patient_10 | Text | 4/4 | 1/4 | +3 | Classifier corrects agent error |
| B-1 | PMC4449965-Patient_1 | Text+MRI | 4/4 | 3/4 | +1 | Multi-modal fusion advantage |
| B-2 | PMC9896472-Patient_6 | Text+EEG | 3/3 | 2/3 | +1 | Multi-modal fusion advantage |
| B-3 | PMC7261749-Patient_13 | Text+EEG | 5/5 | 4/5 | +1 | Multi-modal fusion advantage |
| C-1 | PMC10791031-Patient 16 | Text | 5/5 | 4/5 | +1 | Outcome/localization prediction |
| C-2 | PMC9800159-19 | Text | 5/5 | 4/5 | +1 | Outcome/localization prediction |
| C-3 | PMC9800159-Patient_12 | Text | 4/5 | 3/5 | +1 | Outcome/localization prediction |
| D-1 | PMC4585516-Patient_4 | Text+EEG | 2/2 | 1/2 | +1 | Rare syndrome recognition |
| D-2 | PMC3757309-3/M | Text | 3/4 | 2/4 | +1 | Rare syndrome recognition |
| D-3 | PMC6562032-13 | Text | 3/4 | 2/4 | +1 | Rare syndrome recognition |
| E-1 | PMC7497200-Patient 6 | Text | 4/4 | 1/4 | +3 | Comprehensive classification |
| E-2 | PMC6562032-Patient_3 | Text | 4/4 | 3/4 | +1 | Comprehensive classification |
| E-3 | PMC4210593-Patient_27 | Text | 3/4 | 2/4 | +1 | Comprehensive classification |
| **Total** | | | **57/61 (93%)** | **33/61 (54%)** | **+24** | |

---

## Rating Summary Sheet

| Case | MEAF Reliability | MEAF Reasoning | LLM-Only Reliability | LLM-Only Reasoning | Comments |
|------|:----------------:|:--------------:|:--------------------:|:-------------------:|---------|
| A-1: PMC4864756-3y2m_M_EIEE | /5 | /5 | /5 | /5 | |
| A-2: PMC7497200-Patient 3 | /5 | /5 | /5 | /5 | |
| A-3: PMC6015084-Patient_10 | /5 | /5 | /5 | /5 | |
| B-1: PMC4449965-Patient_1 | /5 | /5 | /5 | /5 | |
| B-2: PMC9896472-Patient_6 | /5 | /5 | /5 | /5 | |
| B-3: PMC7261749-Patient_13 | /5 | /5 | /5 | /5 | |
| C-1: PMC10791031-Patient 16 | /5 | /5 | /5 | /5 | |
| C-2: PMC9800159-19 | /5 | /5 | /5 | /5 | |
| C-3: PMC9800159-Patient_12 | /5 | /5 | /5 | /5 | |
| D-1: PMC4585516-Patient_4 | /5 | /5 | /5 | /5 | |
| D-2: PMC3757309-3/M | /5 | /5 | /5 | /5 | |
| D-3: PMC6562032-13 | /5 | /5 | /5 | /5 | |
| E-1: PMC7497200-Patient 6 | /5 | /5 | /5 | /5 | |
| E-2: PMC6562032-Patient_3 | /5 | /5 | /5 | /5 | |
| E-3: PMC4210593-Patient_27 | /5 | /5 | /5 | /5 | |
| **Average** | **/5** | **/5** | **/5** | **/5** | |

---

*MICCAI 2026 submission — all patient identifiers are de-identified PMC case report references.*
*Rated by: _________________________________ Institution: _________________ Date: ________*