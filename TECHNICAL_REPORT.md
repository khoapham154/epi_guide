# EPI-GUIDE: Technical Report
## Clinical Guideline-Grounded Hybrid Agentic Framework for Holistic Epilepsy Management

**Version:** 2.0
**Status:** Implementation complete, evaluation complete

---

## 1. System Overview

EPI-GUIDE (Multimodal Expert Agent Fusion) is a multi-agent framework for automated presurgical epilepsy evaluation. It addresses **five clinical classification tasks** jointly:

| Task | Classes |
|------|---------|
| `epilepsy_type` | Focal / Generalized / Other |
| `seizure_type` | Focal Onset / Generalized Onset / Unknown or Other |
| `ez_localization` | Temporal / Extratemporal / Multifocal or Hemispheric |
| `aed_response` | Drug-Resistant / Responsive / On Treatment (Unspecified) |
| `surgery_outcome` | Seizure-Free / Improved / No Improvement |

The system is evaluated on a Gold-tier dataset of **306 patients** (94 with MRI images, 71 with EEG images) extracted from PubMed Central epilepsy case reports. Labels follow ILAE 2017/2025 classification criteria.

---

## 2. Architecture

The pipeline has two parallel tracks that are fused by the orchestrator:

```
                    ┌────────────────────────────────────────┐
                    │           PATIENT INPUT                │
                    │  Semiology text | MRI images | EEG images │
                    └──────────┬──────────┬──────────────────┘
                               │          │
          ┌────────────────────┼──────────┼───────────────────────┐
          │                    │          │                       │
          ▼                    ▼          ▼                       │
  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐           │
  │  Text Agent  │  │  MRI Agent  │  │  EEG Agent   │   TRACK A  │
  │ MedGemma-27B │  │ MedGemma-   │  │ MedGemma-    │ Generative │
  │  (GPU 0-1)   │  │  1.5-4B     │  │  1.5-4B      │           │
  │              │  │  (GPU 2)    │  │  (GPU 3)     │           │
  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘           │
         └─────────────────┴─────────────────┘                   │
                           │                                      │
                   Free-text reports                              │
                   (optionally enriched                           │
                    by multi-turn Q&A)                            │
                           │                                      │
          ┌────────────────────────────────────┐                  │
          │         TRACK B: Discriminative    │                  │
          │  PubMedBERT | MedSigLIP-448 (MRI)  │                  │
          │  MedSigLIP-448 (EEG) | Ensemble    │                  │
          │  Out-of-fold softmax probabilities │                  │
          └────────────────┬───────────────────┘                  │
                           │                                      │
                           ▼                                      │
                  ┌──────────────────┐                            │
                  │   Hybrid         │     ◄──── ILAE RAG ────────┘
                  │  Orchestrator    │
                  │  GPT-OSS-120B    │
                  │  (GPUs 4–7)      │
                  └──────────────────┘
                           │
                    Final Classification
                    (JSON + reasoning)
```

### 2.1 Modality Agents

**Text Agent** (`google/medgemma-27b-text-it`, BFloat16, GPUs 0–1)
Processes: seizure semiology narrative, clinical history, MRI text report, EEG text report.
Generates: free-text analysis with embedded JSON classification across all 5 tasks.
System prompt instructs it to reason as a senior epileptologist following ILAE 2017/2025 criteria.
No RAG — all clinical knowledge is embedded in the instruction prompt.

**MRI Agent** (`google/medgemma-1.5-4b-it`, BFloat16, GPU 2)
Processes: brain MRI JPEG subfigures (variable count per patient, up to 16).
Generates: structured neuroradiology report identifying hippocampal sclerosis, FCD, DNET, cavernomas, and other epileptogenic lesions with lateralization and lobe-level localization.
Key generation settings: `repetition_penalty=1.2`, `no_repeat_ngram_size=4` (suppresses repetitive radiological boilerplate).

**EEG Agent** (`google/medgemma-1.5-4b-it`, BFloat16, GPU 3)
Processes: EEG montage images (standard 10-20 system screenshots).
Generates: neurophysiology report characterizing background rhythm, interictal discharges, ictal patterns, and lateralization.
Reuses the same MedGemma-1.5-4B checkpoint as the MRI agent but with an EEG-specific system prompt.

### 2.2 Discriminative Classifiers (Baseline Comparison)

Five-fold stratified cross-validation on the same 306-patient cohort.

| Classifier | Modality | Input | Architecture |
|---|---|---|---|
| TF-IDF + XGBoost | Text | Bag-of-words TF-IDF vectors | XGBoost (100 estimators, scale_pos_weight) |
| PubMedBERT | Text | BERT [CLS] token embedding | Linear head, fine-tuned (5 epochs) |
| ResNet-50 (MRI) | MRI images | ImageNet-pretrained CNN features | FC classification head |
| ResNet-50 (EEG) | EEG images | Same | Same |
| MedSigLIP-448 (MRI) | MRI images | MedSigLIP embeddings (frozen) | Trainable head only |
| MedSigLIP-448 (EEG) | EEG images | Same | Same |
| Weighted Average Ensemble | All | OOF probability averaging | Task-specific weight optimization |
| Stacking Ensemble | All | OOF probabilities as meta-features | Logistic Regression meta-learner |

Out-of-fold (OOF) softmax probabilities from the discriminative models are saved and injected into the orchestrator's prompt at inference. This means the discriminative signal is computed without data leakage — each patient's prediction came from a fold where that patient was held out.

### 2.3 Hybrid Orchestrator

**Model:** `openai/gpt-oss-120b`, BFloat16, GPUs 4–7 via `device_map=auto` with explicit `max_memory` constraints.

The orchestrator receives a structured prompt with four sections in order:

1. **Discriminative model signals** — per-class softmax probabilities from all trained classifiers
2. **Specialist agent reports** — free-text from the Text, MRI, and EEG agents (enriched with follow-up Q&A in multi-turn mode)
3. **ILAE clinical guidelines** — top-5 chunks retrieved by the RAG module
4. **Task instruction** — synthesis directive with binary-format-specific guidance

Output: JSON with predictions for all 5 tasks plus a brief reasoning string.

---

## 3. Key Design Decisions

### 3.1 Binary Probability Injection (vs. Top-K)

**Problem.** Each task has exactly 3 classes. Presenting the discriminative predictions in ranked top-k format (`Focal (0.82), Generalized (0.12), Other (0.06)`) essentially answers the question for the LLM — the highest-ranked class is almost always selected, removing the value of the agent reports.

**Solution.** Present predictions in per-class binary format:
```
Text Classifier (PubMedBERT):
  epilepsy_type:
    P(Focal) = 0.82
    P(Generalized) = 0.12
    P(Other) = 0.06
```

Each class is shown as an independent soft signal. The task instruction then differentiates:
- `P(X) > 0.7` → treat as strong supporting evidence
- Spread probabilities → classifier is uncertain, rely on agent reports

The top-k format is retained under `prediction_format="topk"` for ablation comparison.

**Config:** `OrchestratorConfig.prediction_format = "binary"` (default).

### 3.2 MedSigLIP-448 Vision Encoder

**Motivation.** ResNet-50 (ImageNet-pretrained) provides weak features for medical images. `google/medsiglip-448` is a SigLIP model trained on MIMIC-CXR, skin lesions, histopathology, and PMC-OA — directly in-domain for the epilepsy imaging distribution.

**Architecture.**
- Vision encoder: frozen `full_model.vision_model` (SigLIP ViT, ~900M parameters total)
- Feature dimension: determined via dummy forward pass on `(1, 3, 448, 448)` zero tensor; uses `pooler_output` if available, otherwise `last_hidden_state.mean(dim=1)`
- Classification head (trainable): `Dropout(0.3) → Linear(feat_dim, 256) → ReLU → Dropout(0.2) → Linear(256, n_classes)`

**Patient-level aggregation.** Each patient has a variable number of subfigure images (1–16). Features are extracted per-image, then mean-pooled across all images for that patient before classification.

**Training protocol.**
- Only the classification head is optimized (encoder weights frozen throughout)
- Optimizer: AdamW, `lr=1e-3`, `weight_decay=0.01`
- Loss: cross-entropy with class-frequency inverse weights (handles severe class imbalance)
- Scheduler: cosine annealing over 20 epochs
- Early stopping: patience=5 epochs on validation accuracy
- Preprocessing: `AutoProcessor` (448×448, MedSigLIP normalization — not ImageNet statistics)
- Augmentation: random horizontal flip, light color jitter (encoder already domain-adapted)

MedSigLIP-448 requires gated HuggingFace access (token configured in `configs/default.py`).

### 3.3 Multi-GPU Parallelism

**Hardware:** 8× A100-80GB (640 GB total GPU memory).

**Memory footprint:**
| Component | GPUs | Estimated VRAM |
|---|---|---|
| Text Agent (MedGemma-27B) | 0, 1 | ~54 GB |
| MRI Agent (MedGemma-1.5-4B) | 2 | ~8 GB |
| EEG Agent (MedGemma-1.5-4B) | 3 | ~8 GB |
| Orchestrator (GPT-OSS-120B) | 4, 5, 6, 7 | ~240 GB |
| **Total** | **8** | **~310 GB < 640 GB** |

All models are loaded simultaneously at startup. No loading or unloading occurs between patients.

**Parallel agent inference.** The three modality agents run concurrently using `ThreadPoolExecutor(max_workers=3)`. This is safe because CUDA kernel execution releases the Python GIL — threads on separate GPUs run truly in parallel. `ProcessPoolExecutor` is not used because CUDA tensors cannot be serialized across process boundaries.

**Parallel discriminative training.** `--devices cuda:0,cuda:1,...` spreads task training across GPUs using the same `ThreadPoolExecutor` pattern. With 5 tasks × 5 folds, training jobs are round-robin assigned to available GPUs.

**DataLoader settings.** `num_workers=8`, `pin_memory=True`, `persistent_workers=True` to maximize GPU utilization during training.

**Entry point:** `run_multiturn_pipeline.py` handles multi-GPU loading and coordination. `run_hybrid_pipeline.py` is the one-pass variant (same models, no multi-turn).

### 3.4 Multi-Turn Agent Communication

**Motivation.** The one-pass system cannot resolve ambiguities. If the MRI report says temporal but the EEG report says frontal, the orchestrator cannot ask either agent to re-examine that discordance.

**Flow:**

```
Round 0:  Text Agent ─┐
          MRI Agent  ─┼─ (parallel, ThreadPoolExecutor)
          EEG Agent  ─┘
                      ↓
                  Initial reports (text_report, mri_report, eeg_report)
                      ↓
Round 1:  Orchestrator reviews all reports + discriminative signals
          → generates JSON: {"status": "FOLLOWUP", "questions": [
              {"agent": "mri", "question": "Re-examine lateralization..."},
              {"agent": "eeg", "question": "Are there temporal IEDs?"}
            ]}
          OR → {"status": "SATISFIED"}
                      ↓
          If FOLLOWUP: route each question to the named agent's answer_question()
          Each answer appended to that agent's report:
            "--- Follow-up Clarification ---
             Q: Re-examine lateralization...
             A: [agent response, max 512 tokens]"
                      ↓
Round 2+: Orchestrator receives enriched reports + prior conversation history
          → may ask more questions or declare SATISFIED
                      ↓
Final:    Orchestrator generates integrated diagnosis from enriched reports
          (same generate_hybrid_diagnosis() as one-pass, but with enriched reports)
```

**Termination.** The loop terminates when:
- Orchestrator outputs `{"status": "SATISFIED"}` (dynamic stopping)
- Hard cap of `max_rounds=3` is reached (safety limit)
- Orchestrator produces no parseable JSON (defaults to SATISFIED)

**Follow-up question routing.**
Each agent implements `answer_question(original_report, question, patient_data, max_new_tokens=512)`:
- Text Agent: prepends a focused system prompt referencing the original report and question
- MRI Agent: re-processes the original images with the question as context
- EEG Agent: same, optionally with images if available

**Follow-up prompt for orchestrator** (sent as modified `TASK` section):
> "Identify any gaps, ambiguities, or discordances in the reports that need clarification. Focus on: discordance between modalities, missing lateralization, ambiguous drug response, unclear surgical candidacy, specific patterns not mentioned in the initial report."

**JSON parsing with fallback.** The `_parse_followup_response()` method tries three strategies in order: (1) extract from markdown code block, (2) parse raw JSON, (3) keyword detection. If all fail, defaults to `SATISFIED` to prevent infinite loops.

**Config:** `MultiTurnConfig(enabled=True, max_rounds=3, max_questions_per_round=3, followup_max_tokens=512)`.

---

## 4. RAG Module

The orchestrator uses retrieval-augmented generation (RAG) to ground diagnoses in ILAE clinical guidelines.

**Knowledge base.** 9 hand-curated chunks covering:
- ILAE 2017 epilepsy type and seizure type classification
- Epileptogenic zone localization principles
- MRI findings (hippocampal sclerosis, FCD, DNET, cavernomas, heterotopia)
- EEG interpretation (IEDs, ictal patterns, PDR)
- Presurgical evaluation protocol (Phase I/II)
- Multimodal concordance assessment
- AED response (ILAE 2010 drug-resistance definition)
- Engel surgical outcome classification
- ILAE 2025 updates

**Retrieval.** Query = concatenation of the three agent reports. Encoded with `BAAI/bge-base-en-v1.5` (SentenceTransformer). Top-k=5 chunks selected by cosine similarity. Formatted as structured context with relevance scores.

**Integration.** Retrieved chunks are inserted into the orchestrator prompt between the agent reports and the task instruction. The orchestrator is instructed to prioritize concordant findings and use the guidelines to resolve conflicts.

---

## 5. Evaluation Protocol

### 5.1 Metrics
Per-task accuracy across all patients with available labels. Reported as mean ± std (over folds for discriminative models, single pass for generative and hybrid).

### 5.2 Comparison Table Structure
Five method categories:

1. **Generative (zero-shot):** Text Agent only / MRI Agent only / EEG Agent only / Full pipeline (no discriminative signals)
2. **Discriminative (fine-tuned, 5-fold CV):** TF-IDF+XGBoost / PubMedBERT / ResNet-50 (MRI) / ResNet-50 (EEG) / MedSigLIP-448 (MRI) / MedSigLIP-448 (EEG)
3. **Ensemble:** Weighted Average / Stacking
4. **Hybrid (one-pass):** Orchestrator + binary discriminative signals + agent reports + RAG
5. **Multi-turn (ours):** Same as hybrid + iterative follow-up Q&A

### 5.3 Ablations
| Ablation | Flag |
|---|---|
| Binary vs. top-k prediction format | `prediction_format=topk` |
| One-pass vs. multi-turn | `--no_multiturn` in `run_multiturn_pipeline.py` |
| MedSigLIP-448 vs. ResNet-50 | Compare `medsiglip_*_results.json` vs `*_resnet_results.json` |
| With vs. without RAG | (planned; not yet implemented as a flag) |

---

## 6. Output Files

All outputs are written to timestamped directories under `logs/baselines/`.

| Directory | Contents |
|---|---|
| `full_gold_{timestamp}/` | Text, MRI, EEG, and pipeline baseline results; `agent_reports.json` |
| `classifiers_gold_{timestamp}/` | All discriminative classifier results + ensemble |
| `hybrid_gold_{timestamp}/` | One-pass hybrid results |
| `multiturn_gold_{timestamp}/` | Multi-turn results + conversation logs |
| `multiturn_gold_{timestamp}_onepass/` | One-pass ablation |

Key result files (JSON):
- `text_baseline_results.json`, `mri_baseline_results.json`, `eeg_image_baseline_results.json`, `pipeline_results.json`
- `tfidf_xgboost_results.json`, `pubmedbert_results.json`
- `mri_resnet_results.json`, `eeg_resnet_results.json`
- `medsiglip_mri_results.json`, `medsiglip_eeg_results.json`
- `ensemble_results.json`
- `hybrid_results.json`
- `multiturn_results.json` — includes `conversation_logs` per patient and `num_rounds` statistics

Evaluation script (`baselines/evaluate_baselines.py`) aggregates all results into:
- `full_comparison.csv` — machine-readable table
- `full_comparison.tex` — LaTeX table for paper

---

## 7. Running the System

```bash
# Start tmux session
tmux new -s meaf
cd .

# Full run (all 5 parts)
bash run_all_baselines.sh 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M%S).log
```

**Parts:**
1. Generative agent baselines (Text, MRI, EEG, full pipeline)
2. Discriminative classifiers (TF-IDF, PubMedBERT, ResNet-50 MRI/EEG, MedSigLIP-448 MRI/EEG, Ensemble)
3. Hybrid one-pass pipeline (uses cached agent reports from Part 1)
4. Multi-turn pipeline + one-pass ablation (uses cached agent reports from Part 1)
5. Evaluation and comparison table generation

**Re-running specific parts.** Set `--skip_agents` and point `--agent_reports` to an existing `agent_reports.json` to skip agent inference and re-run only the orchestrator.

---

## 8. File Map

```

├── configs/
│   └── default.py              — All dataclass configs (DataConfig, TextAgentConfig,
│                                  MRIAgentConfig, EEGImageConfig, OrchestratorConfig,
│                                  RAGConfig, PipelineConfig, MultiTurnConfig, Config)
├── models/
│   ├── text_agent.py           — MedGemma-27B text analysis + answer_question()
│   ├── mri_agent.py            — MedGemma-1.5-4B MRI VLM + answer_question()
│   ├── eeg_agent.py            — MedGemma-1.5-4B EEG VLM + answer_question()
│   ├── orchestrator.py         — Base orchestrator (GPT-OSS-120B + RAG)
│   ├── hybrid_orchestrator.py  — Hybrid orchestrator: binary injection + followup questions
│   ├── multi_turn_pipeline.py  — MultiTurnPipeline: manages iterative Q&A loop
│   └── rag.py                  — ILAE knowledge base + BGE retrieval
├── baselines/
│   ├── train_text_classifiers.py   — TF-IDF+XGB, PubMedBERT
│   ├── train_mri_classifiers.py    — ResNet-50 MRI
│   ├── train_eeg_classifiers.py    — ResNet-50 EEG
│   ├── train_vlm_classifiers.py    — MedSigLIP-448 (MRI + EEG)
│   ├── train_ensemble.py           — Weighted average + stacking ensemble
│   ├── run_text_baseline.py        — Text agent zero-shot baseline
│   ├── run_mri_baseline.py         — MRI agent zero-shot baseline
│   ├── run_eeg_baseline.py         — EEG agent zero-shot baseline
│   └── evaluate_baselines.py       — Aggregate all results, generate tables
├── data/
│   └── dataset.py              — Patient data loading (CSV, images, label maps)
├── run_agent_pipeline.py        — Full generative pipeline (all 3 agents + base orchestrator)
├── run_hybrid_pipeline.py       — One-pass hybrid pipeline
├── run_multiturn_pipeline.py    — Multi-GPU + multi-turn hybrid pipeline
└── run_all_baselines.sh         — Master script, all 5 parts
```

---

## 9. Limitations and Known Issues

1. **MedSigLIP gated access.** Model requires HuggingFace token with approved access to `google/medsiglip-448`. Token is configured in `configs/default.py`.

2. **LaTeX table.** `generate_paper_table()` in `evaluate_baselines.py` does not yet include MedSigLIP or multi-turn columns. The CSV includes all columns. LaTeX needs to be updated before submission.

3. **Ensemble weighting.** `train_ensemble.py` currently weights based on individual classifier OOF accuracy. It may benefit from being updated to include MedSigLIP results explicitly.

4. **Multi-turn cost.** Multi-turn adds 1–3 orchestrator inference passes per patient on top of the base hybrid cost. With all models loaded simultaneously on 8 GPUs, this does not require any model reloading, but wall-clock time per patient increases proportionally.

5. **No RAG ablation flag.** Currently no CLI flag to disable RAG for ablation; would require code modification.
