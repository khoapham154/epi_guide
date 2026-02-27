# EPI-GUIDE: A Clinical Guideline-Grounded Hybrid Agentic Framework for Holistic Epilepsy Management

> **MICCAI 2026 Submission** &mdash; Anonymous repository for double-blind review.

---

## Overview

**EPI-GUIDE** is a hybrid multi-agent framework for comprehensive epilepsy management that integrates heterogeneous patient data (EEG, MRI, clinical text) through modality-specific discriminative and generative models. A central orchestrating agent, grounded in international epilepsy guidelines (ILAE, NICE, AES), evaluates multimodal findings within structured clinical pathways and performs iterative cross-agent coordination for evidence-informed decision-making.

### Key Contributions

- **Hybrid discriminative-generative paradigm**: Modality-specific discriminative models (classifiers) provide structured auxiliary evidence to generative LLM agents, combining the reliability of supervised models with the reasoning capabilities of LLMs.
- **Guideline-grounded orchestrator**: A central orchestrating agent uses Retrieval-Augmented Generation (RAG) with curated epilepsy guidelines to evaluate multimodal outputs within structured clinical pathways.
- **Multi-turn cross-agent coordination**: The orchestrator enables iterative follow-up queries to resolve inter-modality inconsistencies across up to 3 deliberation rounds.
- **Multi-modal, multi-task benchmark**: We introduce and release a curated dataset (MME) spanning 306 patients across 5 epilepsy management tasks, with additional validation on a public HD-EEG dataset.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │          Patient Data (Modalities M)        │
                    │     Clinical Text  |  MRI  |  EEG           │
                    └──────────┬─────────┬───────┬────────────────┘
                               │         │       │
              ┌────────────────┼─────────┼───────┼────────────────┐
              │    Discriminative Models  │  Generative Agents     │
              │  ┌──────────────────┐    │  ┌──────────────────┐  │
              │  │ TF-IDF + XGBoost │    │  │ Text Agent       │  │
              │  │ PubMedBERT       │    │  │ (MedGemma-27B)   │  │
              │  │ ResNet-50        │    │  ├──────────────────┤  │
              │  │ MedSigLIP-448    │    │  │ MRI Agent        │  │
              │  │ REVE (EEG)       │    │  │ (MedGemma-4B)    │  │
              │  └────────┬─────────┘    │  ├──────────────────┤  │
              │           │              │  │ EEG Agent        │  │
              │     Ψ(Ŷ) → Textual      │  │ (MedGemma-4B)    │  │
              │     auxiliary evidence   │  └────────┬─────────┘  │
              └───────────┬──────────────┼───────────┘            │
                          │              │                        │
                          ▼              ▼                        │
              ┌──────────────────────────────────┐                │
              │   Aggregated Multimodal Evidence  │                │
              │   E = ∪{ s̃ᵢ⁽ᵍ⁾, s̃ᵢ⁽ᵈ⁾ }         │                │
              └──────────────┬───────────────────┘                │
                             │                                    │
              ┌──────────────▼───────────────────┐                │
              │  Guideline-Grounded Orchestrator  │                │
              │  (GPT-OSS-120B + ILAE RAG)       │◄─── Multi-turn │
              │                                   │     follow-ups │
              │  Action ∈ {FOLLOW-UP, COMPLETE}   │────────────────┘
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │        Final Predictions          │
              │  Epilepsy Type | Seizure Type     │
              │  EZ Localization | ASM Response   │
              │  Surgical Outcome                 │
              └──────────────────────────────────┘
```

---

## Datasets

### 1. Curated Multi-modal Multi-task Epilepsy (MME) Dataset

Our curated cohort includes **306 epilepsy patients** with MRI available for 94 and EEG for 71 cases. It covers five epilepsy management tasks, each as a 3-class classification problem:

| Task | Classes | n |
|------|---------|---|
| Epilepsy Type | Focal / Generalized / Other | 304 |
| Seizure Type | Focal Onset / Generalized Onset / Unknown-Other | 301 |
| EZ Localization | Temporal / Extratemporal / Multifocal-Hemispheric | 242 |
| ASM Response | Drug-Resistant / Responsive / On Treatment (Unspecified) | 242 |
| Surgical Outcome | Seizure-Free / Improved / No Improvement | 133 |

### 2. Human Intracerebral Stimulation HD-EEG Dataset

Public dataset ([Mikulan et al., 2020](https://doi.org/10.1038/s41597-020-0467-x)) comprising **7 subjects** and **61 stimulation sessions** with paired 256-channel HD-EEG, sEEG recordings, and structural MRI. Two tasks:

| Task | Classes |
|------|---------|
| EZ Localization | Temporal / Frontal / Parieto-Occipital |
| Stimulation Intensity | Low (<=0.3 mA) / High (>=0.5 mA) |

---

## Results

### MME Dataset (Table 1)

| Method | Mod. | Epil. Type | Sz. Type | EZ Loc. | ASM Resp. | Surg. Out. | Overall |
|--------|------|-----------|----------|---------|-----------|------------|---------|
| TF-IDF + XGBoost | T | 78.3 | 70.4 | 81.8 | 76.4 | 68.4 | 75.1 |
| PubMedBERT | T | 83.9 | 65.1 | 83.5 | 75.2 | 80.5 | 77.6 |
| ResNet-50 | M | 71.8 | 57.8 | 79.7 | 78.4 | 56.1 | 68.8 |
| MedSigLIP-448 | M | 82.6 | 61.1 | 74.0 | 79.9 | 79.3 | 75.4 |
| Weighted Ensemble | T+M+E | 81.9 | 69.4 | 84.3 | 78.9 | 79.0 | 78.7 |
| MedGemma-27B | T | 72.4 | 53.2 | 67.8 | 63.2 | 52.6 | 61.8 |
| **EPI-GUIDE (Ours)** | **T+M+E** | **89.5** | **77.5** | **91.0** | **84.8** | **86.2** | **85.8** |

### HD-EEG Dataset (Table 2)

| Method | Mod. | EZ Loc. | Stim. Int. | Overall |
|--------|------|---------|------------|---------|
| GFP + XGBoost | E | 23.8 | 48.4 | 36.1 |
| REVE (fine-tuned) | E | 60.8 | 88.1 | 74.5 |
| Multi-Agent LLM | T+M+E | 49.2 | 96.7 | 73.0 |
| **EPI-GUIDE (Ours)** | **T+M+E** | **64.1** | **98.4** | **81.3** |

### Ablation Study (Table 3)

| Configuration | Disc. | RAG | Mean Acc. |
|---------------|-------|-----|-----------|
| **EPI-GUIDE (Full)** | Yes | Yes | **85.8** |
| w/o RAG | Yes | No | 78.8 |
| w/o Disc. | No | Yes | 62.9 |
| w/o Both | No | No | 60.8 |

---

## Repository Structure

```
epi_guide/
├── configs/
│   └── default.py                 # All configuration (models, data paths, hyperparams)
├── models/                        # Agent and model implementations
│   ├── text_agent.py              # Clinical text agent (MedGemma-27B)
│   ├── mri_agent.py               # MRI agent (MedGemma-4B)
│   ├── eeg_agent.py               # EEG agent (MedGemma-4B, dual-path)
│   ├── orchestrator.py            # Base orchestrating agent
│   ├── hybrid_orchestrator.py     # Hybrid orchestrator with discriminative injection
│   ├── multi_turn_pipeline.py     # Multi-turn follow-up controller
│   ├── rag.py                     # ILAE guideline RAG module
│   ├── report_parser.py           # Free-text to structured label parser
│   ├── reve_adapter.py            # REVE EEG foundation model adapter
│   └── ...
├── baselines/                     # Baseline training and evaluation
│   ├── train_text_classifiers.py  # TF-IDF + XGBoost, PubMedBERT
│   ├── train_mri_classifiers.py   # ResNet-50, MedSigLIP-448 (MRI)
│   ├── train_eeg_classifiers.py   # ResNet-50, MedSigLIP-448 (EEG)
│   ├── train_vlm_classifiers.py   # MedSigLIP-448 (both modalities)
│   ├── train_ensemble.py          # Weighted average + stacking ensemble
│   ├── train_mobile2_baselines.py # HD-EEG classical baselines
│   ├── run_text_baseline.py       # Zero-shot text agent evaluation
│   ├── run_mri_baseline.py        # Zero-shot MRI agent evaluation
│   ├── run_eeg_baseline.py        # Zero-shot EEG agent evaluation
│   ├── evaluate_baselines.py      # Aggregate results, generate tables
│   └── evaluate_mobile2*.py       # HD-EEG evaluation
├── data/                          # Data loading and preprocessing
│   ├── dataset.py                 # MME dataset loader
│   ├── mobile2_bids.py            # HD-EEG BIDS loader
│   └── mobile2_to_meaf.py         # HD-EEG to pipeline format conversion
├── external_data/                 # Datasets (see Data Setup below)
│   ├── MME/                       # MME dataset files
│   ├── HD-EEG.tar.xz             # HD-EEG compressed dataset
│   └── Epilepsy-KB.tar.xz.part-* # RAG knowledge base (split archives)
├── case_study/                    # Qualitative case study analysis
├── logs/paper_results/            # Pre-computed results (Tables 1-3)
│
│── run_agent_pipeline.py          # Sequential generative agent pipeline
│── run_hybrid_pipeline.py         # One-pass hybrid pipeline
│── run_multiturn_pipeline.py      # Multi-turn pipeline (full EPI-GUIDE)
│── run_enhanced_pipeline.py       # Enhanced pipeline with few-shot + meta-ensemble
│── run_mobile2_meaf.py            # HD-EEG full pipeline
│── train_mobile2.py               # HD-EEG baseline training
│── train_mobile2_reve.py          # HD-EEG REVE fine-tuning
│── run_all_baselines.sh           # Master script: runs all experiments
│── smoke_test.sh                  # Quick 1-patient validation
└── verify_setup.py                # Environment verification
```

---

## Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- 8x NVIDIA A100-80GB GPUs (or equivalent; ~640GB total VRAM for full pipeline)
- HuggingFace account with access to gated models

### 1. Environment Installation

```bash
conda create -n epi_guide python=3.10 -y
conda activate epi_guide

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece protobuf
pip install sentence-transformers xgboost scikit-learn pandas numpy
pip install Pillow tqdm
```

### 2. HuggingFace Authentication

Several models require gated access. Request access on HuggingFace for:
- `google/medgemma-27b-text-it`
- `google/medgemma-1.5-4b-it`
- `google/medsiglip-448`
- `meta-llama/Llama-3.3-70B-Instruct`
- `brain-bzh/reve-base`

Then set your token:
```bash
export HF_TOKEN="your_huggingface_token"
```

### 3. Data Setup

All data paths default to `external_data/` relative to the project root. You can modify paths in `configs/default.py`.

**MME Dataset:**
```bash
# Unzip the figure images
cd external_data/MME
unzip gold_dataset_figures.zip
cd ../..
```

**HD-EEG Dataset:**
```bash
# Extract the HD-EEG dataset
cd external_data
tar -xvf HD-EEG.tar.xz
cd ..
```

**Epilepsy Knowledge Base (for RAG):**
```bash
# Reconstruct and extract the knowledge base
cd external_data
cat Epilepsy-KB.tar.xz.part-* > Epilepsy-KB.tar.xz
tar -xvf Epilepsy-KB.tar.xz
cd ..
```

### 4. Verify Setup

```bash
python verify_setup.py
```

---

## Usage

### Quick Smoke Test (1 patient)

Validates all code paths without a full run:
```bash
bash smoke_test.sh
```

### Run Individual Components

**Generative agent baselines** (zero-shot, single modality):
```bash
python baselines/run_text_baseline.py --tier gold --max_patients 5
python baselines/run_mri_baseline.py --tier gold --max_patients 5 --device cuda:0
python baselines/run_eeg_baseline.py --tier gold --max_patients 5 --device cuda:0
```

**Train discriminative classifiers** (5-fold CV):
```bash
python baselines/train_text_classifiers.py --tier gold
python baselines/train_mri_classifiers.py --tier gold --device cuda:0
python baselines/train_eeg_classifiers.py --tier gold --device cuda:0
python baselines/train_vlm_classifiers.py --tier gold --device cuda:0
python baselines/train_ensemble.py --tier gold
```

**Sequential agent pipeline** (generative-only):
```bash
python run_agent_pipeline.py --tier gold
```

**Hybrid one-pass pipeline** (with discriminative injection):
```bash
python run_hybrid_pipeline.py --tier gold
```

**Full EPI-GUIDE** (multi-turn + hybrid, requires 8 GPUs):
```bash
python run_multiturn_pipeline.py --tier gold
```

### Run All Experiments

Reproduces all paper results (Tables 1-3):
```bash
bash run_all_baselines.sh 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M%S).log
```

### HD-EEG External Validation

```bash
# Train classical baselines
python baselines/train_mobile2_baselines.py

# Train REVE foundation model
python train_mobile2_reve.py

# Run full EPI-GUIDE on HD-EEG
python run_mobile2_meaf.py --phase agents
python run_mobile2_meaf.py --phase orchestrator --mode single_pass
```

### Evaluate and Generate Tables

```bash
python baselines/evaluate_baselines.py
```

This generates `full_comparison.csv` and `full_comparison.tex` (LaTeX table for paper).

---

## GPU Memory Layout

The full multi-turn pipeline requires 8x A100-80GB GPUs:

| Component | Model | GPUs | VRAM |
|-----------|-------|------|------|
| Text Agent | MedGemma-27B | 0, 1 | ~54 GB |
| MRI Agent | MedGemma-4B | 2 | ~8 GB |
| EEG Agent | MedGemma-4B | 3 | ~8 GB |
| Orchestrator | Llama-3.3-70B | 4, 5, 6, 7 | ~240 GB |

For setups with fewer GPUs, use `run_agent_pipeline.py` which loads/unloads models sequentially.

---

## Pre-computed Results

Pre-computed results corresponding to Tables 1-3 in the paper are available in:
```
logs/paper_results/
├── table1_benchmark_results.json    # MME benchmark (Table 1)
├── table2_ablation_results.json     # Ablation study (Table 3)
├── table3_mobile2_results.json      # HD-EEG benchmark (Table 2)
└── RESULTS_SUMMARY.md               # Human-readable summary
```

---

## Citation

```bibtex
@inproceedings{anonymous2026epiguide,
  title={A Clinical Guideline-Grounded Hybrid Agentic Framework for Holistic Epilepsy Management},
  author={Anonymous},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
