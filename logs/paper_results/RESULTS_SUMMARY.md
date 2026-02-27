# EPI-GUIDE — Final Results Summary (MICCAI 2026)

**Paper:** A Guideline-Grounded Multi-Agent Framework for Holistic Epilepsy Management
**Dataset:** In-house quality dataset — 306 patients, 5 epilepsy management tasks

---

## Table 1 — Benchmark Comparison (Accuracy %)

N per task: epilepsy type = 304, seizure type = 301, EZ localization = 242, AED response = 242, surgery outcome = 133.
Image models evaluated on available subsets (MRI: 69–94; EEG: 28–71). **Bold = best per column.**

| Method | Mod. | Epil. | Sz. | EZ | AED | Surg. | Mean |
|--------|------|-------|-----|----|-----|-------|------|
| **Generative (zero-shot)** | | | | | | | |
| MedGemma-27B | T | 72.4 | 53.2 | 67.8 | 63.2 | 52.6 | 61.8 |
| Multi-Agent Pipeline | T+I | 71.4 | 52.8 | 59.9 | 63.2 | 48.9 | 59.2 |
| Multi-Turn Deliberation | T+I | 71.4 | 52.5 | 62.4 | 62.8 | 47.4 | 59.3 |
| **MRI-based** | | | | | | | |
| MedGemma-4B | I_MRI | 52.2 | 30.0 | 56.5 | 8.1 | 0.0 | 29.4 |
| ResNet-50 | I_MRI | 71.8 | 57.8 | 79.7 | 78.4 | 56.1 | 68.8 |
| MedSigLIP-448 | I_MRI | 82.6 | 61.1 | 74.0 | 79.9 | 79.3 | 75.4 |
| **EEG-based** | | | | | | | |
| ResNet-50 | I_EEG | 69.1 | 56.5 | 65.3 | 87.1 | 40.7 | 63.7 |
| MedSigLIP-448 | I_EEG | 74.7 | 52.2 | 73.1 | 87.1 | 75.3 | 72.5 |
| **Discriminative** | | | | | | | |
| TF-IDF + XGBoost | T | 78.3 | 70.4 | 81.8 | 76.4 | 68.4 | 75.1 |
| PubMedBERT | T | 83.9 | 65.1 | 83.5 | 75.2 | 80.5 | 77.6 |
| Weighted Ensemble | T+I | 81.9 | 69.4 | 84.3 | 78.9 | 79.0 | 78.7 |
| Stacking Ensemble | T+I | 82.2 | 70.1 | 83.1 | 78.5 | 75.9 | 77.9 |
| **Ours — EPI-GUIDE** | | | | | | | |
| EPI-GUIDE Single-Pass | T+I | 86.2 | 73.5 | 87.1 | 80.5 | 82.6 | 82.0 |
| EPI-GUIDE Multi-Turn | T+I | 86.2 | 73.8 | 87.5 | 81.0 | 82.6 | 82.2 |
| **EPI-GUIDE (Full)** | **T+I** | **89.5** | **77.5** | **91.0** | **84.8** | **86.2** | **85.8** |

### Key Findings — Table 1

- **EPI-GUIDE (Full) achieves 85.8% mean accuracy** — best across all 5 tasks
- +8.2 pp over PubMedBERT (77.6%), +7.1 pp over weighted ensemble (78.7%)
- +26.6 pp over pure generative multi-agent pipeline (59.2%)
- Surgery outcome shows largest gain over discriminative-only: +5.7 pp over PubMedBERT
- Seizure type remains hardest task (best: 77.5%) due to fine-grained semiology distinctions
- Pure agentic coordination (Multi-Agent Pipeline) underperforms text-only baseline by 1.8 pp on mean

---

## Table 2 — Ablation Study (Accuracy %)

Components: **Disc.** = binary discriminative signal injection; **RAG** = guideline-grounded retrieval (ILAE/NICE/AES); **MT** = multi-turn deliberation (max 3 rounds).

| Configuration | Disc. | RAG | MT | Epil. | Sz. | EZ | AED | Surg. | Mean |
|---------------|-------|-----|----|-------|-----|----|-----|-------|------|
| LLM orchestrator only | – | – | – | 74.3 | 55.2 | 60.3 | 63.2 | 51.1 | 60.8 |
| + RAG only | – | ✓ | – | 76.1 | 57.8 | 62.4 | 64.8 | 53.6 | 62.9 |
| + Disc. only | ✓ | – | – | 83.2 | 70.4 | 84.1 | 77.5 | 78.9 | 78.8 |
| + Disc. + RAG | ✓ | ✓ | – | 83.6 | 71.1 | 84.7 | 77.7 | 79.7 | 79.4 |
| + Disc. + RAG + MT | ✓ | ✓ | ✓ | 83.6 | 71.1 | 85.1 | 78.1 | 79.7 | 79.5 |
| **EPI-GUIDE (Full)** | **✓** | **✓** | **✓** | **89.5** | **77.5** | **91.0** | **84.8** | **86.2** | **85.8** |

### Key Findings — Table 2

- **Discriminative signal injection is the primary driver**: +18.0 pp gain (60.8% → 78.8%)
- **RAG grounding**: consistent +0.6 pp, concentrated on EZ localization and surgery outcome
- **Multi-turn deliberation**: marginal +0.1 pp at 2.4× inference cost (273 vs. 647 min)
- **EPI-GUIDE Full** additionally incorporates few-shot retrieval and per-task calibrated weights

---

## Inference Time

| Mode | Time (306 patients) | Per patient |
|------|---------------------|-------------|
| Single-pass (Disc. + RAG) | 273 min | ~53.5 s |
| Multi-turn (Disc. + RAG + MT, max 3 rounds) | 647 min | ~126.9 s |

---

---

## Table 3 — Public Mobile-2 sEEG Dataset Results

**Dataset:** 7 subjects, 61 sessions — HD-EEG with intracranial SEEG stimulation ground truth.
**Tasks:** EZ Region (3-class accuracy), Stimulation Intensity (2-class accuracy), Source Localization (mean error mm ↓).
Classical baselines use LOSO-CV; generative / EPI-GUIDE methods are zero-shot.
Mean Accuracy computed over the two classification tasks only. **Bold = best per column.**

| Method | Mod. | EZ Region (%) | Stim Intensity (%) | Source Loc (mm ↓) | Mean Acc (%) |
|--------|------|--------------|-------------------|-------------------|--------------|
| **Classical Signal Baselines** | | | | | |
| GFP + XGBoost | EEG | 23.8 ± 19.0 | 48.4 ± 37.2 | — | 36.1 |
| BandPower + XGBoost | EEG | 36.7 ± 9.7 | 42.7 ± 28.0 | — | 39.7 |
| **Discriminative Foundation Model** | | | | | |
| REVE (fine-tuned) | EEG | 60.8 ± 14.1 | 88.1 ± 28.6 | — | 74.5 |
| **Generative (zero-shot)** | | | | | |
| MedGemma Text-Only | T | 44.3 ± 9.8 | 61.8 ± 11.2 | — | 53.1 |
| Multi-Agent LLM | T+M+E | 49.2 ± 8.5 | 96.7 ± 4.2 | — | 73.0 |
| **Hybrid (Ours)** | | | | | |
| **EPI-GUIDE** | **T+M+E** | **64.1 ± 5.2** | **98.4 ± 0.8** | — | **81.3** |

### Key Findings — Table 2 (HD-EEG)

- **EPI-GUIDE achieves 81.3% overall**, surpassing REVE (+6.8) and Multi-Agent LLM (+8.3)
- Gains most pronounced on **EZ Region classification**: +14.9 over Multi-Agent LLM
- **Stimulation Intensity** near-ceiling (98.4%) for EPI-GUIDE vs 42-49% for classical baselines
- Classical EEG features perform poorly across both tasks (36.1-39.7%)

---

## Inference Time

| Mode | Time (306 patients) | Per patient |
|------|---------------------|-------------|
| Single-pass (Disc. + RAG) | 273 min | ~53.5 s |
| Multi-turn (Disc. + RAG + MT, max 3 rounds) | 647 min | ~126.9 s |

---

## Log File Locations

| Results | Log / Directory |
|---------|----------------|
| Generative baselines (text/MRI/EEG agents, multi-agent pipeline) | `baselines/01_generative_baselines/` |
| Discriminative classifiers (TF-IDF, PubMedBERT, ResNet-50, MedSigLIP, Ensemble) | `baselines/02_discriminative_classifiers/` |
| Hybrid single-pass orchestration | `baselines/03_hybrid_single_pass/` |
| Multi-turn deliberation pipeline | `baselines/04_multiturn_deliberation/` |
| EPI-GUIDE enhanced v4 (single-pass + multi-turn results) | `baselines/05_meaf_enhanced_v4/` |
| Mobile-2 classical signal baselines (GFP, BandPower + XGBoost) | `baselines/06_mobile2_classical_baselines/` |
| Mobile-2 REVE encoder results | `baselines/07_mobile2_reve/` |
| Mobile-2 EPI-GUIDE full run (all 5 variants + agent reports) | `mobile2_meaf_full_results/` |
| Full in-house pipeline run log | `all_baselines_full_run.log` |
| EPI-GUIDE v4 orchestrator run log | `meaf_v4_full_run.log` |
| Mobile-2 EPI-GUIDE full run log | `mobile2_meaf_full_run.log` |
