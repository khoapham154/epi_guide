#!/bin/bash
# =============================================================================
# EPI-GUIDE: Run All Baselines + Multi-Turn Pipeline
# =============================================================================
# Runs ALL components:
#   Part 1: Generative agent baselines (text, MRI, EEG, pipeline)
#   Part 2: Discriminative classifiers (TF-IDF, PubMedBERT, ResNet-50, MedSigLIP-448)
#   Part 3: Hybrid pipeline (one-pass, binary probability injection)
#   Part 4: Multi-turn pipeline (multi-GPU, iterative follow-ups)
#   Part 5: Evaluation & comparison tables
#
# All logs go to timestamped directories.
#
# Usage (in tmux):
#   tmux new -s meaf
#   cd /path/to/epi_guide
#   bash run_all_baselines.sh 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M%S).log
# =============================================================================

set -e  # Exit on error

# --- Configuration ---
CONDA_ENV="${CONDA_ENV:-base}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIER="gold"
DEVICE="cuda:0"
CV_FOLDS=5
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Timestamped output directories
GEN_DIR="${PROJECT_DIR}/logs/baselines/full_gold_${TIMESTAMP}"
CLF_DIR="${PROJECT_DIR}/logs/baselines/classifiers_gold_${TIMESTAMP}"
HYB_DIR="${PROJECT_DIR}/logs/baselines/hybrid_gold_${TIMESTAMP}"
MT_DIR="${PROJECT_DIR}/logs/baselines/multiturn_gold_${TIMESTAMP}"

echo "============================================================"
echo "EPI-GUIDE - Full Baseline + Multi-Turn Re-run"
echo "============================================================"
echo "Timestamp:       ${TIMESTAMP}"
echo "Conda env:       ${CONDA_ENV}"
echo "Tier:            ${TIER}"
echo "Device:          ${DEVICE}"
echo "Generative dir:  ${GEN_DIR}"
echo "Classifier dir:  ${CLF_DIR}"
echo "Hybrid dir:      ${HYB_DIR}"
echo "Multi-turn dir:  ${MT_DIR}"
echo "============================================================"
echo ""

# --- Activate conda ---
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

cd ${PROJECT_DIR}

mkdir -p "${GEN_DIR}" "${CLF_DIR}" "${HYB_DIR}" "${MT_DIR}"

# =============================================================================
# PART 1: Generative Agent Baselines (with fixed repetition_penalty + max_tokens)
# =============================================================================
echo ""
echo "============================================================"
echo "PART 1: Generative Agent Baselines"
echo "============================================================"

echo ""
echo "--- 1a. Text Agent Baseline ---"
echo "Started: $(date)"
python baselines/run_text_baseline.py \
    --tier ${TIER} \
    --save_dir "${GEN_DIR}"
# Note: text agent uses device_map=auto (no --device flag)
echo "Finished: $(date)"

echo ""
echo "--- 1b. MRI Agent Baseline ---"
echo "Started: $(date)"
python baselines/run_mri_baseline.py \
    --tier ${TIER} \
    --save_dir "${GEN_DIR}" \
    --device ${DEVICE}
echo "Finished: $(date)"

echo ""
echo "--- 1c. EEG Image Agent Baseline ---"
echo "Started: $(date)"
python baselines/run_eeg_baseline.py \
    --tier ${TIER} \
    --save_dir "${GEN_DIR}" \
    --device ${DEVICE}
echo "Finished: $(date)"

echo ""
echo "--- 1d. Full Agent Pipeline (Text + MRI + EEG + Orchestrator) ---"
echo "Started: $(date)"
python run_agent_pipeline.py \
    --tier ${TIER} \
    --save_dir "${GEN_DIR}" \
    --device ${DEVICE}
echo "Finished: $(date)"

echo ""
echo "PART 1 COMPLETE. Generative results in: ${GEN_DIR}"

# =============================================================================
# PART 2: Discriminative Classifiers (5-fold CV)
# =============================================================================
echo ""
echo "============================================================"
echo "PART 2: Discriminative Classifiers"
echo "============================================================"

echo ""
echo "--- 2a. TF-IDF + XGBoost ---"
echo "Started: $(date)"
python baselines/train_text_classifiers.py \
    --tier ${TIER} \
    --cv ${CV_FOLDS} \
    --model tfidf \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "--- 2b. PubMedBERT ---"
echo "Started: $(date)"
python baselines/train_text_classifiers.py \
    --tier ${TIER} \
    --cv ${CV_FOLDS} \
    --model pubmedbert \
    --device ${DEVICE} \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "--- 2c. MRI ResNet-50 ---"
echo "Started: $(date)"
python baselines/train_mri_classifiers.py \
    --tier ${TIER} \
    --cv ${CV_FOLDS} \
    --device ${DEVICE} \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "--- 2d. EEG ResNet-50 ---"
echo "Started: $(date)"
python baselines/train_eeg_classifiers.py \
    --tier ${TIER} \
    --cv ${CV_FOLDS} \
    --device ${DEVICE} \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "--- 2e. MedSigLIP-448 MRI ---"
echo "Started: $(date)"
python baselines/train_vlm_classifiers.py \
    --tier ${TIER} \
    --cv ${CV_FOLDS} \
    --modality mri \
    --device ${DEVICE} \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "--- 2f. MedSigLIP-448 EEG ---"
echo "Started: $(date)"
python baselines/train_vlm_classifiers.py \
    --tier ${TIER} \
    --cv ${CV_FOLDS} \
    --modality eeg \
    --device ${DEVICE} \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "--- 2g. Ensemble (Weighted Average + Stacking) ---"
echo "Started: $(date)"
python baselines/train_ensemble.py \
    --tier ${TIER} \
    --save_dir "${CLF_DIR}"
echo "Finished: $(date)"

echo ""
echo "PART 2 COMPLETE. Classifier results in: ${CLF_DIR}"

# =============================================================================
# PART 3: Hybrid Pipeline (one-pass, binary probability injection)
# =============================================================================
echo ""
echo "============================================================"
echo "PART 3: Hybrid Pipeline (one-pass)"
echo "============================================================"

echo ""
echo "--- 3a. Hybrid Pipeline (using cached agent reports from Part 1) ---"
echo "Started: $(date)"
python run_hybrid_pipeline.py \
    --tier ${TIER} \
    --classifier_dir "${CLF_DIR}" \
    --agent_reports "${GEN_DIR}/agent_reports.json" \
    --save_dir "${HYB_DIR}" \
    --skip_agents
echo "Finished: $(date)"

echo ""
echo "PART 3 COMPLETE. Hybrid results in: ${HYB_DIR}"

# =============================================================================
# PART 4: Multi-Turn Pipeline (multi-GPU, iterative agent communication)
# =============================================================================
echo ""
echo "============================================================"
echo "PART 4: Multi-Turn Pipeline (8x A100, iterative follow-ups)"
echo "============================================================"

echo ""
echo "--- 4a. Multi-Turn Hybrid (using cached agent reports, max 3 rounds) ---"
echo "Started: $(date)"
python run_multiturn_pipeline.py \
    --tier ${TIER} \
    --classifier_dir "${CLF_DIR}" \
    --agent_reports "${GEN_DIR}/agent_reports.json" \
    --save_dir "${MT_DIR}" \
    --skip_agents
echo "Finished: $(date)"

echo ""
echo "--- 4b. One-pass baseline (same pipeline, no follow-ups, for ablation) ---"
echo "Started: $(date)"
python run_multiturn_pipeline.py \
    --tier ${TIER} \
    --classifier_dir "${CLF_DIR}" \
    --agent_reports "${GEN_DIR}/agent_reports.json" \
    --save_dir "${MT_DIR}_onepass" \
    --skip_agents \
    --no_multiturn
echo "Finished: $(date)"

echo ""
echo "PART 4 COMPLETE. Multi-turn results in: ${MT_DIR}"

# =============================================================================
# PART 5: Comprehensive Evaluation & Comparison Tables
# =============================================================================
echo ""
echo "============================================================"
echo "PART 5: Evaluation & Comparison"
echo "============================================================"

echo ""
echo "--- 5a. Generate comparison tables ---"
echo "Started: $(date)"
python baselines/evaluate_baselines.py \
    --results_dir "${GEN_DIR}" \
    --classifier_dir "${CLF_DIR}" \
    --hybrid_dir "${HYB_DIR}" \
    --multiturn_dir "${MT_DIR}"
echo "Finished: $(date)"

# =============================================================================
# PART 6: Mobile-2 External Validation (REVE + Classical Baselines)
# =============================================================================
echo ""
echo "============================================================"
echo "PART 6: Mobile-2 HD-EEG External Validation"
echo "============================================================"

M2_DIR="${PROJECT_DIR}/logs/baselines/mobile2_${TIMESTAMP}"
mkdir -p "${M2_DIR}" "${M2_DIR}/reve" "${M2_DIR}/labram"

echo ""
echo "--- 6a. Mobile-2 Classical Baselines (GFP, Band Power + XGBoost) ---"
echo "Started: $(date)"
python baselines/train_mobile2_baselines.py \
    --save_dir "${M2_DIR}" \
    --device ${DEVICE}
echo "Finished: $(date)"

echo ""
echo "--- 6b. Mobile-2 LaBraM (existing encoder, source localization) ---"
echo "Started: $(date)"
python train_mobile2.py \
    --log_dir "${M2_DIR}/labram" \
    --checkpoint_dir "${M2_DIR}/labram/ckpt"
echo "Finished: $(date)"

echo ""
echo "--- 6c. Mobile-2 REVE (frozen backbone, all 3 tasks) ---"
echo "Started: $(date)"
python train_mobile2_reve.py \
    --task all \
    --device cuda:6 \
    --save_dir "${M2_DIR}/reve"
echo "Finished: $(date)"

echo ""
echo "--- 6d. Mobile-2 Evaluation & Comparison Tables ---"
echo "Started: $(date)"
python baselines/evaluate_mobile2.py \
    --results_dir "${M2_DIR}"
echo "Finished: $(date)"

echo ""
echo "PART 6 COMPLETE. Mobile-2 results in: ${M2_DIR}"

echo ""
echo "============================================================"
echo "ALL DONE!"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Finished at: $(date)"
echo ""
echo "Results locations:"
echo "  Generative:     ${GEN_DIR}"
echo "  Discriminative: ${CLF_DIR}"
echo "  Hybrid:         ${HYB_DIR}"
echo "  Multi-turn:     ${MT_DIR}"
echo "  Mobile-2:       ${M2_DIR}"
echo ""
echo "Key output files:"
echo "  ${GEN_DIR}/full_comparison.csv"
echo "  ${GEN_DIR}/full_comparison.tex"
echo "  ${M2_DIR}/mobile2_comparison.csv"
echo "  ${M2_DIR}/mobile2_comparison.tex"
echo ""
echo "Compare with original logs in:"
echo "  logs/baselines/full_gold/"
echo "  logs/baselines/classifiers_gold/"
echo "============================================================"
