"""
Configuration for EPI-GUIDE: Clinical Guideline-Grounded Hybrid Agentic
Framework for Holistic Epilepsy Management.

Models:
  - Text Agent: google/medgemma-27b-text-it (NO RAG)
  - MRI Agent: google/medgemma-1.5-4b-it
  - EEG Agent: google/medgemma-1.5-4b-it
  - Orchestrator: meta-llama/Llama-3.3-70B-Instruct (WITH RAG)
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    quality_dataset_dir: str = "external_data/MME"
    classification_csv: str = "external_data/MME/classification_gold.csv"
    label_maps_json: str = "external_data/MME/label_maps.json"
    mobile2_dir: str = "external_data/HD-EEG"
    seed: int = 42


@dataclass
class TextAgentConfig:
    """MedGemma-27B text-only for clinical text analysis. NO RAG."""
    model_name: str = "google/medgemma-27b-text-it"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    do_sample: bool = False
    repetition_penalty: float = 1.1

    system_prompt: str = (
        "You are a senior epileptologist with 20+ years of experience in ILAE classification "
        "and presurgical epilepsy evaluation. Analyze the patient's clinical information and "
        "provide a comprehensive, evidence-based diagnosis.\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Carefully read ALL provided information: patient demographics, clinical facts, "
        "seizure semiology, MRI reports, and EEG reports.\n"
        "2. Use patient demographics (age, sex, comorbidities) and clinical facts "
        "(medications, lab results, etiology, disease timeline) to contextualize the diagnosis.\n"
        "3. For EACH classification question, consider ALL options before selecting.\n"
        "4. Base reasoning on ILAE 2017/2025 classification criteria.\n"
        "5. IMPORTANT: Output the JSON classification FIRST, then provide reasoning.\n\n"
        "=== CLASSIFICATION OPTIONS ===\n"
        "1. epilepsy_type: Focal | Generalized | Other\n"
        "2. seizure_type: Focal Onset | Generalized Onset | Unknown/Other\n"
        "3. ez_localization: Temporal | Extratemporal | Multifocal/Hemispheric\n"
        "4. aed_response: Drug-Resistant | Responsive | On Treatment (Unspecified)\n"
        "5. surgery_outcome: Seizure-Free | Improved | No Improvement\n\n"
        "Option guidance:\n"
        "- Focal: Seizures from one hemisphere (TLE, FLE). Generalized: Bilateral from onset (JME, CAE). Other: Combined/DEE.\n"
        "- Temporal: Hippocampal/mesial/lateral temporal. Extratemporal: Frontal/parietal/occipital/insular. Multifocal: Multiple foci/hemispheric.\n"
        "- Drug-Resistant: Failed 2+ AED trials. Responsive: Seizure-free on meds. Unspecified: Response unclear.\n"
        "- Seizure-Free: MRI lesion + concordant data (Engel I). Improved: Lesion with discordance (Engel II/III). No Improvement: MRI-negative/multifocal.\n\n"
        "=== OUTPUT FORMAT (JSON FIRST) ===\n"
        "```json\n"
        "{\n"
        '  "epilepsy_type": "YOUR_ANSWER",\n'
        '  "seizure_type": "YOUR_ANSWER",\n'
        '  "ez_localization": "YOUR_ANSWER",\n'
        '  "aed_response": "YOUR_ANSWER",\n'
        '  "surgery_outcome": "YOUR_ANSWER",\n'
        '  "reasoning": "Brief clinical reasoning (2-3 sentences)"\n'
        "}\n"
        "```"
    )


@dataclass
class MRIAgentConfig:
    """MedGemma-1.5-4B multimodal for MRI report generation."""
    model_name: str = "google/medgemma-1.5-4b-it"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = False
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 4

    system_prompt: str = (
        "You are a neuroradiologist specializing in epilepsy surgery evaluation. "
        "Analyze these brain MRI images and provide a structured report.\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Examine images systematically, comparing hemispheres.\n"
        "2. Look for: FCD (cortical thickening, blurred gray-white junction, transmantle sign), "
        "hippocampal sclerosis (volume loss, T2/FLAIR hyperintensity), tumors (DNET, ganglioglioma), "
        "vascular malformations, heterotopia, tuberous sclerosis.\n"
        "3. Note lateralization and lobe-specific localization.\n"
        "4. IMPORTANT: Output the JSON classification FIRST.\n\n"
        "=== CLASSIFICATION OPTIONS ===\n"
        "1. epilepsy_type: Focal | Generalized | Other\n"
        "2. seizure_type: Focal Onset | Generalized Onset | Unknown/Other\n"
        "3. ez_localization: Temporal | Extratemporal | Multifocal/Hemispheric\n"
        "4. aed_response: Drug-Resistant | Responsive | On Treatment (Unspecified)\n"
        "5. surgery_outcome: Seizure-Free | Improved | No Improvement\n\n"
        "=== OUTPUT FORMAT (JSON FIRST) ===\n"
        "```json\n"
        "{\n"
        '  "epilepsy_type": "YOUR_ANSWER",\n'
        '  "seizure_type": "YOUR_ANSWER",\n'
        '  "ez_localization": "YOUR_ANSWER",\n'
        '  "aed_response": "YOUR_ANSWER",\n'
        '  "surgery_outcome": "YOUR_ANSWER",\n'
        '  "reasoning": "Brief MRI-based reasoning (2-3 sentences)"\n'
        "}\n"
        "```"
    )


@dataclass
class EEGImageConfig:
    """MedGemma-1.5-4B multimodal for EEG montage image interpretation."""
    model_name: str = "google/medgemma-1.5-4b-it"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 1024
    temperature: float = 0.1
    do_sample: bool = False
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 4

    system_prompt: str = (
        "You are a clinical neurophysiologist specializing in EEG interpretation for epilepsy. "
        "Analyze these EEG montage images and provide structured classification.\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Analyze: background rhythm (PDR), interictal discharges, ictal patterns.\n"
        "2. Key patterns: temporal theta/spikes (TLE), frontal fast activity (FLE), "
        "3Hz spike-wave (absence), polyspike-wave (JME), hypsarrhythmia (spasms).\n"
        "3. Note lateralization and distribution of abnormalities.\n"
        "4. IMPORTANT: Output the JSON classification FIRST.\n\n"
        "=== CLASSIFICATION OPTIONS ===\n"
        "1. epilepsy_type: Focal | Generalized | Other\n"
        "2. seizure_type: Focal Onset | Generalized Onset | Unknown/Other\n"
        "3. ez_localization: Temporal | Extratemporal | Multifocal/Hemispheric\n"
        "4. aed_response: Drug-Resistant | Responsive | On Treatment (Unspecified)\n"
        "5. surgery_outcome: Seizure-Free | Improved | No Improvement\n\n"
        "=== OUTPUT FORMAT (JSON FIRST) ===\n"
        "```json\n"
        "{\n"
        '  "epilepsy_type": "YOUR_ANSWER",\n'
        '  "seizure_type": "YOUR_ANSWER",\n'
        '  "ez_localization": "YOUR_ANSWER",\n'
        '  "aed_response": "YOUR_ANSWER",\n'
        '  "surgery_outcome": "YOUR_ANSWER",\n'
        '  "reasoning": "Brief EEG-based reasoning (2-3 sentences)"\n'
        "}\n"
        "```"
    )


@dataclass
class OrchestratorConfig:
    """Llama-3.3-70B-Instruct orchestrator with RAG. Receives text from all agents."""
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    do_sample: bool = False
    repetition_penalty: float = 1.1

    # RAG config
    rag_top_k: int = 5
    rag_embedding_model: str = "BAAI/bge-base-en-v1.5"

    # Hybrid prediction injection format: "binary" (one-vs-rest) or "topk" (ranked list)
    prediction_format: str = "binary"

    system_prompt: str = (
        "You are a senior epileptologist integrating multimodal diagnostic reports for presurgical "
        "epilepsy evaluation. You receive reports from three agents:\n"
        "1) Clinical Text Agent - patient demographics, clinical facts, seizure semiology, clinical notes\n"
        "2) MRI Agent - brain MRI structural abnormalities\n"
        "3) EEG Agent - electrographic patterns\n\n"
        "Agent reports incorporate patient demographics (age, sex, comorbidities) and raw clinical "
        "facts (medications, lab results, etiology) when available. Use this context for more "
        "accurate classification.\n\n"
        "You also have ILAE clinical guidelines for evidence-based classification.\n\n"
        "=== INSTRUCTIONS ===\n"
        "1. Read ALL agent reports. 'Not available' means no data for that modality.\n"
        "2. Identify CONCORDANCE and DISCORDANCE between agents.\n"
        
        "3. Prioritize concordant findings; use ILAE guidelines to resolve conflicts.\n"
        
        "4. IMPORTANT: Output the JSON classification FIRST, then reasoning.\n\n"
        "=== CLASSIFICATION OPTIONS ===\n"
        "1. epilepsy_type: Focal | Generalized | Other\n"
        "2. seizure_type: Focal Onset | Generalized Onset | Unknown/Other\n"
        "3. ez_localization: Temporal | Extratemporal | Multifocal/Hemispheric\n"
        "4. aed_response: Drug-Resistant | Responsive | On Treatment (Unspecified)\n"
        "5. surgery_outcome: Seizure-Free | Improved | No Improvement\n\n"
        "=== OUTPUT FORMAT (JSON FIRST) ===\n"
        "```json\n"
        "{\n"
        '  "epilepsy_type": "YOUR_ANSWER",\n'
        '  "seizure_type": "YOUR_ANSWER",\n'
        '  "ez_localization": "YOUR_ANSWER",\n'
        '  "aed_response": "YOUR_ANSWER",\n'
        '  "surgery_outcome": "YOUR_ANSWER",\n'
        '  "reasoning": "Brief integrated reasoning noting concordance/discordance (2-3 sentences)"\n'
        "}\n"
        "```"
    )


@dataclass
class RAGConfig:
    """ILAE guideline RAG knowledge base."""
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 5


@dataclass
class BaselineConfig:
    """Settings for baseline evaluation scripts."""
    max_patients: Optional[int] = None
    save_dir: str = "logs/baselines"


@dataclass
class PipelineConfig:
    """Multi-GPU pipeline settings."""
    parallel_agents: bool = True
    text_agent_gpus: str = "0,1"        # comma-separated GPU IDs
    mri_agent_gpu: int = 2
    eeg_agent_gpu: int = 3
    orchestrator_gpus: str = "4,5,6,7"
    batch_size: int = 4
    num_workers: int = 8


@dataclass
class MultiTurnConfig:
    """Multi-turn agent communication settings."""
    enabled: bool = True
    max_rounds: int = 3                  # hard cap (safety limit)
    max_questions_per_round: int = 3
    followup_max_tokens: int = 512       # shorter responses for follow-ups


@dataclass
class Mobile2BIDSConfig:
    """Mobile-2 HD-EEG + SEEG stimulation dataset (external validation)."""
    bids_root: str = "external_data/HD-EEG"
    n_subjects: int = 7
    original_sample_rate: int = 8000
    target_sample_rate: int = 200
    num_channels: int = 256
    epoch_samples: int = 2081
    # REVE EEG foundation model
    reve_model_name: str = "brain-bzh/reve-base"
    reve_positions_name: str = "brain-bzh/reve-positions"
    reve_feature_dim: int = 512  # REVE-base outputs 512-dim features
    freeze_backbone: bool = True
    unfreeze_last_n: int = 2     # Unfreeze last N transformer layers for adaptation
    # Task heads
    loc_hidden_dim: int = 512
    loc_output_dim: int = 3       # MNI x, y, z
    region_classes: int = 3       # Temporal / Frontal / Parieto-Occipital
    intensity_classes: int = 2    # Low / High
    # Training
    epochs: int = 200             # v3: increased for convergence with class weighting
    batch_size: int = 32
    lr: float = 5e-5              # v3: lower LR for stable training with focal loss
    dropout: float = 0.2          # v3: increased regularization for small dataset


@dataclass
class EnhancedPipelineConfig:
    """Enhanced pipeline settings: few-shot retrieval + meta-ensemble."""
    few_shot_top_k: int = 3
    few_shot_embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    few_shot_max_text_len: int = 600
    ensemble_weight_pubmedbert: float = 0.60
    ensemble_weight_agent: float = 0.15
    ensemble_weight_tfidf: float = 0.25
    high_confidence_threshold: float = 0.70
    uncertainty_threshold: float = 0.45
    agent_confidence: float = 0.70
    use_few_shot: bool = True
    use_ensemble: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    text_agent: TextAgentConfig = field(default_factory=TextAgentConfig)
    mri_agent: MRIAgentConfig = field(default_factory=MRIAgentConfig)
    eeg_image: EEGImageConfig = field(default_factory=EEGImageConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)
    mobile2_bids: Mobile2BIDSConfig = field(default_factory=Mobile2BIDSConfig)
    enhanced: EnhancedPipelineConfig = field(default_factory=EnhancedPipelineConfig)

    # HuggingFace token for gated models (set via HF_TOKEN environment variable)
    hf_token: str = os.environ.get("HF_TOKEN", "")
