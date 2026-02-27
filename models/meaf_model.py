"""
MEAF v2: Language-Mediated Multi-Agent Epilepsy Diagnosis Model.

Orchestrates three specialized modality agents (MRI, EEG, Text) that generate
structured clinical reports, synthesized by an orchestrator agent with
RAG-based ILAE guideline adherence into an integrated epilepsy diagnosis.

Architecture:
    Patient Data -> [MRI Agent]  -> MRI Report (text)
                 -> [EEG Agent]  -> EEG Report (text)
                 -> [Text Agent] -> Clinical Summary (text)
                 -> [Orchestrator + ILAE RAG] -> Integrated Diagnosis
                 -> [Report Parser] -> Classification Labels
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from PIL import Image

from .mri_agent import MRIAgent
from .eeg_agent import EEGAgent
from .text_agent import TextAgent
from .orchestrator import OrchestratorAgent
from .report_parser import parse_diagnosis, parse_to_label_indices


class MEAFModel(nn.Module):
    """
    Full MEAF v2 model for language-mediated multi-agent epilepsy diagnosis.

    Unlike v1 (feature vector fusion), v2 operates entirely in text space:
    each agent generates a report, the orchestrator synthesizes them,
    and a parser extracts classification labels for evaluation.
    """

    TASKS = [
        "epilepsy_type", "seizure_type", "ez_localization",
        "aed_response", "surgery_outcome", "survival",
    ]

    def __init__(
        self,
        mri_config: dict = None,
        eeg_config: dict = None,
        text_config: dict = None,
        orchestrator_config: dict = None,
    ):
        super().__init__()
        mri_config = mri_config or {}
        eeg_config = eeg_config or {}
        text_config = text_config or {}
        orchestrator_config = orchestrator_config or {}

        # === Three Modality Agents ===
        self.mri_agent = MRIAgent(
            model_name=mri_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
            lora_r=mri_config.get("lora_r", 64),
            lora_alpha=mri_config.get("lora_alpha", 128),
            lora_dropout=mri_config.get("lora_dropout", 0.05),
            max_new_tokens=mri_config.get("max_new_tokens", 512),
            system_prompt=mri_config.get("system_prompt", ""),
            torch_dtype=mri_config.get("torch_dtype", "bfloat16"),
        )

        self.eeg_agent = EEGAgent(
            vlm_model_name=eeg_config.get("vlm_model_name",
                                           eeg_config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")),
            llm_model_name=eeg_config.get("llm_model_name", "Qwen/Qwen2.5-7B-Instruct"),
            labram_hidden_dim=eeg_config.get("labram_hidden_dim", 200),
            labram_num_layers=eeg_config.get("labram_num_layers", 12),
            labram_num_heads=eeg_config.get("labram_num_heads", 8),
            labram_patch_size=eeg_config.get("labram_patch_size", 200),
            labram_max_channels=eeg_config.get("labram_max_channels", 256),
            num_eeg_tokens=eeg_config.get("num_eeg_tokens", 32),
            lora_r=eeg_config.get("lora_r", 64),
            lora_alpha=eeg_config.get("lora_alpha", 128),
            max_new_tokens=eeg_config.get("max_new_tokens", 512),
            system_prompt=eeg_config.get("system_prompt", ""),
            torch_dtype=eeg_config.get("torch_dtype", "bfloat16"),
        )

        self.text_agent = TextAgent(
            model_name=text_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            lora_r=text_config.get("lora_r", 64),
            lora_alpha=text_config.get("lora_alpha", 128),
            max_new_tokens=text_config.get("max_new_tokens", 512),
            system_prompt=text_config.get("system_prompt", ""),
            torch_dtype=text_config.get("torch_dtype", "bfloat16"),
        )

        # === Orchestrator Agent ===
        self.orchestrator = OrchestratorAgent(
            model_name=orchestrator_config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            lora_r=orchestrator_config.get("lora_r", 64),
            lora_alpha=orchestrator_config.get("lora_alpha", 128),
            max_new_tokens=orchestrator_config.get("max_new_tokens", 768),
            system_prompt=orchestrator_config.get("system_prompt", ""),
            torch_dtype=orchestrator_config.get("torch_dtype", "bfloat16"),
            rag_top_k=orchestrator_config.get("rag_top_k", 5),
        )

    def load_all_models(self, device: Optional[torch.device] = None):
        """Load all agent models. Call once before training/inference."""
        print("Loading MRI Agent...")
        self.mri_agent.load_model(device)
        print("Loading EEG Agent (VLM + LLM)...")
        self.eeg_agent.load_model(device)
        print("Loading Text Agent...")
        self.text_agent.load_model(device)
        print("Loading Orchestrator...")
        self.orchestrator.load_model(device)
        print("All MEAF agents loaded.")

    def load_agent(self, agent_name: str, device: Optional[torch.device] = None):
        """Load a specific agent model."""
        if agent_name == "mri":
            self.mri_agent.load_model(device)
        elif agent_name == "eeg":
            self.eeg_agent.load_model(device)
        elif agent_name == "eeg_vlm":
            self.eeg_agent.load_vlm(device)
        elif agent_name == "eeg_llm":
            self.eeg_agent.load_llm(device)
        elif agent_name == "text":
            self.text_agent.load_model(device)
        elif agent_name == "orchestrator":
            self.orchestrator.load_model(device)

    # ----- Training Forward Passes (per-agent) -----

    def train_mri_agent(
        self,
        images_batch: List[List[Image.Image]],
        target_reports: List[str],
        subcaptions_batch: Optional[List[List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train MRI agent on report generation."""
        return self.mri_agent.forward_train(
            images_batch, target_reports, subcaptions_batch
        )

    def train_eeg_agent_images(
        self,
        images_batch: List[List[Image.Image]],
        target_reports: List[str],
        subcaptions_batch: Optional[List[List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train EEG agent image path."""
        return self.eeg_agent.forward_train_images(
            images_batch, target_reports, subcaptions_batch
        )

    def train_eeg_agent_signals(
        self,
        raw_eeg: torch.Tensor,
        target_reports: List[str],
        channel_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train EEG agent signal path."""
        return self.eeg_agent.forward_train_signals(
            raw_eeg, target_reports, channel_ids, attention_mask
        )

    def train_text_agent(
        self,
        text_fields_batch: List[Dict[str, Optional[str]]],
        target_summaries: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Train text agent on clinical text structuring."""
        return self.text_agent.forward_train(
            text_fields_batch, target_summaries
        )

    def train_orchestrator(
        self,
        mri_reports: List[str],
        eeg_reports: List[str],
        clinical_summaries: List[str],
        target_diagnoses: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Train orchestrator on report synthesis."""
        return self.orchestrator.forward_train(
            mri_reports, eeg_reports, clinical_summaries, target_diagnoses
        )

    # ----- Full Inference Pipeline -----

    @torch.no_grad()
    def diagnose(
        self,
        mri_images: Optional[List[Image.Image]] = None,
        mri_subcaptions: Optional[List[str]] = None,
        eeg_images: Optional[List[Image.Image]] = None,
        eeg_subcaptions: Optional[List[str]] = None,
        raw_eeg: Optional[torch.Tensor] = None,
        eeg_channel_ids: Optional[torch.Tensor] = None,
        eeg_attention_mask: Optional[torch.Tensor] = None,
        semiology: Optional[str] = None,
        mri_report_text: Optional[str] = None,
        eeg_report_text: Optional[str] = None,
        label_maps: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict:
        """
        Full MEAF inference pipeline for a single patient.

        1. Each modality agent generates its report
        2. Orchestrator synthesizes reports with RAG
        3. Parser extracts classification labels

        Args:
            mri_images: Brain MRI images.
            mri_subcaptions: MRI image subcaptions.
            eeg_images: EEG montage images.
            eeg_subcaptions: EEG image subcaptions.
            raw_eeg: Raw EEG signals (C, T).
            eeg_channel_ids: EEG channel indices.
            eeg_attention_mask: EEG attention mask.
            semiology: Seizure semiology text.
            mri_report_text: Raw MRI report text.
            eeg_report_text: Raw EEG report text.
            label_maps: For converting parsed labels to indices.

        Returns:
            Dict with agent reports, final diagnosis, and parsed labels.
        """
        result = {
            "mri_report": "",
            "eeg_report": "",
            "clinical_summary": "",
            "diagnosis": "",
            "parsed_labels": {},
            "label_indices": {},
        }

        # Step 1: MRI Agent
        if mri_images and self.mri_agent.model is not None:
            result["mri_report"] = self.mri_agent.generate_report(
                mri_images, mri_subcaptions
            )

        # Step 2: EEG Agent (dual-path)
        if (eeg_images or raw_eeg is not None) and (
            self.eeg_agent.vlm is not None or self.eeg_agent.llm is not None
        ):
            result["eeg_report"] = self.eeg_agent.generate_report(
                images=eeg_images,
                subcaptions=eeg_subcaptions,
                raw_eeg=raw_eeg,
                channel_ids=eeg_channel_ids,
                eeg_attention_mask=eeg_attention_mask,
            )

        # Step 3: Clinical Text Agent
        if (semiology or mri_report_text or eeg_report_text) and self.text_agent.model is not None:
            result["clinical_summary"] = self.text_agent.generate_summary(
                semiology=semiology,
                mri_report=mri_report_text,
                eeg_report=eeg_report_text,
            )

        # Step 4: Orchestrator synthesis
        if self.orchestrator.model is not None:
            result["diagnosis"] = self.orchestrator.generate_diagnosis(
                mri_report=result["mri_report"],
                eeg_report=result["eeg_report"],
                clinical_summary=result["clinical_summary"],
            )

        # Step 5: Parse diagnosis into labels
        result["parsed_labels"] = parse_diagnosis(result["diagnosis"])

        if label_maps:
            result["label_indices"] = parse_to_label_indices(
                result["diagnosis"], label_maps
            )

        return result

    @torch.no_grad()
    def diagnose_batch(
        self,
        batch: Dict,
        label_maps: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> List[Dict]:
        """
        Run full pipeline on a batch from the dataloader.

        Args:
            batch: Collated batch dict from EpilepsyMultimodalDataset.
            label_maps: Label maps for index conversion.

        Returns:
            List of result dicts, one per patient.
        """
        results = []
        batch_size = len(batch.get("patient_id", []))

        for i in range(batch_size):
            # Extract per-patient data from batch
            mri_images = self._extract_patient_images(batch, "mri_images", i)
            eeg_images = self._extract_patient_images(batch, "eeg_images", i)
            mri_subcaptions = batch.get("mri_subcaptions_text", [[]])[i] if "mri_subcaptions_text" in batch else None
            eeg_subcaptions = batch.get("eeg_subcaptions_text", [[]])[i] if "eeg_subcaptions_text" in batch else None

            text_fields = batch.get("text_fields", [{}])
            semiology = text_fields[i].get("semiology") if i < len(text_fields) else None
            mri_report_text = text_fields[i].get("mri_report") if i < len(text_fields) else None
            eeg_report_text = text_fields[i].get("eeg_report") if i < len(text_fields) else None

            result = self.diagnose(
                mri_images=mri_images,
                mri_subcaptions=mri_subcaptions,
                eeg_images=eeg_images,
                eeg_subcaptions=eeg_subcaptions,
                semiology=semiology,
                mri_report_text=mri_report_text,
                eeg_report_text=eeg_report_text,
                label_maps=label_maps,
            )
            result["patient_id"] = batch["patient_id"][i]
            results.append(result)

        return results

    def _extract_patient_images(self, batch, key, idx) -> Optional[List[Image.Image]]:
        """Extract PIL images for a single patient from batch."""
        if key not in batch or "pil_images" not in batch:
            return None
        pil_key = key.replace("images", "pil_images")
        if pil_key in batch and idx < len(batch[pil_key]):
            return batch[pil_key][idx]
        return None

    # ----- Save / Load -----

    def save_all_adapters(self, save_dir: str):
        """Save all LoRA adapters and signal components."""
        os.makedirs(save_dir, exist_ok=True)
        self.mri_agent.save_adapter(os.path.join(save_dir, "mri_adapter"))
        self.eeg_agent.save_adapters(os.path.join(save_dir, "eeg_adapters"))
        self.text_agent.save_adapter(os.path.join(save_dir, "text_adapter"))
        self.orchestrator.save_adapter(os.path.join(save_dir, "orchestrator_adapter"))
        print(f"All adapters saved to {save_dir}")

    def get_agent_params(self, agent_name: str):
        """Get trainable parameters for a specific agent."""
        if agent_name == "mri":
            return self.mri_agent.get_trainable_params()
        elif agent_name == "eeg":
            return self.eeg_agent.get_trainable_params()
        elif agent_name == "text":
            return self.text_agent.get_trainable_params()
        elif agent_name == "orchestrator":
            return self.orchestrator.get_trainable_params()
        return []
