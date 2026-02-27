"""
Text Agent: MedGemma-27B-text-it for clinical text structuring.

Zero-shot with prompt engineering (NO RAG, NO fine-tuning).
RAG is handled by the Orchestrator agent only.
Generates structured clinical summaries from patient text data.

Model: google/medgemma-27b-text-it (~54GB BF16, 1 A100-80GB)
"""

import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextAgent:
    """
    Clinical text structuring agent using MedGemma-27B-text-it.

    Takes raw clinical text (semiology, MRI reports, EEG reports) and generates
    structured clinical summaries with classification labels.

    NO RAG - RAG is only in the Orchestrator.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-27b-text-it",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        system_prompt: str = "",
        hf_token: str = "",
    ):
        self.model_name = model_name
        self._dtype = getattr(torch, torch_dtype)
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.system_prompt = system_prompt
        self.hf_token = hf_token or None

        self.model = None
        self.tokenizer = None

    def load_model(self, device_map: Optional[str] = None, max_memory: Optional[Dict] = None):
        """Load MedGemma-27B-text-it."""
        if self.model is not None:
            return

        if device_map is None:
            device_map = self.device_map

        # Login to HF for gated models
        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token, add_to_git_credential=False)

        print(f"Loading {self.model_name} with device_map={device_map}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        load_kwargs = dict(
            torch_dtype=self._dtype,
            device_map=device_map,
            token=self.hf_token,
        )
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        self.model.eval()
        print(f"Text Agent loaded: {self.model_name}")

    def _build_messages(
        self,
        demographics_notes: Optional[str] = None,
        raw_facts: Optional[str] = None,
        semiology: Optional[str] = None,
        mri_report: Optional[str] = None,
        eeg_report: Optional[str] = None,
    ) -> List[dict]:
        """Build chat messages from patient data."""
        user_parts = ["=== PATIENT CLINICAL INFORMATION ===\n"]

        if demographics_notes and demographics_notes.strip():
            user_parts.append(f"PATIENT DEMOGRAPHICS:\n{demographics_notes.strip()}\n")

        if raw_facts and raw_facts.strip():
            user_parts.append(f"CLINICAL FACTS:\n{raw_facts.strip()}\n")

        if semiology and semiology.strip():
            user_parts.append(f"SEIZURE SEMIOLOGY:\n{semiology.strip()}\n")
        else:
            user_parts.append("SEIZURE SEMIOLOGY: Not available.\n")

        if mri_report and mri_report.strip():
            user_parts.append(f"MRI REPORT:\n{mri_report.strip()}\n")
        else:
            user_parts.append("MRI REPORT: Not available.\n")

        if eeg_report and eeg_report.strip():
            user_parts.append(f"EEG REPORT:\n{eeg_report.strip()}\n")
        else:
            user_parts.append("EEG REPORT: Not available.\n")

        user_parts.append(
            "\nGenerate a comprehensive structured analysis following "
            "the format specified in your instructions."
        )

        messages = [
            {"role": "user", "content": self.system_prompt + "\n\n" + "\n".join(user_parts)},
        ]
        return messages

    @torch.no_grad()
    def generate_summary(
        self,
        demographics_notes: Optional[str] = None,
        raw_facts: Optional[str] = None,
        semiology: Optional[str] = None,
        mri_report: Optional[str] = None,
        eeg_report: Optional[str] = None,
    ) -> str:
        """
        Generate structured clinical summary for a single patient.

        Returns:
            Structured clinical summary text.
        """
        assert self.model is not None, "Call load_model() first"

        messages = self._build_messages(demographics_notes, raw_facts, semiology, mri_report, eeg_report)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            repetition_penalty=self.repetition_penalty,
        )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        summary = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return summary

    @torch.no_grad()
    def answer_question(
        self,
        original_report: str,
        question: str,
        demographics_notes: Optional[str] = None,
        raw_facts: Optional[str] = None,
        semiology: Optional[str] = None,
        mri_report: Optional[str] = None,
        eeg_report: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Answer a follow-up question from the orchestrator.

        The agent receives its original report, the patient data,
        and the specific question, then generates a focused response.
        """
        assert self.model is not None, "Call load_model() first"

        followup_prompt = (
            f"You previously generated this clinical analysis:\n"
            f"---\n{original_report[:1000]}\n---\n\n"
            f"The integrating physician has a follow-up question:\n"
            f"Q: {question}\n\n"
            f"Re-examine the clinical data below and provide a focused, specific answer. "
            f"If you need to revise your previous assessment, explain why.\n"
        )

        messages = self._build_messages(demographics_notes, raw_facts, semiology, mri_report, eeg_report)
        messages[0]["content"] = followup_prompt + "\n" + messages[0]["content"]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            repetition_penalty=self.repetition_penalty,
        )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            print("Text Agent unloaded")
