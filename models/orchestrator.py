"""
Orchestrator Agent: Synthesizes modality agent reports into integrated diagnosis.

Uses GPT-OSS-120B with RAG-based ILAE guideline retrieval.
Receives TEXT reports from all modality agents (text-only communication).
No LoRA, no fine-tuning — zero-shot inference only.

Model: openai/gpt-oss-120b (~240GB BF16, 3-4 A100-80GB)
"""

import torch
from typing import List, Optional

from .rag import ILAEKnowledgeBase


class OrchestratorAgent:
    """
    Orchestrator LLM that synthesizes modality agent reports.

    RAG-augmented: retrieves relevant ILAE guidelines based on agent reports,
    then generates an integrated diagnosis.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        system_prompt: str = "",
        rag_top_k: int = 5,
        rag_embedding_model: str = "BAAI/bge-base-en-v1.5",
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
        self.rag_top_k = rag_top_k
        self.hf_token = hf_token or None

        self.knowledge_base = ILAEKnowledgeBase(
            embedding_model=rag_embedding_model,
        )

        self.model = None
        self.tokenizer = None

    def load_model(self, device_map: Optional[str] = None, max_memory: Optional[dict] = None):
        """Load orchestrator LLM and build RAG index."""
        if self.model is not None:
            return

        if device_map is None:
            device_map = self.device_map

        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token, add_to_git_credential=False)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading Orchestrator ({self.model_name}) with device_map={device_map}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        print(f"Orchestrator loaded: {self.model_name}")

        # Build RAG index
        self.knowledge_base.build_index()

    def _build_rag_query(
        self,
        text_report: str = "",
        mri_report: str = "",
        eeg_report: str = "",
    ) -> str:
        """Build RAG query from agent reports."""
        parts = []
        if text_report:
            parts.append(f"Clinical: {text_report[:300]}")
        if mri_report:
            parts.append(f"MRI: {mri_report[:300]}")
        if eeg_report:
            parts.append(f"EEG: {eeg_report[:300]}")
        return " ".join(parts) if parts else "epilepsy classification diagnosis"

    def _build_messages(
        self,
        text_report: str = "",
        mri_report: str = "",
        eeg_report: str = "",
        rag_context: str = "",
    ) -> List[dict]:
        """Build chat messages with agent reports and RAG context."""
        user_parts = []

        if rag_context:
            user_parts.append(f"RELEVANT CLINICAL GUIDELINES:\n{rag_context}\n")

        user_parts.append("MODALITY AGENT REPORTS:\n")

        if text_report:
            user_parts.append(f"--- CLINICAL TEXT AGENT REPORT ---\n{text_report}\n")
        else:
            user_parts.append("--- CLINICAL TEXT AGENT REPORT ---\nNo clinical text available.\n")

        if mri_report:
            user_parts.append(f"--- MRI AGENT REPORT ---\n{mri_report}\n")
        else:
            user_parts.append("--- MRI AGENT REPORT ---\nNo MRI data available.\n")

        if eeg_report:
            user_parts.append(f"--- EEG AGENT REPORT ---\n{eeg_report}\n")
        else:
            user_parts.append("--- EEG AGENT REPORT ---\nNo EEG data available.\n")

        user_parts.append(
            "\nBased on the above reports and clinical guidelines, provide your "
            "integrated epilepsy diagnosis in the specified structured format."
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    @torch.no_grad()
    def synthesize_guidelines(self, rag_context: str) -> str:
        """
        Synthesize retrieved guidelines (V(R(K))) as described in Section 2.2.
        
        This implements the Guideline Synthesis LLM V(·). It takes the raw 
        retrieved segments R(K) and synthesizes them into procedural guidelines G.
        """
        if not rag_context or "No relevant clinical guidelines" in rag_context:
            return rag_context

        assert self.model is not None, "Call load_model() first"

        system_prompt = (
            "You are an expert epileptologist. Your task is to synthesize the following "
            "retrieved clinical guidelines into a single, cohesive procedural guideline "
            "for presurgical epilepsy evaluation. Focus on actionable decision-making criteria."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Synthesize these retrieved guidelines:\n\n{rag_context}"},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
            
        output_ids = self.model.generate(**gen_kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        synthesis = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        return f"=== SYNTHESIZED PROCEDURAL GUIDELINES (G) ===\n{synthesis}"

    @torch.no_grad()
    def generate_diagnosis(
        self,
        text_report: str = "",
        mri_report: str = "",
        eeg_report: str = "",
    ) -> str:
        """
        Generate integrated diagnosis from agent reports.

        All communication is text-only.
        """
        assert self.model is not None, "Call load_model() first"

        # RAG retrieval
        rag_query = self._build_rag_query(text_report, mri_report, eeg_report)
        rag_context = self.knowledge_base.retrieve_formatted(
            rag_query, top_k=self.rag_top_k
        )

        messages = self._build_messages(
            text_report, mri_report, eeg_report, rag_context
        )

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        output_ids = self.model.generate(**gen_kwargs)

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
            print("Orchestrator unloaded")
