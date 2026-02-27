"""
MRI Agent: MedGemma-1.5-4B-it for epilepsy brain MRI report generation.

Zero-shot inference. Generates structured radiology reports
focused on epileptogenic lesion detection.

Model: google/medgemma-1.5-4b-it (~8GB BF16, 1 GPU)
"""

import torch
from typing import List, Optional
from PIL import Image


class MRIAgent:
    """
    MRI report generation agent using MedGemma-1.5-4B-it.

    Zero-shot inference with specialized epilepsy prompt.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 4,
        system_prompt: str = "",
        hf_token: str = "",
    ):
        self.model_name = model_name
        self._dtype = getattr(torch, torch_dtype)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.system_prompt = system_prompt
        self.hf_token = hf_token or None

        self.model = None
        self.processor = None
        self.device = None

    def load_model(self, device: str = "cuda:0"):
        """Load MedGemma-1.5-4B-it for MRI report generation."""
        if self.model is not None:
            return

        if self.hf_token:
            from huggingface_hub import login
            login(token=self.hf_token, add_to_git_credential=False)

        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"Loading MRI Agent ({self.model_name}) on {device}...")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            device_map=device,
            token=self.hf_token,
        )

        self.device = device
        self.model.eval()
        print(f"MRI Agent loaded on {device}")

    @torch.no_grad()
    def generate_report(
        self,
        images: List[Image.Image],
        prompt: str = "",
    ) -> str:
        """
        Generate MRI report for a single patient.

        Args:
            images: List of PIL images (MRI subfigures)
            prompt: Optional custom prompt override

        Returns:
            Generated radiology report text.
        """
        assert self.model is not None, "Call load_model() first"

        if not images:
            return "No MRI images provided."

        # Build conversation
        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})

        task_text = prompt if prompt else (
            self.system_prompt + "\n\nGenerate the radiology report based on these MRI images."
        )
        user_content.append({"type": "text", "text": task_text})

        messages = [
            {"role": "user", "content": user_content},
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        report = self.processor.decode(generated, skip_special_tokens=True)
        return report.strip()

    @torch.no_grad()
    def answer_question(
        self,
        original_report: str,
        question: str,
        images: List[Image.Image],
        max_new_tokens: int = 512,
    ) -> str:
        """
        Answer a follow-up question from the orchestrator with re-examination of images.
        """
        assert self.model is not None, "Call load_model() first"

        if not images:
            return "No images available for re-examination."

        followup_text = (
            f"You previously generated this report:\n---\n{original_report[:800]}\n---\n\n"
            f"The integrating physician asks:\nQ: {question}\n\n"
            f"Re-examine the images and provide a focused, specific answer."
        )

        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": followup_text})

        messages = [{"role": "user", "content": user_content}]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
        )
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

        output_ids = self.model.generate(**gen_kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def unload_model(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.device = None
            torch.cuda.empty_cache()
            print("MRI Agent unloaded")
