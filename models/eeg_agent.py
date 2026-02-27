"""
EEG Agent: Dual-path architecture for EEG report generation.

Path A (Image): Qwen2.5-VL for EEG montage images (CerebraGloss-style)
Path B (Signal): LaBraM encoder -> projection -> Qwen2.5-7B adapter (CELM-style)

Both paths independently generate text reports. When both are available,
reports are concatenated. NO fine-tuning — zero-shot inference only.

Input: EEG montage images AND/OR raw EEG signals
Output: Structured EEG report text
"""

import torch
import torch.nn as nn
from typing import List, Optional
from PIL import Image


# ---------------------------------------------------------------------------
# Path B: LaBraM encoder for raw EEG signals (kept from v1, proven design)
# ---------------------------------------------------------------------------

class TemporalPatchEmbedding(nn.Module):
    """Convert raw EEG signals into temporal patch tokens."""

    def __init__(self, patch_size: int = 200, hidden_dim: int = 200, max_channels: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.patch_proj = nn.Linear(patch_size, hidden_dim)
        self.channel_embed = nn.Embedding(max_channels, hidden_dim)
        self.temporal_embed = nn.Embedding(64, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, channel_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T = x.shape
        num_patches = T // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]
        x = x.reshape(B, C, num_patches, self.patch_size)
        tokens = self.patch_proj(x)

        temporal_pos = torch.arange(num_patches, device=x.device)
        tokens = tokens + self.temporal_embed(temporal_pos).unsqueeze(0).unsqueeze(0)

        if channel_ids is None:
            channel_ids = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        ch_emb = self.channel_embed(channel_ids)
        tokens = tokens + ch_emb.unsqueeze(2)

        tokens = tokens.reshape(B, C * num_patches, self.hidden_dim)
        return self.norm(tokens)


class LaBraMEncoder(nn.Module):
    """LaBraM-style transformer encoder for raw EEG signals."""

    def __init__(
        self,
        hidden_dim: int = 200,
        num_layers: int = 12,
        num_heads: int = 8,
        patch_size: int = 200,
        max_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_embed = TemporalPatchEmbedding(
            patch_size=patch_size, hidden_dim=hidden_dim, max_channels=max_channels
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        channel_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.patch_embed(x, channel_ids)
        B, N, _ = tokens.shape

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        if attention_mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
            pad_mask = ~torch.cat([cls_mask, attention_mask], dim=1)
        else:
            pad_mask = None

        tokens = self.transformer(tokens, src_key_padding_mask=pad_mask)
        return self.norm(tokens[:, 0])


class EEGSignalProjector(nn.Module):
    """
    Projects LaBraM CLS output into a sequence of soft tokens for the LLM.

    Follows the CELM / E2-LLM pattern: EEG encoder output -> learnable
    projection -> sequence of tokens in LLM embedding space.
    """

    def __init__(
        self,
        labram_dim: int = 200,
        llm_dim: int = 3584,  # Qwen2.5-7B hidden dim
        num_tokens: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Project from single CLS vector to sequence of LLM-compatible tokens
        self.projector = nn.Sequential(
            nn.Linear(labram_dim, llm_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim * 2, llm_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_output: (B, labram_dim) from LaBraM encoder.
        Returns:
            (B, num_tokens, llm_dim) soft token sequence for LLM.
        """
        B = cls_output.shape[0]
        projected = self.projector(cls_output)
        tokens = projected.reshape(B, self.num_tokens, -1)
        return self.norm(tokens)


# ---------------------------------------------------------------------------
# EEG Agent: Dual-path (zero-shot inference only, no fine-tuning)
# ---------------------------------------------------------------------------

class EEGAgent:
    """
    Dual-path EEG report generation agent.

    Path A: Qwen2.5-VL for EEG montage images -> text report
    Path B: LaBraM encoder -> soft tokens -> Qwen2.5-7B -> text report

    NO fine-tuning — zero-shot inference only with specialized prompts.
    """

    def __init__(
        self,
        # VLM config (Path A: images)
        vlm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        # LLM config (Path B: signals)
        llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        # LaBraM config
        labram_hidden_dim: int = 200,
        labram_num_layers: int = 12,
        labram_num_heads: int = 8,
        labram_patch_size: int = 200,
        labram_max_channels: int = 256,
        # Projector config
        num_eeg_tokens: int = 32,
        projector_hidden_dim: int = 512,
        # Generation config
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        # System prompts
        signal_system_prompt: str = "",
        image_system_prompt: str = "",
        torch_dtype: str = "bfloat16",
    ):
        self.vlm_model_name = vlm_model_name
        self.llm_model_name = llm_model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.signal_system_prompt = signal_system_prompt
        self.image_system_prompt = image_system_prompt
        self._dtype = getattr(torch, torch_dtype)
        self.num_eeg_tokens = num_eeg_tokens

        # LaBraM encoder (for signal path)
        self.labram = LaBraMEncoder(
            hidden_dim=labram_hidden_dim,
            num_layers=labram_num_layers,
            num_heads=labram_num_heads,
            patch_size=labram_patch_size,
            max_channels=labram_max_channels,
        )

        # Signal projector (for signal path)
        # Will be properly initialized when LLM is loaded
        self._labram_hidden_dim = labram_hidden_dim
        self._projector_hidden_dim = projector_hidden_dim
        self.signal_projector = None

        # Lazy init for VLM and LLM
        self.vlm = None
        self.vlm_processor = None
        self.llm = None
        self.llm_tokenizer = None
        self.llm_embedding_dim = None

    def load_image_path(self, device: str = "cuda:0"):
        """Load Qwen2.5-VL for EEG image interpretation (Path A)."""
        if self.vlm is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading VLM ({self.vlm_model_name}) for EEG image path...")

        self.vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            self.vlm_model_name,
            dtype=self._dtype,
            device_map=device,
        )

        self.vlm.eval()  # Inference mode only
        print(f"VLM loaded on {device}")

    def load_signal_path(self, device: str = "cuda:0"):
        """Load LaBraM + Qwen2.5-7B for EEG signal interpretation (Path B)."""
        if self.llm is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading LLM ({self.llm_model_name}) for EEG signal path...")

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            padding_side="left",
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=self._dtype,
            device_map=device,
        )

        self.llm.eval()  # Inference mode only

        # Get LLM embedding dimension
        self.llm_embedding_dim = self.llm.config.hidden_size

        # Initialize signal projector now that we know LLM dim
        self.signal_projector = EEGSignalProjector(
            labram_dim=self._labram_hidden_dim,
            llm_dim=self.llm_embedding_dim,
            num_tokens=self.num_eeg_tokens,
            dropout=0.0,  # No dropout in inference
        )

        # Move components to device
        self.labram.to(device)
        self.signal_projector.to(device)

        print(f"LLM + LaBraM loaded on {device}")

    @torch.no_grad()
    def generate_image_report(self, images: List[Image.Image]) -> str:
        """
        Generate EEG report from montage images (Path A: CerebraGloss-style).

        Args:
            images: List of PIL images (EEG montage screenshots)

        Returns:
            Generated EEG report text
        """
        assert self.vlm is not None, "Call load_image_path() first"

        if not images:
            return "No EEG images provided."

        # Build conversation messages
        messages = [
            {
                "role": "system",
                "content": self.image_system_prompt,
            },
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": "Analyze the EEG waveforms and generate a structured report."},
                ],
            },
        ]

        # Apply chat template
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.vlm_processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.vlm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
        )
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        output_ids = self.vlm.generate(**gen_kwargs)

        # Decode only generated tokens
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        report = self.vlm_processor.decode(generated, skip_special_tokens=True)

        return report.strip()

    @torch.no_grad()
    def generate_signal_report(
        self,
        raw_eeg: torch.Tensor,
        channel_ids: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate EEG report from raw signals (Path B: CELM-style).

        Args:
            raw_eeg: (B, C, T) raw EEG tensor at target sample rate
            channel_ids: (B, C) channel indices (optional)

        Returns:
            Generated EEG report text
        """
        assert self.llm is not None, "Call load_signal_path() first"

        device = next(self.llm.parameters()).device
        raw_eeg = raw_eeg.to(device)
        if channel_ids is not None:
            channel_ids = channel_ids.to(device)

        # Encode EEG signals with LaBraM
        cls_output = self.labram(raw_eeg, channel_ids)  # (B, labram_dim)

        # Project to soft tokens
        soft_tokens = self.signal_projector(cls_output)  # (B, num_eeg_tokens, llm_dim)

        # Build text prompt
        prompt = f"{self.signal_system_prompt}\n\nThe EEG signals have been encoded. Generate a structured EEG report."

        # Tokenize prompt
        text_inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(text_inputs["input_ids"])

        # Concatenate soft tokens + text embeddings
        combined_embeds = torch.cat([soft_tokens, text_embeds], dim=1)

        # Create attention mask
        soft_mask = torch.ones(
            soft_tokens.shape[0], soft_tokens.shape[1],
            dtype=torch.long, device=device
        )
        attention_mask = torch.cat([soft_mask, text_inputs["attention_mask"]], dim=1)

        # Generate
        gen_kwargs = dict(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        output_ids = self.llm.generate(**gen_kwargs)

        # Decode
        report = self.llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove prompt from output
        if prompt in report:
            report = report.replace(prompt, "").strip()

        return report

    @torch.no_grad()
    def generate_report(
        self,
        images: Optional[List[Image.Image]] = None,
        raw_eeg: Optional[torch.Tensor] = None,
        channel_ids: Optional[torch.Tensor] = None,
    ) -> str:
        """
        Generate EEG report from available data (images and/or signals).

        Args:
            images: Optional list of EEG montage images
            raw_eeg: Optional raw EEG tensor (B, C, T)
            channel_ids: Optional channel IDs (B, C)

        Returns:
            Combined EEG report text
        """
        reports = []

        # Image path
        if images and self.vlm is not None:
            try:
                image_report = self.generate_image_report(images)
                reports.append(f"=== EEG IMAGE ANALYSIS ===\n{image_report}")
            except Exception as e:
                print(f"Error in image path: {e}")

        # Signal path
        if raw_eeg is not None and self.llm is not None:
            try:
                signal_report = self.generate_signal_report(raw_eeg, channel_ids)
                reports.append(f"=== EEG SIGNAL ANALYSIS ===\n{signal_report}")
            except Exception as e:
                print(f"Error in signal path: {e}")

        if not reports:
            return "No EEG data available for analysis."

        # Concatenate reports
        return "\n\n".join(reports)

    @torch.no_grad()
    def answer_question(
        self,
        original_report: str,
        question: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Answer a follow-up question from the orchestrator.

        Uses the image path (VLM) if images are available, otherwise returns
        a text-only re-analysis.
        """
        if images and self.vlm is not None:
            followup_text = (
                f"You previously generated this EEG report:\n---\n{original_report[:800]}\n---\n\n"
                f"The integrating physician asks:\nQ: {question}\n\n"
                f"Re-examine the EEG images and provide a focused, specific answer."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in images],
                        {"type": "text", "text": followup_text},
                    ],
                },
            ]

            text = self.vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.vlm_processor(
                text=[text], images=images, return_tensors="pt",
            )
            device = next(self.vlm.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

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

            output_ids = self.vlm.generate(**gen_kwargs)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            return self.vlm_processor.decode(generated, skip_special_tokens=True).strip()

        return f"No EEG data available for follow-up. Original report: {original_report[:200]}"
