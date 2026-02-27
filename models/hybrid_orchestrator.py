"""
Hybrid Orchestrator: Combines discriminative model predictions with generative agent reports.

The key innovation: injects discriminative predictions + confidence scores
into the orchestrator prompt alongside agent reports and RAG context.

Supports two prediction formats:
  - "binary": Per-class one-vs-rest probabilities (softer, lets agents reason)
  - "topk": Ranked list of all classes by probability (more directive)

This bridges discriminative (ML classifiers) and generative (LLM agents) approaches.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.orchestrator import OrchestratorAgent


FOLLOWUP_SYSTEM_PROMPT = (
    "You are reviewing specialist agent reports for an epilepsy presurgical evaluation. "
    "Identify any gaps, ambiguities, or discordances in the reports that need clarification.\n\n"
    "If you need more information from specific agents, output ONLY this JSON:\n"
    "```json\n"
    "{\n"
    '  "status": "FOLLOWUP",\n'
    '  "questions": [\n'
    '    {"agent": "text", "question": "Your specific question for the clinical text agent"},\n'
    '    {"agent": "mri", "question": "Your specific question for the MRI agent"},\n'
    '    {"agent": "eeg", "question": "Your specific question for the EEG agent"}\n'
    "  ]\n"
    "}\n"
    "```\n"
    "Include only the agents you need to ask. You can ask 1-3 agents.\n\n"
    "If you have enough information to make a confident diagnosis, output ONLY:\n"
    "```json\n"
    '{"status": "SATISFIED"}\n'
    "```\n\n"
    "Focus follow-up questions on:\n"
    "- Discordance between modalities (e.g., MRI shows temporal but EEG shows frontal)\n"
    "- Missing lateralization or localization details\n"
    "- Ambiguous drug response classification\n"
    "- Unclear surgical candidacy or outcome prediction\n"
    "- Request for specific clinical signs or patterns not mentioned in initial report\n"
)


class HybridOrchestrator(OrchestratorAgent):
    """
    Extends the base Orchestrator with discriminative model prediction injection
    and multi-turn follow-up question generation.

    The orchestrator receives:
    1. Discriminative model predictions with confidence scores
    2. Agent free-text reports (text, MRI, EEG)
    3. ILAE guideline RAG context
    """

    TASK_NAMES = [
        "epilepsy_type", "seizure_type", "ez_localization",
        "aed_response", "surgery_outcome",
    ]

    CLF_DISPLAY_NAMES = {
        "text_classifier": "Text Classifier (PubMedBERT)",
        "mri_classifier": "MRI Classifier (MedSigLIP-448)",
        "eeg_classifier": "EEG Classifier (MedSigLIP-448)",
        "ensemble": "Ensemble (Late Fusion)",
    }

    def __init__(self, *args, prediction_format: str = "binary", **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_format = prediction_format

    def format_predictions(
        self,
        predictions: Dict[str, Dict[str, List[Tuple[str, float]]]],
    ) -> str:
        """Dispatch to binary or topk formatting based on config."""
        if self.prediction_format == "binary":
            return self._format_predictions_binary(predictions)
        return self._format_predictions_topk(predictions)

    def _format_predictions_binary(
        self,
        predictions: Dict[str, Dict[str, List[Tuple[str, float]]]],
    ) -> str:
        """
        Format as confidence-tiered per-class probabilities.

        Uses tiered framing based on max probability:
          > 0.85: STRONG RECOMMENDATION
          > 0.70: Moderate signal
          < 0.50: Uncertain — rely on clinical reasoning
        """
        sections = []

        for clf_name, clf_preds in predictions.items():
            if clf_preds is None:
                continue

            display_name = self.CLF_DISPLAY_NAMES.get(clf_name, clf_name)
            lines = [f"{display_name}:"]

            for task in self.TASK_NAMES:
                if task not in clf_preds or clf_preds[task] is None:
                    continue

                # Determine confidence tier
                max_prob = max(prob for _, prob in clf_preds[task])
                top_label = max(clf_preds[task], key=lambda x: x[1])[0]

                if max_prob > 0.85:
                    tier = f"STRONG RECOMMENDATION: {top_label} ({max_prob:.0%} confidence)"
                elif max_prob > 0.70:
                    tier = f"Moderate signal: {top_label} ({max_prob:.0%})"
                else:
                    tier = "Uncertain (split across classes)"

                lines.append(f"  {task}: {tier}")
                for label, prob in clf_preds[task]:
                    lines.append(f"    P({label}) = {prob:.2f}")

            sections.append("\n".join(lines))

        if not sections:
            return ""

        header = (
            "=== DISCRIMINATIVE MODEL SIGNALS ===\n"
            "These are data-driven probability estimates from classifiers trained on labeled patient data.\n"
            "STRONG RECOMMENDATION means the classifier is very confident — follow it unless you have\n"
            "compelling clinical evidence to the contrary.\n\n"
        )
        return header + "\n\n".join(sections)

    def _format_predictions_topk(
        self,
        predictions: Dict[str, Dict[str, List[Tuple[str, float]]]],
    ) -> str:
        """Format as ranked list (original top-k format, kept for ablation)."""
        sections = []

        for clf_name, clf_preds in predictions.items():
            if clf_preds is None:
                continue

            display_name = self.CLF_DISPLAY_NAMES.get(clf_name, clf_name)
            lines = [f"{display_name}:"]

            for task in self.TASK_NAMES:
                if task not in clf_preds or clf_preds[task] is None:
                    continue
                sorted_preds = sorted(clf_preds[task], key=lambda x: x[1], reverse=True)
                pred_str = ", ".join(f"{label} ({conf:.2f})" for label, conf in sorted_preds)
                lines.append(f"  {task}: {pred_str}")

            sections.append("\n".join(lines))

        if not sections:
            return ""

        header = (
            "=== DISCRIMINATIVE MODEL PREDICTIONS ===\n"
            "These are data-driven probability estimates from trained classifiers.\n"
            "Use these to calibrate your confidence, especially when agent reports are ambiguous.\n\n"
        )
        return header + "\n\n".join(sections)

    def _build_hybrid_messages(
        self,
        text_report: str,
        mri_report: str,
        eeg_report: str,
        discriminative_predictions: Dict[str, Dict[str, List[Tuple[str, float]]]],
        rag_context: str = "",
        few_shot_examples: str = "",
    ) -> list:
        """Build messages with few-shot examples + discriminative predictions injected."""
        user_parts = []

        # 1. Few-shot examples FIRST (JSON demonstrations calibrate both format and content)
        if few_shot_examples:
            user_parts.append(
                "=== REFERENCE CASES (similar patients with known correct diagnoses) ===\n"
                "These show the correct classification for patients similar to this one.\n\n"
                f"{few_shot_examples}\n"
            )

        # 2. Discriminative predictions (confidence-tiered signals)
        pred_text = self.format_predictions(discriminative_predictions)
        if pred_text:
            user_parts.append(pred_text)

        # 3. Agent reports
        user_parts.append("=== SPECIALIST AGENT REPORTS ===")

        user_parts.append("--- Clinical Text Agent Report ---")
        if text_report and text_report.strip():
            user_parts.append(text_report.strip())
        else:
            user_parts.append("No clinical text data available.")

        user_parts.append("\n--- MRI Agent Report ---")
        if mri_report and mri_report.strip():
            user_parts.append(mri_report.strip())
        else:
            user_parts.append("No MRI data available.")

        user_parts.append("\n--- EEG Agent Report ---")
        if eeg_report and eeg_report.strip():
            user_parts.append(eeg_report.strip())
        else:
            user_parts.append("No EEG data available.")

        # 4. RAG context
        if rag_context:
            user_parts.append(f"\n=== ILAE CLINICAL GUIDELINES ===\n{rag_context}")

        # 5. Task instruction at END with JSON template
        user_parts.append(
            "\n=== TASK ===\n"
            "Integrate all evidence above and classify this patient.\n"
            "When the classifier shows STRONG RECOMMENDATION, follow it unless "
            "agent reports provide direct contradicting clinical evidence.\n\n"
            "First briefly reason (2-3 sentences), then output your classification "
            "as a JSON code block:\n"
            "```json\n"
            "{\n"
            '  "epilepsy_type": "Focal | Generalized | Other",\n'
            '  "seizure_type": "Focal Onset | Generalized Onset | Unknown/Other",\n'
            '  "ez_localization": "Temporal | Extratemporal | Multifocal/Hemispheric",\n'
            '  "aed_response": "Drug-Resistant | Responsive | On Treatment (Unspecified)",\n'
            '  "surgery_outcome": "Seizure-Free | Improved | No Improvement"\n'
            "}\n"
            "```"
        )

        messages = [
            {"role": "user", "content": self.system_prompt + "\n\n" + "\n".join(user_parts)},
        ]
        return messages

    def _get_rag_context(self, text_report: str, mri_report: str, eeg_report: str) -> str:
        """Retrieve RAG context from agent reports."""
        rag_query = self._build_rag_query(text_report, mri_report, eeg_report)
        if self.knowledge_base is not None:
            return self.knowledge_base.retrieve_formatted(rag_query, top_k=self.rag_top_k)
        return ""

    def _generate_from_messages(
        self, messages: list, max_new_tokens: Optional[int] = None, assistant_prefill: str = ""
    ) -> str:
        """Generate text from chat messages, optionally with assistant pre-fill."""
        import torch

        if assistant_prefill:
            # Add assistant pre-fill: append partial assistant message to force output format
            messages_with_prefill = messages + [
                {"role": "assistant", "content": assistant_prefill}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages_with_prefill,
                return_tensors="pt",
                add_generation_prompt=False,
                return_dict=True,
            )
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if self.repetition_penalty > 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # If we used pre-fill, prepend it back so the full response is available for parsing
        if assistant_prefill:
            result = assistant_prefill + result

        return result

    def generate_hybrid_diagnosis(
        self,
        text_report: str,
        mri_report: str,
        eeg_report: str,
        discriminative_predictions: Dict[str, Dict[str, List[Tuple[str, float]]]],
        few_shot_examples: str = "",
    ) -> str:
        """
        Generate integrated diagnosis using both discriminative and generative inputs.

        Args:
            text_report: Clinical text agent report
            mri_report: MRI agent report
            eeg_report: EEG agent report
            discriminative_predictions: Per-classifier prediction probabilities
            few_shot_examples: Formatted few-shot reference cases (from FewShotRetriever)

        Returns:
            Integrated diagnosis text with JSON classification.
        """
        assert self.model is not None, "Call load_model() first"

        rag_context = self._get_rag_context(text_report, mri_report, eeg_report)
        
        # TODO: enable this guideline synthesis step later.
        # currently commented out to save inference time.
        # rag_context = self.synthesize_guidelines(rag_context)

        messages = self._build_hybrid_messages(
            text_report, mri_report, eeg_report,
            discriminative_predictions, rag_context,
            few_shot_examples=few_shot_examples,
        )
        # No assistant pre-fill — it was hurting accuracy in v2 (50.3% vs 61.4%)
        # Instead rely on strong prompt instructions and better JSON extraction
        return self._generate_from_messages(messages)

    def generate_followup_questions(
        self,
        text_report: str,
        mri_report: str,
        eeg_report: str,
        discriminative_predictions: Dict[str, Dict[str, List[Tuple[str, float]]]],
        conversation_history: Optional[List[dict]] = None,
        few_shot_examples: str = "",
    ) -> dict:
        """
        Review agent reports and decide whether to ask follow-up questions.

        Returns:
            {"status": "SATISFIED"} or
            {"status": "FOLLOWUP", "questions": [{"agent": "text|mri|eeg", "question": "..."}]}
        """
        assert self.model is not None, "Call load_model() first"

        rag_context = self._get_rag_context(text_report, mri_report, eeg_report)

        # TODO: enable this guideline synthesis step later.
        # currently commented out to save inference time.
        # rag_context = self.synthesize_guidelines(rag_context)

        messages = self._build_hybrid_messages(
            text_report, mri_report, eeg_report,
            discriminative_predictions, rag_context,
            few_shot_examples=few_shot_examples,
        )

        # Replace the TASK instruction with followup prompt
        content = messages[0]["content"]
        # Remove the existing === TASK === section
        task_idx = content.rfind("\n=== TASK ===")
        if task_idx > 0:
            content = content[:task_idx]
        content += "\n\n" + FOLLOWUP_SYSTEM_PROMPT
        messages[0]["content"] = content

        # Add conversation history if any
        if conversation_history:
            for entry in conversation_history:
                messages.append(entry)

        response = self._generate_from_messages(messages, max_new_tokens=512)
        return self._parse_followup_response(response)

    @staticmethod
    def _parse_followup_response(response: str) -> dict:
        """Parse orchestrator's follow-up response into structured format."""
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "status" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Try raw JSON
        try:
            parsed = json.loads(response.strip())
            if "status" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        # Check for keywords
        lower = response.lower()
        if "satisfied" in lower:
            return {"status": "SATISFIED"}

        # Default: satisfied (safe fallback to avoid infinite loops)
        return {"status": "SATISFIED"}


def predictions_from_probabilities(
    probabilities: Dict[str, np.ndarray],
    label_maps: Dict[str, Dict[str, int]],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Convert classifier probability arrays to structured predictions.

    Args:
        probabilities: {task_name: np.array of shape (n_classes,)}
        label_maps: {task_name: {label_str: index}}

    Returns:
        {task_name: [(label_str, probability), ...]}
    """
    predictions = {}
    for task, probs in probabilities.items():
        if task not in label_maps:
            continue
        inv_map = {v: k for k, v in label_maps[task].items()}
        task_preds = []
        for idx, prob in enumerate(probs):
            label = inv_map.get(idx, f"class_{idx}")
            task_preds.append((label, float(prob)))
        predictions[task] = task_preds
    return predictions
