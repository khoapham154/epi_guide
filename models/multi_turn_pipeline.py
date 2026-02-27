"""
Multi-Turn Agent Communication Pipeline.

Manages iterative communication between the orchestrator and modality agents.
The orchestrator can ask follow-up questions to specific agents when it
identifies gaps, discordances, or ambiguities in the initial reports.

Flow:
    Round 0: Agents generate initial reports (parallel if multi-GPU)
    Round 1+: Orchestrator reviews -> asks follow-ups OR declares SATISFIED
    Final: Orchestrator produces integrated diagnosis with enriched reports
"""

import json
from typing import Dict, List, Optional, Tuple

from PIL import Image


class MultiTurnPipeline:
    """
    Manages iterative communication between orchestrator and modality agents.

    All models must be pre-loaded (supports multi-GPU with all models resident).
    """

    def __init__(
        self,
        text_agent,
        mri_agent,
        eeg_agent,
        orchestrator,
        max_rounds: int = 3,
        max_questions_per_round: int = 3,
        followup_max_tokens: int = 512,
    ):
        self.text_agent = text_agent
        self.mri_agent = mri_agent
        self.eeg_agent = eeg_agent
        self.orchestrator = orchestrator
        self.max_rounds = max_rounds
        self.max_questions_per_round = max_questions_per_round
        self.followup_max_tokens = followup_max_tokens

    def run_patient(
        self,
        patient_data: dict,
        discriminative_predictions: dict,
        text_report: str = "",
        mri_report: str = "",
        eeg_report: str = "",
        few_shot_examples: str = "",
    ) -> dict:
        """
        Run multi-turn pipeline for a single patient.

        Args:
            patient_data: {
                "semiology": str, "mri_report_text": str, "eeg_report_text": str,
                "mri_images": List[PIL.Image], "eeg_images": List[PIL.Image]
            }
            discriminative_predictions: per-classifier predictions dict
            text_report: pre-generated text agent report (from cache or round 0)
            mri_report: pre-generated MRI agent report
            eeg_report: pre-generated EEG agent report

        Returns:
            {
                "final_diagnosis": str,
                "conversation_log": list,
                "num_rounds": int,
                "text_report": str,
                "mri_report": str,
                "eeg_report": str,
            }
        """
        conversation_log = []

        # Round 0: initial reports (may already be provided via cache)
        if not text_report and self.text_agent is not None:
            text_report = self._generate_text_report(patient_data)
        if not mri_report and self.mri_agent is not None:
            mri_report = self._generate_mri_report(patient_data)
        if not eeg_report and self.eeg_agent is not None:
            eeg_report = self._generate_eeg_report(patient_data)

        conversation_log.append({
            "round": 0,
            "type": "initial_reports",
            "text_len": len(text_report),
            "mri_len": len(mri_report),
            "eeg_len": len(eeg_report),
        })

        # Iterative follow-up rounds
        conversation_history = []

        for round_num in range(1, self.max_rounds + 1):
            # Orchestrator decides: SATISFIED or FOLLOWUP
            followup_result = self.orchestrator.generate_followup_questions(
                text_report=text_report,
                mri_report=mri_report,
                eeg_report=eeg_report,
                discriminative_predictions=discriminative_predictions,
                conversation_history=conversation_history if conversation_history else None,
                few_shot_examples=few_shot_examples,
            )

            if followup_result.get("status") == "SATISFIED":
                conversation_log.append({
                    "round": round_num,
                    "type": "satisfied",
                })
                break

            questions = followup_result.get("questions", [])[:self.max_questions_per_round]

            if not questions:
                conversation_log.append({
                    "round": round_num,
                    "type": "no_questions",
                })
                break

            round_answers = []
            for q in questions:
                agent_name = q.get("agent", "")
                question_text = q.get("question", "")

                answer = self._route_question(
                    agent_name, question_text,
                    text_report, mri_report, eeg_report,
                    patient_data,
                )
                round_answers.append({
                    "agent": agent_name,
                    "question": question_text,
                    "answer": answer,
                })

            # Append follow-up Q&A to agent reports
            text_report, mri_report, eeg_report = self._enrich_reports(
                text_report, mri_report, eeg_report, round_answers
            )

            # Build conversation history entry for orchestrator context
            qa_summary = "\n".join(
                f"[Follow-up to {a['agent']} agent]\nQ: {a['question']}\nA: {a['answer']}"
                for a in round_answers
            )
            conversation_history.append({
                "role": "assistant",
                "content": f"Follow-up round {round_num}:\n{qa_summary}",
            })

            conversation_log.append({
                "round": round_num,
                "type": "followup",
                "num_questions": len(round_answers),
                "questions_answers": round_answers,
            })

        # Final diagnosis with all accumulated information
        final_diagnosis = self.orchestrator.generate_hybrid_diagnosis(
            text_report=text_report,
            mri_report=mri_report,
            eeg_report=eeg_report,
            discriminative_predictions=discriminative_predictions,
            few_shot_examples=few_shot_examples,
        )

        total_rounds = len(conversation_log)

        return {
            "final_diagnosis": final_diagnosis,
            "conversation_log": conversation_log,
            "num_rounds": total_rounds,
            "text_report": text_report,
            "mri_report": mri_report,
            "eeg_report": eeg_report,
        }

    def _generate_text_report(self, patient_data: dict) -> str:
        """Generate initial text agent report."""
        try:
            return self.text_agent.generate_summary(
                demographics_notes=patient_data.get("demographics_notes"),
                raw_facts=patient_data.get("raw_facts"),
                semiology=patient_data.get("semiology"),
                mri_report=patient_data.get("mri_report_text"),
                eeg_report=patient_data.get("eeg_report_text"),
            )
        except Exception as e:
            return f"ERROR: {e}"

    def _generate_mri_report(self, patient_data: dict) -> str:
        """Generate initial MRI agent report."""
        images = patient_data.get("mri_images", [])
        if not images:
            return ""
        try:
            return self.mri_agent.generate_report(images)
        except Exception as e:
            return f"ERROR: {e}"

    def _generate_eeg_report(self, patient_data: dict) -> str:
        """Generate initial EEG agent report."""
        images = patient_data.get("eeg_images", [])
        if not images:
            return ""
        try:
            # EEG agent may be MRIAgent (reused) or EEGAgent
            if hasattr(self.eeg_agent, 'generate_image_report'):
                return self.eeg_agent.generate_image_report(images)
            return self.eeg_agent.generate_report(images)
        except Exception as e:
            return f"ERROR: {e}"

    def _route_question(
        self,
        agent_name: str,
        question: str,
        text_report: str,
        mri_report: str,
        eeg_report: str,
        patient_data: dict,
    ) -> str:
        """Route a follow-up question to the appropriate agent."""
        try:
            if agent_name == "text" and self.text_agent is not None:
                return self.text_agent.answer_question(
                    original_report=text_report,
                    question=question,
                    demographics_notes=patient_data.get("demographics_notes"),
                    raw_facts=patient_data.get("raw_facts"),
                    semiology=patient_data.get("semiology"),
                    mri_report=patient_data.get("mri_report_text"),
                    eeg_report=patient_data.get("eeg_report_text"),
                    max_new_tokens=self.followup_max_tokens,
                )
            elif agent_name == "mri" and self.mri_agent is not None:
                return self.mri_agent.answer_question(
                    original_report=mri_report,
                    question=question,
                    images=patient_data.get("mri_images", []),
                    max_new_tokens=self.followup_max_tokens,
                )
            elif agent_name == "eeg" and self.eeg_agent is not None:
                images = patient_data.get("eeg_images", [])
                if hasattr(self.eeg_agent, 'answer_question'):
                    return self.eeg_agent.answer_question(
                        original_report=eeg_report,
                        question=question,
                        images=images,
                        max_new_tokens=self.followup_max_tokens,
                    )
        except Exception as e:
            return f"Error answering follow-up: {e}"

        return f"Agent '{agent_name}' not available for follow-up."

    @staticmethod
    def _enrich_reports(
        text_report: str,
        mri_report: str,
        eeg_report: str,
        round_answers: List[dict],
    ) -> Tuple[str, str, str]:
        """Append follow-up Q&A to the relevant agent reports."""
        for answer_entry in round_answers:
            agent = answer_entry["agent"]
            qa_block = (
                f"\n--- Follow-up Clarification ---\n"
                f"Q: {answer_entry['question']}\n"
                f"A: {answer_entry['answer']}"
            )

            if agent == "text":
                text_report += qa_block
            elif agent == "mri":
                mri_report += qa_block
            elif agent == "eeg":
                eeg_report += qa_block

        return text_report, mri_report, eeg_report
