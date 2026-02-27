"""
Meta-Ensemble v4: Calibrated task-specific discriminative ensemble + LLM tie-breaking.

Key improvements over v3:
  1. Task-specific PubMedBERT/TF-IDF weights calibrated from OOF predictions
  2. OOF index maps for correct CSV row → OOF array alignment
  3. LLM tie-breaking only when discriminative margin < threshold

At startup, calibrate() sweeps bert_w per task and picks the weight
that maximizes OOF accuracy. This is legitimate since OOF predictions
are out-of-fold (each patient predicted by model that didn't train on it).
"""

import json
from typing import Dict, Optional

import numpy as np


class MetaEnsemble:
    """Calibrated per-task discriminative ensemble with LLM tie-breaking."""

    def __init__(
        self,
        label_maps: Dict[str, Dict[str, int]],
        weight_pubmedbert: float = 0.60,
        weight_tfidf: float = 0.40,
        tie_break_margin: float = 0.10,
    ):
        self.label_maps = label_maps
        self.default_weight_bert = weight_pubmedbert
        self.default_weight_tfidf = weight_tfidf
        self.tie_break_margin = tie_break_margin

        # Per-task calibrated weights (set by calibrate())
        self.task_weights = {}  # task -> (bert_w, tfidf_w)

        self.pubmedbert_probs = {}
        self.tfidf_probs = {}
        self.oof_maps = None

    def load_oof_predictions(self, pubmedbert_path: str, tfidf_path: str):
        """Load out-of-fold probability predictions from cached classifier results."""
        with open(pubmedbert_path) as f:
            pubmedbert_data = json.load(f)
        with open(tfidf_path) as f:
            tfidf_data = json.load(f)

        for task in self.label_maps:
            if task.endswith("_id2label"):
                continue
            if task in pubmedbert_data and "oof_probabilities" in pubmedbert_data[task]:
                self.pubmedbert_probs[task] = np.array(
                    pubmedbert_data[task]["oof_probabilities"], dtype=np.float32
                )
            if task in tfidf_data and "oof_probabilities" in tfidf_data[task]:
                self.tfidf_probs[task] = np.array(
                    tfidf_data[task]["oof_probabilities"], dtype=np.float32
                )

        print("MetaEnsemble v4 loaded OOF predictions:")
        for task in self.label_maps:
            if task.endswith("_id2label"):
                continue
            bert_n = self.pubmedbert_probs[task].shape[0] if task in self.pubmedbert_probs else 0
            tfidf_n = self.tfidf_probs[task].shape[0] if task in self.tfidf_probs else 0
            print(f"  {task}: PubMedBERT={bert_n}, TF-IDF={tfidf_n}")

    def set_oof_maps(self, oof_maps: Dict[str, Dict[int, int]]):
        """Set OOF index maps (CSV row index → OOF array index per task)."""
        self.oof_maps = oof_maps

    def calibrate(self, df, label_maps):
        """
        Find optimal per-task bert/tfidf weights by sweeping on OOF predictions.

        This is legitimate since OOF predictions are out-of-fold: each patient
        was predicted by a model that did not see that patient during training.
        """
        print("\nCalibrating per-task ensemble weights...")
        tasks = [t for t in label_maps if not t.endswith("_id2label")]

        for task in tasks:
            if task not in self.pubmedbert_probs or task not in self.tfidf_probs:
                self.task_weights[task] = (self.default_weight_bert, self.default_weight_tfidf)
                continue

            label_col = f"{task}_label_id"
            if label_col not in df.columns:
                label_col = f"{task}_label"
            if label_col not in df.columns:
                self.task_weights[task] = (self.default_weight_bert, self.default_weight_tfidf)
                continue

            n_classes = len(label_maps[task])
            bert_arr = self.pubmedbert_probs[task]
            tfidf_arr = self.tfidf_probs[task]

            # Build valid patient list with correct OOF mapping
            valid_patients = []
            for csv_idx in range(len(df)):
                val = df.iloc[csv_idx].get(label_col)
                if not (hasattr(val, '__float__') or isinstance(val, (int, float))):
                    continue
                try:
                    label_val = float(val)
                except (ValueError, TypeError):
                    continue
                if np.isnan(label_val) or label_val < 0:
                    continue
                oof_idx = self._get_oof_idx(task, csv_idx)
                if oof_idx is None or oof_idx >= len(bert_arr) or oof_idx >= len(tfidf_arr):
                    continue
                valid_patients.append((csv_idx, oof_idx, int(label_val)))

            if len(valid_patients) < 10:
                self.task_weights[task] = (self.default_weight_bert, self.default_weight_tfidf)
                continue

            # Sweep bert_w from 0.0 to 1.0
            best_acc = 0.0
            best_w = self.default_weight_bert
            for w_bert_int in range(0, 21):  # 0.00, 0.05, ..., 1.00
                w_bert = w_bert_int / 20.0
                w_tfidf = 1.0 - w_bert
                correct = 0
                for _, oof_idx, label in valid_patients:
                    bp = self._normalize(bert_arr[oof_idx], n_classes)
                    tp = self._normalize(tfidf_arr[oof_idx], n_classes)
                    combined = w_bert * bp + w_tfidf * tp
                    if combined.argmax() == label:
                        correct += 1
                acc = correct / len(valid_patients)
                if acc > best_acc:
                    best_acc = acc
                    best_w = w_bert

            self.task_weights[task] = (best_w, 1.0 - best_w)
            print(f"  {task}: bert_w={best_w:.2f}, tfidf_w={1.0 - best_w:.2f} "
                  f"-> {best_acc:.1%} ({len(valid_patients)} patients)")

        print("Calibration complete.\n")

    def _get_oof_idx(self, task: str, patient_idx: int) -> Optional[int]:
        """Map CSV patient index to OOF array index."""
        if self.oof_maps and task in self.oof_maps:
            return self.oof_maps[task].get(patient_idx)
        return patient_idx

    def predict(
        self,
        task: str,
        patient_idx: int,
        agent_label: int,
        agent_text: str = "",
    ) -> int:
        """
        Calibrated discriminative prediction with LLM tie-breaking.
        """
        n_classes = len(self.label_maps[task])
        oof_idx = self._get_oof_idx(task, patient_idx)

        bert_probs = self._get_probs(self.pubmedbert_probs, task, oof_idx, n_classes)
        tfidf_probs = self._get_probs(self.tfidf_probs, task, oof_idx, n_classes)

        # Use task-specific calibrated weights
        w_bert, w_tfidf = self.task_weights.get(
            task, (self.default_weight_bert, self.default_weight_tfidf)
        )

        disc_probs = w_bert * bert_probs + w_tfidf * tfidf_probs
        total = disc_probs.sum()
        if total > 0:
            disc_probs /= total
        disc_pred = int(disc_probs.argmax())

        # LLM tie-breaking when discriminative margin is very small
        sorted_probs = np.sort(disc_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]

        if margin < self.tie_break_margin and 0 <= agent_label < n_classes:
            top2_indices = np.argsort(disc_probs)[-2:]
            if agent_label in top2_indices:
                return agent_label

        return disc_pred

    def predict_all_tasks(
        self,
        patient_idx: int,
        agent_labels: Dict[str, int],
        agent_text: str = "",
    ) -> Dict[str, int]:
        """Predict all tasks for a single patient."""
        return {
            task: self.predict(task, patient_idx, agent_labels.get(task, -1), agent_text)
            for task in self.label_maps
            if not task.endswith("_id2label")
        }

    @staticmethod
    def _normalize(p: np.ndarray, n_classes: int) -> np.ndarray:
        """Normalize probability vector, uniform fallback."""
        s = p.sum()
        if s > 0:
            return p / s
        return np.full(n_classes, 1.0 / n_classes)

    @staticmethod
    def _get_probs(
        probs_dict: dict,
        task: str,
        patient_idx: Optional[int],
        n_classes: int,
    ) -> np.ndarray:
        """Get probability array for a patient, with uniform fallback."""
        if patient_idx is not None and task in probs_dict and patient_idx < len(probs_dict[task]):
            p = probs_dict[task][patient_idx]
            if p.sum() > 0:
                return p / p.sum()
        return np.full(n_classes, 1.0 / n_classes)
