"""
Few-Shot Retriever: PubMedBERT embedding-based patient retrieval for in-context learning.

Uses PubMedBERT [CLS] embeddings + FAISS to find similar training patients,
then formats them as few-shot examples for the orchestrator prompt.

This transforms the orchestrator from zero-shot to few-shot classification.
"""

import json
import os
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


class FewShotRetriever:
    """FAISS-based patient retriever using PubMedBERT [CLS] embeddings."""

    def __init__(
        self,
        embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda:0",
        max_length: int = 512,
    ):
        self.embedding_model_name = embedding_model
        self.device = device
        self.max_length = max_length

        self.tokenizer = None
        self.model = None
        self.index = None
        self.embeddings = None
        self.texts = None
        self.patient_labels = None  # {task: np.array of label_ids}
        self.label_maps = None
        self.df = None

    def _load_model(self):
        """Load PubMedBERT for embedding extraction."""
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
        self.model.eval()

    def _unload_model(self):
        """Free GPU memory after index is built."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to [CLS] embeddings using PubMedBERT."""
        self._load_model()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            outputs = self.model(**encodings)
            cls_embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
            all_embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        return embeddings

    def build_index(
        self,
        texts: List[str],
        df: pd.DataFrame,
        label_maps: Dict[str, Dict[str, int]],
    ):
        """
        Build FAISS index from patient texts.

        Args:
            texts: List of concatenated patient texts (SEMIOLOGY + MRI + EEG).
            df: Patient DataFrame with label columns.
            label_maps: {task: {label_str: int}} mapping.
        """
        print(f"Building few-shot index for {len(texts)} patients...")

        self.texts = texts
        self.label_maps = label_maps
        self.df = df

        # Extract ground-truth labels for each task
        self.patient_labels = {}
        for task in label_maps:
            id_col = f"{task}_label_id"
            if id_col in df.columns:
                self.patient_labels[task] = df[id_col].values.astype(float)
            else:
                label_col = f"{task}_label"
                labels = np.full(len(df), -1.0)
                if label_col in df.columns:
                    for i, val in enumerate(df[label_col]):
                        if pd.notna(val) and str(val).strip() in label_maps[task]:
                            labels[i] = label_maps[task][str(val).strip()]
                self.patient_labels[task] = labels

        # Encode all patients
        self.embeddings = self._encode_texts(texts)
        self._unload_model()

        # Build FAISS index (inner product = cosine similarity on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve_similar(
        self,
        query_idx: int,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve top-K most similar patients, excluding the query patient (leave-one-out).

        Args:
            query_idx: Index of the query patient in the DataFrame.
            top_k: Number of similar patients to retrieve.

        Returns:
            List of dicts with text_summary, labels, similarity, patient_id.
        """
        assert self.index is not None, "Call build_index() first"

        query_vec = self.embeddings[query_idx : query_idx + 1]

        # Retrieve top_k + 1 to account for self-match
        scores, indices = self.index.search(query_vec, top_k + 1)

        results = []
        inv_maps = {
            task: {v: k for k, v in lm.items()}
            for task, lm in self.label_maps.items()
        }

        for score, idx in zip(scores[0], indices[0]):
            if idx == query_idx:
                continue  # skip self

            # Build label dict for this patient
            labels = {}
            for task in self.label_maps:
                label_id = int(self.patient_labels[task][idx])
                if label_id >= 0:
                    labels[task] = inv_maps[task].get(label_id, f"class_{label_id}")
                else:
                    labels[task] = "N/A"

            # Truncate text for prompt (keep it concise)
            text_summary = self.texts[idx][:600]  # ~150 words

            patient_id = ""
            if self.df is not None and "patient_id" in self.df.columns:
                patient_id = self.df.iloc[idx]["patient_id"]

            results.append({
                "patient_id": patient_id,
                "text_summary": text_summary,
                "labels": labels,
                "similarity": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def format_few_shot_examples(self, examples: List[Dict]) -> str:
        """
        Format retrieved examples as JSON output demonstrations.

        Shows the model what correct JSON output looks like for similar patients.
        This teaches the output format AND calibrates the classification decisions.
        """
        if not examples:
            return ""

        parts = []
        for i, ex in enumerate(examples, 1):
            # Build JSON object for the correct answer
            json_obj = {}
            for task, label in ex["labels"].items():
                if label != "N/A":
                    json_obj[task] = label

            if not json_obj:
                continue

            # Truncate clinical text for prompt efficiency
            clinical_text = ex["text_summary"][:400]

            parts.append(
                f"Case {i}: {clinical_text}\n"
                f"Correct output:\n"
                f"```json\n{json.dumps(json_obj, indent=2)}\n```"
            )

        return "\n\n".join(parts)

    def save_index(self, save_dir: str):
        """Save FAISS index and metadata for reuse."""
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "few_shot.index"))
        np.save(os.path.join(save_dir, "few_shot_embeddings.npy"), self.embeddings)
        print(f"  Few-shot index saved to {save_dir}")

    def load_index(self, save_dir: str):
        """Load pre-built FAISS index."""
        self.index = faiss.read_index(os.path.join(save_dir, "few_shot.index"))
        self.embeddings = np.load(os.path.join(save_dir, "few_shot_embeddings.npy"))
        print(f"  Few-shot index loaded from {save_dir} ({self.index.ntotal} vectors)")
