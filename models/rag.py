"""
ILAE Guideline RAG Knowledge Base.

Provides retrieval-augmented generation for the orchestrator agent,
ensuring diagnoses follow ILAE 2017/2025 classification criteria and
established presurgical evaluation protocols.

Uses sentence-transformers for embedding and FAISS for vector retrieval.
"""

import json
import os
from typing import List, Optional, Tuple

import torch
import numpy as np


# ILAE classification knowledge — embedded directly for self-contained deployment
ILAE_KNOWLEDGE = [
    # --- Epilepsy Type Classification (ILAE 2017) ---
    {
        "section": "epilepsy_type_classification",
        "title": "ILAE 2017 Classification of Epilepsy Types",
        "content": (
            "The ILAE 2017 classification defines four epilepsy types: "
            "1) Focal Epilepsy: seizures originating within networks limited to one hemisphere. "
            "May be unifocal or multifocal. Includes temporal lobe epilepsy (TLE), frontal lobe "
            "epilepsy (FLE), parietal lobe epilepsy, occipital lobe epilepsy, and insular epilepsy. "
            "2) Generalized Epilepsy: seizures arising from and rapidly engaging bilaterally "
            "distributed networks. Includes genetic generalized epilepsies (JME, JAE, CAE, GTCS alone). "
            "3) Combined Generalized and Focal Epilepsy: patients have both generalized and focal "
            "seizure types. Examples include Dravet syndrome and Lennox-Gastaut syndrome. "
            "4) Unknown Epilepsy: when there is insufficient information to classify."
        ),
    },
    # --- Seizure Type Classification (ILAE 2017) ---
    {
        "section": "seizure_type_classification",
        "title": "ILAE 2017 Operational Classification of Seizure Types",
        "content": (
            "Focal seizures: classified by awareness (aware vs impaired awareness), by motor onset "
            "(automatisms, atonic, clonic, epileptic spasms, hyperkinetic, myoclonic, tonic) or "
            "non-motor onset (autonomic, behavior arrest, cognitive, emotional, sensory). "
            "Focal to bilateral tonic-clonic: focal onset propagating to bilateral tonic-clonic. "
            "Generalized seizures: motor (tonic-clonic, clonic, tonic, myoclonic, myoclonic-tonic-clonic, "
            "myoclonic-atonic, atonic, epileptic spasms) or non-motor/absence (typical, atypical, "
            "myoclonic, eyelid myoclonia). "
            "Unknown onset: may be motor (tonic-clonic, epileptic spasms) or non-motor (behavior arrest)."
        ),
    },
    # --- Epileptogenic Zone Localization ---
    {
        "section": "ez_localization",
        "title": "Epileptogenic Zone Localization Principles",
        "content": (
            "The epileptogenic zone (EZ) is the minimum cortical area that must be resected to "
            "produce seizure freedom. It is inferred from convergence of multiple data sources: "
            "1) Seizure semiology: provides lateralizing and localizing signs (e.g., unilateral "
            "automatisms suggest ipsilateral temporal, hyperkinetic movements suggest frontal). "
            "2) Scalp EEG: interictal epileptiform discharges provide lateralization; ictal EEG "
            "onset provides localization (temporal: rhythmic theta/alpha, frontal: bilateral onset). "
            "3) MRI: structural lesions (FCD, hippocampal sclerosis, tumors) are the strongest "
            "predictor of surgical outcome. Concordance between MRI lesion and EEG focus is critical. "
            "4) Additional: PET (hypometabolism), SPECT (ictal hyperperfusion), MEG source analysis. "
            "Concordance across modalities increases confidence in EZ localization."
        ),
    },
    # --- MRI Findings in Epilepsy ---
    {
        "section": "mri_epilepsy_findings",
        "title": "MRI Findings in Epilepsy",
        "content": (
            "Key MRI findings: "
            "Hippocampal sclerosis (HS): volume loss and T2/FLAIR hyperintensity of hippocampus, "
            "most common finding in temporal lobe epilepsy, associated with good surgical outcome. "
            "Focal cortical dysplasia (FCD): cortical thickening, blurring of gray-white junction, "
            "transmantle sign on FLAIR. Type I: subtle, often MRI-negative. Type II: more conspicuous. "
            "Type III: associated with other lesions. "
            "Tumors: low-grade tumors (DNET, ganglioglioma) commonly cause epilepsy. "
            "Vascular malformations: cavernomas with hemosiderin ring on T2*. "
            "Tuberous sclerosis: cortical tubers, subependymal nodules. "
            "Heterotopia: periventricular or band heterotopia. "
            "MRI-negative epilepsy: 20-40% of surgical candidates, requires advanced post-processing."
        ),
    },
    # --- EEG Interpretation for Epilepsy ---
    {
        "section": "eeg_epilepsy_interpretation",
        "title": "EEG Interpretation in Epilepsy Diagnosis",
        "content": (
            "Background activity: normal posterior dominant rhythm (PDR) is 8-13 Hz alpha rhythm. "
            "Slowing (theta 4-7 Hz, delta 1-3 Hz) suggests cortical dysfunction. "
            "Focal slowing: localizing, suggests structural lesion. "
            "Interictal epileptiform discharges (IEDs): spikes (<70ms), sharp waves (70-200ms), "
            "spike-and-wave complexes. Location suggests EZ. Bilateral independent temporal IEDs "
            "suggest bitemporal epilepsy. 3 Hz spike-wave: typical absence. 4-6 Hz polyspike-wave: JME. "
            "Ictal patterns: rhythmic activity evolving in frequency, morphology, and distribution. "
            "Temporal onset: rhythmic theta/alpha. Frontal: lower voltage fast or bilateral. "
            "EEG-fMRI: BOLD correlates of IEDs for network analysis. "
            "HD-EEG: source localization with 256 channels improves spatial resolution."
        ),
    },
    # --- Presurgical Evaluation ---
    {
        "section": "presurgical_evaluation",
        "title": "Presurgical Evaluation Protocol for Epilepsy Surgery",
        "content": (
            "Phase I (non-invasive): "
            "1) Comprehensive seizure history and semiology analysis. "
            "2) Scalp video-EEG monitoring: capture habitual seizures, identify interictal/ictal patterns. "
            "3) High-resolution MRI (3T preferred): epilepsy protocol including 3D T1, 3D FLAIR, "
            "T2, T2*, coronal oblique sequences. "
            "4) Neuropsychological assessment: cognitive baseline, lateralization of language/memory. "
            "5) FDG-PET: interictal hypometabolism concordant with EZ. "
            "6) Ictal SPECT: ictal hyperperfusion. "
            "Phase II (invasive, if Phase I non-concordant): "
            "7) Intracranial EEG (SEEG or subdural grids): direct cortical recording. "
            "8) Cortical stimulation: functional mapping. "
            "Decision: surgery offered when EZ is well-localized, concordant across modalities, "
            "and does not overlap eloquent cortex."
        ),
    },
    # --- Concordance Assessment ---
    {
        "section": "concordance_assessment",
        "title": "Multimodal Concordance Assessment",
        "content": (
            "Concordance: agreement between independent diagnostic modalities on EZ localization. "
            "Concordant: MRI lesion, EEG focus, and semiology all point to same region/hemisphere. "
            "Associated with Engel Class I outcome (seizure-free) in >70% of cases. "
            "Partially concordant: 2 of 3 modalities agree, or agreement on hemisphere but not lobe. "
            "May still proceed to surgery with additional workup. "
            "Discordant: modalities point to different regions. Requires Phase II evaluation "
            "(intracranial EEG). Lower surgical success rates. "
            "Key concordance patterns: "
            "- Temporal lobe: hippocampal sclerosis (MRI) + temporal IEDs (EEG) + deja vu/automatisms (semiology) = highly concordant. "
            "- Frontal lobe: FCD (MRI) + frontal IEDs + hyperkinetic seizures = concordant. "
            "- MRI-negative + clear EEG focus: partially concordant, may need advanced imaging."
        ),
    },
    # --- Drug Response Assessment ---
    {
        "section": "aed_response",
        "title": "Anti-Epileptic Drug Response Classification",
        "content": (
            "Drug-responsive: seizure-free on current AED regimen (first or second adequate trial). "
            "Drug-resistant (ILAE 2010 definition): failure of adequate trials of two tolerated, "
            "appropriately chosen and used AED schedules (mono or combination) to achieve sustained "
            "seizure freedom. Affects approximately 30% of epilepsy patients. "
            "Drug-resistant patients should be referred for presurgical evaluation. "
            "Temporal lobe epilepsy with hippocampal sclerosis: high drug resistance rate (60-70%), "
            "excellent surgical outcomes (60-80% seizure-free). "
            "Lesional epilepsy generally has better surgical outcomes than non-lesional."
        ),
    },
    # --- Surgical Outcome (Engel Classification) ---
    {
        "section": "surgical_outcome",
        "title": "Engel Surgical Outcome Classification",
        "content": (
            "Engel Class I (Free of disabling seizures): "
            "Ia - completely seizure-free since surgery. "
            "Ib - non-disabling simple partial seizures only. "
            "Ic - some disabling seizures after surgery but free for ≥2 years. "
            "Id - generalized convulsions with AED withdrawal only. "
            "Engel Class II (Rare disabling seizures): almost seizure-free. "
            "Engel Class III (Worthwhile improvement): ≥90% seizure reduction. "
            "Engel Class IV (No worthwhile improvement): <90% reduction. "
            "Predictors of Engel I: MRI-visible lesion, concordant EZ localization, "
            "complete resection, temporal lobe location, hippocampal sclerosis pathology."
        ),
    },
    # --- ILAE 2025 Updates ---
    {
        "section": "ilae_2025_updates",
        "title": "ILAE 2025 Classification Updates",
        "content": (
            "Key updates in ILAE 2025 seizure classification: "
            "1) Epileptic spasms reclassified as potentially focal, generalized, or unknown onset. "
            "2) Enhanced emphasis on epilepsy syndromes: neonatal-infantile (self-limited familial "
            "neonatal epilepsy, etc.), childhood (CAE, JAE, JME, childhood occipital), and "
            "variable age (genetic generalized, focal). "
            "3) Recognition of developmental and/or epileptic encephalopathies (DEE): "
            "Dravet syndrome, Lennox-Gastaut, West syndrome, etc. "
            "4) Etiology given greater prominence: structural, genetic, infectious, metabolic, "
            "immune, unknown — with recognition that multiple etiologies may coexist. "
            "5) Comorbidities integrated into classification framework."
        ),
    },
]


class ILAEKnowledgeBase:
    """
    ILAE guideline retrieval system using sentence embeddings + cosine similarity.

    For MICCAI submission: uses built-in knowledge chunks.
    Can be extended with external documents via add_documents().
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Knowledge store
        self.chunks: List[dict] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.encoder = None

    def _load_encoder(self):
        """Lazy load sentence encoder."""
        if self.encoder is not None:
            return

        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(
            self.embedding_model_name, device=self.device
        )

    def build_index(self, additional_docs: Optional[List[dict]] = None):
        """
        Build retrieval index from ILAE knowledge + optional documents.

        Args:
            additional_docs: List of {"title": str, "content": str} dicts.
        """
        self._load_encoder()

        self.chunks = list(ILAE_KNOWLEDGE)
        if additional_docs:
            self.chunks.extend(additional_docs)

        texts = [
            f"{chunk['title']}\n{chunk['content']}"
            for chunk in self.chunks
        ]

        embeddings = self.encoder.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        self.embeddings = embeddings.to(self.device)
        print(f"ILAE knowledge base: {len(self.chunks)} chunks indexed")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        Retrieve relevant knowledge chunks for a query.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of (chunk_dict, similarity_score) tuples.
        """
        assert self.embeddings is not None, "Call build_index() first"
        self._load_encoder()

        query_emb = self.encoder.encode(
            [query], convert_to_tensor=True
        ).to(self.device)

        similarities = torch.nn.functional.cosine_similarity(
            query_emb, self.embeddings
        )

        top_k = min(top_k, len(self.chunks))
        scores, indices = similarities.topk(top_k)

        results = []
        for score, idx in zip(scores.cpu().tolist(), indices.cpu().tolist()):
            results.append((self.chunks[idx], score))

        return results

    def retrieve_formatted(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve and format as context string for LLM prompt."""
        results = self.retrieve(query, top_k)

        if not results:
            return "No relevant clinical guidelines found."

        lines = ["=== ILAE Clinical Guidelines ===\n"]
        for chunk, score in results:
            lines.append(f"### {chunk['title']} (relevance: {score:.3f})")
            lines.append(chunk['content'])
            lines.append("")

        return "\n".join(lines)

    def save_index(self, path: str):
        """Save embeddings to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "chunks": self.chunks,
            "embeddings": self.embeddings.cpu(),
        }, path)

    def load_index(self, path: str):
        """Load pre-computed embeddings."""
        data = torch.load(path, weights_only=False)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"].to(self.device)
