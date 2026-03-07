"""
Configuration pour le RAG Optimisé.

Étend la configuration du RAG Basique avec :
- Paramètres BM25 (retrieval sparse)
- Paramètres du reranker CrossEncoder
- Pipeline hybride : FAISS + BM25 → CrossEncoder
"""

from dataclasses import dataclass, field
from pathlib import Path

from experiments.rag_basic.config import RAGBasicConfig


@dataclass
class RAGOptimizedConfig(RAGBasicConfig):
    """Configuration du RAG optimisé (hybride + reranking)."""

    # === Pipeline optimisé ===
    # Nombre de candidats récupérés avant reranking
    retrieval_top_k: int = 20
    # Nombre final de documents après reranking
    rerank_top_k: int = 5

    # === Retrieval hybride FAISS + BM25 ===
    # Alpha = poids du score dense (FAISS)
    # (1-alpha) = poids du score sparse (BM25)
    hybrid_alpha: float = 0.6

    # === BM25 ===
    bm25_k1: float = 1.5   # Saturation du terme TF
    bm25_b: float = 0.75   # Normalisation par longueur de document
    bm25_cache_path: str = ""  # Chemin pour persister l'index BM25 (optionnel)

    # === Reranker CrossEncoder ===
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_length: int = 512
    reranker_batch_size: int = 32

    # === Modèle LLM (upgrade vs basic) ===
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 600  # Légèrement plus pour réponses plus complètes

    @property
    def results_dir(self) -> Path:
        return self.project_root / "experiments" / "rag_optimized" / "results"

    def __post_init__(self):
        """Crée les répertoires nécessaires."""
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
