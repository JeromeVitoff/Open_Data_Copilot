"""
Configuration pour le RAG Spécialisé v2 (embeddings médicaux + LLM médical).
"""

from dataclasses import dataclass, field
from pathlib import Path

from experiments.rag_optimized.config import RAGOptimizedConfig


@dataclass
class RAGSpecializedV2Config(RAGOptimizedConfig):
    """Configuration du RAG Spécialisé v2 avec modèles médicaux."""

    # === Embeddings médicaux (CamemBERT-bio) ===
    medical_embedding_model: str = "almanach/camembert-bio-base"
    medical_embedding_dim: int = 768       # CamemBERT-bio dimension
    embedding_batch_size: int = 32         # Batch GPU pour indexation
    normalize_embeddings: bool = True      # Normalisation cosine

    # === Index FAISS médical (séparé de l'index OpenAI) ===
    medical_index_name: str = "index_medical.faiss"
    medical_metadata_name: str = "metadata_medical.json"

    # === LLM médical (BioMistral-7B) ===
    use_biomistral: bool = False           # False = GPT fallback, True = BioMistral local
    biomistral_model: str = "BioMistral/BioMistral-7B"
    biomistral_max_new_tokens: int = 512
    biomistral_temperature: float = 0.7

    # === Pipeline (hérité RAGOptimizedConfig) ===
    # retrieval_top_k = 20 (candidats avant reranking)
    # rerank_top_k = 5 (après CrossEncoder)
    # hybrid_alpha = 0.6

    @property
    def medical_vectorstore_dir(self) -> Path:
        return self.project_root / "data" / "vectorstore" / "faiss_medical"

    @property
    def medical_index_path(self) -> Path:
        return self.medical_vectorstore_dir / self.medical_index_name

    @property
    def medical_metadata_path(self) -> Path:
        return self.medical_vectorstore_dir / self.medical_metadata_name

    @property
    def results_dir(self) -> Path:
        return self.project_root / "experiments" / "rag_specialized_v2" / "results"

    def __post_init__(self):
        """Crée les répertoires nécessaires."""
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.medical_vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
