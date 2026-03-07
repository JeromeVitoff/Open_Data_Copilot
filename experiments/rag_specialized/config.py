"""Configuration pour le RAG Spécialisé Multi-Domaines."""

from dataclasses import dataclass
from pathlib import Path

from experiments.rag_optimized.config import RAGOptimizedConfig


@dataclass
class RAGSpecializedConfig(RAGOptimizedConfig):
    """Configuration du RAG spécialisé multi-domaines."""

    # Expansion terminologique
    use_llm_expansion: bool = False   # True = GPT-3.5 génère variantes (lent)
    num_query_variants: int = 2       # Nb variantes LLM si activé

    # Scoring domaine
    domain_score_weight: float = 0.15  # Poids du boost domaine sur score final
    domain_diversity_bonus: float = 0.10  # Bonus si doc couvre santé + pollution

    # Filtrage source
    strict_domain_filter: bool = False  # False = filtre soft (boost), True = hard

    @property
    def results_dir(self) -> Path:
        return self.project_root / "experiments" / "rag_specialized" / "results"

    def __post_init__(self):
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
