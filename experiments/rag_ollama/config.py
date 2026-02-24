"""
Configuration pour le RAG Ollama (LLM local).

Réutilise l'index FAISS et les embeddings OpenAI de RAG Basic.
Seul le LLM de génération change : GPT-3.5 → Ollama local.
"""

from dataclasses import dataclass, field
from pathlib import Path


# Modèles disponibles sur la VM
AVAILABLE_MODELS = {
    "mistral": "mistral:7b",
    "llama3": "llama3:8b",
}


@dataclass
class OllamaConfig:
    """Configuration du RAG Ollama."""

    # === Modèle Ollama ===
    model_name: str = "mistral:7b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1   # Identique RAG Basic pour équité
    max_tokens: int = 500       # Identique RAG Basic pour équité
    timeout_seconds: int = 120  # Timeout par requête
    num_retries: int = 2        # Retries sur timeout

    # === Embeddings OpenAI (retrieval inchangé) ===
    embedding_model: str = "text-embedding-3-small"

    # === Retrieval (identique RAG Basic) ===
    top_k: int = 5
    min_relevance_score: float = 0.3

    # === Coûts ===
    # Ollama est gratuit (local), seuls les embeddings coûtent
    embedding_cost_per_1k: float = 0.00002   # OpenAI text-embedding-3-small
    llm_input_cost_per_1k: float = 0.0       # Ollama = FREE
    llm_output_cost_per_1k: float = 0.0      # Ollama = FREE

    # === Chemins (réutiliser l'index FAISS de RAG Basic) ===
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    @property
    def vectorstore_dir(self) -> Path:
        """Réutilise l'index FAISS existant — PAS de ré-indexation !"""
        return self.project_root / "data" / "vectorstore" / "faiss"

    @property
    def index_path(self) -> Path:
        return self.vectorstore_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.vectorstore_dir / "metadata.json"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "experiments" / "rag_ollama" / "results"

    @property
    def generate_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"

    @property
    def tags_url(self) -> str:
        return f"{self.ollama_base_url}/api/tags"

    # === Prompt système IDENTIQUE à RAG Basic (équité de comparaison) ===
    system_prompt: str = """Tu es un assistant spécialisé dans les données ouvertes françaises sur la santé publique et la qualité de l'air.

Tu réponds aux questions en utilisant UNIQUEMENT les données fournies dans le contexte.

RÈGLES IMPORTANTES :
1. Base TOUJOURS tes réponses sur les données du contexte
2. Cite TOUJOURS les sources avec leurs dates
3. Si les données ne permettent pas de répondre, dis-le clairement
4. N'invente JAMAIS de chiffres ou de statistiques
5. Indique la fraîcheur des données (date la plus récente)

FORMAT DE RÉPONSE :
- Commence par répondre directement à la question
- Cite les chiffres avec leurs sources et dates
- Termine par une note sur la fraîcheur des données si pertinent"""

    def __post_init__(self):
        """Crée les répertoires nécessaires."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
