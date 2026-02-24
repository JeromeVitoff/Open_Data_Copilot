"""
Configuration pour le RAG Basique avec FAISS.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGBasicConfig:
    """Configuration du RAG basique."""

    # === Modèles OpenAI ===
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 500

    # === Retrieval ===
    top_k: int = 5
    min_relevance_score: float = 0.3

    # === Chunking ===
    # Pour le RAG basique : 1 ligne = 1 document
    chunk_strategy: str = "row"
    max_chunk_size: int = 512  # tokens max par chunk

    # === Coûts OpenAI (USD) ===
    # text-embedding-3-small: $0.00002 / 1K tokens
    embedding_cost_per_1k: float = 0.00002
    # gpt-3.5-turbo: $0.0015 / 1K input, $0.002 / 1K output
    llm_input_cost_per_1k: float = 0.0015
    llm_output_cost_per_1k: float = 0.002

    # === Chemins ===
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def sante_dir(self) -> Path:
        return self.data_dir / "sante"

    @property
    def sante_odisse_dir(self) -> Path:
        return self.data_dir / "sante_odisse"

    @property
    def pollution_dir(self) -> Path:
        return self.data_dir / "pollution"

    @property
    def airparif_hist_dir(self) -> Path:
        return self.data_dir / "pollution_airparif_hist"

    @property
    def vectorstore_dir(self) -> Path:
        return self.project_root / "data" / "vectorstore" / "faiss"

    @property
    def index_path(self) -> Path:
        return self.vectorstore_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.vectorstore_dir / "metadata.json"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "experiments" / "rag_basic" / "results"

    # === Fichiers de données ===
    data_files: dict = field(default_factory=lambda: {
        "sante": [
            "covid_hospitalisations.csv",
            "covid_tests_dep.csv",
            "professionnels_sante_dep.csv",
            "sursaud_urgences.csv",
        ],
        "pollution": [
            "airparif_indices.csv",
            "openaq_paris_latest.csv",
            "openaq_lyon_latest.csv",
            "openaq_marseille_latest.csv",
        ],
    })

    # === Prompt système pour le RAG ===
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
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
