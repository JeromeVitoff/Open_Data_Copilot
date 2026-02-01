"""
OpenDataCopilot - Configuration Settings
=========================================

Gestion centralisée de la configuration via Pydantic Settings.
Les variables sont chargées depuis les variables d'environnement et/ou .env

Usage:
    from src.config.settings import settings
    print(settings.OPENAI_API_KEY)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Chemin racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Configuration principale de l'application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────────
    # LLM Provider - OpenAI
    # ─────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(
        ...,
        description="Clé API OpenAI (obligatoire)",
    )
    OPENAI_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="Modèle OpenAI pour le chat",
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="Modèle OpenAI pour les embeddings",
    )

    # ─────────────────────────────────────────────────────────────────
    # Data APIs
    # ─────────────────────────────────────────────────────────────────
    AIRPARIF_API_KEY: str = Field(
        default="",
        description="Clé API Airparif (optionnel)",
    )
    OPENAQ_API_KEY: str = Field(
        default="",
        description="Clé API OpenAQ (optionnel)",
    )
    COHERE_API_KEY: str = Field(
        default="",
        description="Clé API Cohere pour reranking (optionnel)",
    )

    # ─────────────────────────────────────────────────────────────────
    # Application Settings
    # ─────────────────────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environnement d'exécution",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Niveau de logging",
    )
    API_HOST: str = Field(
        default="0.0.0.0",
        description="Host de l'API FastAPI",
    )
    API_PORT: int = Field(
        default=8000,
        description="Port de l'API FastAPI",
    )
    STREAMLIT_PORT: int = Field(
        default=8501,
        description="Port de l'interface Streamlit",
    )

    # ─────────────────────────────────────────────────────────────────
    # Database Settings
    # ─────────────────────────────────────────────────────────────────
    SANTE_DB_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "sante.db",
        description="Chemin vers la base SQLite santé",
    )
    POLLUTION_DB_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "pollution.db",
        description="Chemin vers la base SQLite pollution",
    )
    METADATA_DB_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "metadata.db",
        description="Chemin vers la base SQLite métadonnées",
    )

    # ─────────────────────────────────────────────────────────────────
    # Vector Store Settings
    # ─────────────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "vectorstore" / "faiss",
        description="Chemin vers l'index FAISS",
    )
    CHROMA_PERSIST_PATH: Path = Field(
        default=PROJECT_ROOT / "data" / "vectorstore" / "chroma",
        description="Chemin vers la base ChromaDB",
    )

    # ─────────────────────────────────────────────────────────────────
    # Cache Settings
    # ─────────────────────────────────────────────────────────────────
    ENABLE_CACHE: bool = Field(
        default=True,
        description="Activer le cache des réponses API",
    )
    CACHE_DIR: Path = Field(
        default=PROJECT_ROOT / ".cache",
        description="Répertoire de cache",
    )
    CACHE_TTL: int = Field(
        default=3600,
        description="Durée de vie du cache en secondes",
    )

    # ─────────────────────────────────────────────────────────────────
    # RAG Settings
    # ─────────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Taille des chunks pour le text splitting",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Chevauchement entre les chunks",
    )
    TOP_K: int = Field(
        default=5,
        description="Nombre de documents à récupérer",
    )

    # ─────────────────────────────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────────────────────────────
    MAX_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        description="Nombre max de requêtes par minute",
    )
    API_CALL_DELAY: float = Field(
        default=0.5,
        description="Délai entre les appels API (secondes)",
    )

    # ─────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────
    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        if not v or v == "sk-your-openai-api-key-here":
            raise ValueError(
                "OPENAI_API_KEY doit être configuré dans .env"
            )
        return v

    @field_validator("CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100 or v > 10000:
            raise ValueError("CHUNK_SIZE doit être entre 100 et 10000")
        return v

    @field_validator("TOP_K")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("TOP_K doit être entre 1 et 20")
        return v

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────
    @property
    def is_development(self) -> bool:
        """Vérifie si on est en mode développement."""
        return self.ENVIRONMENT == "development"

    @property
    def has_airparif_key(self) -> bool:
        """Vérifie si la clé Airparif est configurée."""
        return bool(self.AIRPARIF_API_KEY)

    @property
    def has_openaq_key(self) -> bool:
        """Vérifie si la clé OpenAQ est configurée."""
        return bool(self.OPENAQ_API_KEY)

    @property
    def has_cohere_key(self) -> bool:
        """Vérifie si la clé Cohere est configurée."""
        return bool(self.COHERE_API_KEY)


@lru_cache
def get_settings() -> Settings:
    """
    Retourne une instance singleton des settings.

    Usage:
        from src.config.settings import get_settings
        settings = get_settings()
    """
    return Settings()


# Instance globale pour import direct
settings = get_settings()
