"""
OpenDataCopilot - RAG Base Interface
====================================

Interface abstraite (ABC) définissant le contrat commun pour toutes
les implémentations RAG du projet.

Permet de garantir la cohérence entre :
- Baseline (sans RAG)
- RAG Basic (FAISS)
- RAG Optimized (ChromaDB + reranking)
- RAG Specialized (fine-tuned embeddings)

Usage:
    from src.core.rag_interface import RAGInterface

    class MyRAG(RAGInterface):
        def retrieve(self, query: str) -> list[Document]:
            ...
        def generate(self, query: str, context: list[Document]) -> RAGResponse:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RAGType(str, Enum):
    """Types d'architecture RAG disponibles."""

    BASELINE = "baseline"
    RAG_BASIC = "rag_basic"
    RAG_OPTIMIZED = "rag_optimized"
    RAG_SPECIALIZED = "rag_specialized"


class Domain(str, Enum):
    """Domaines de données supportés."""

    SANTE = "sante"
    POLLUTION = "pollution"
    MIXED = "mixed"


@dataclass
class Document:
    """
    Représente un document ou chunk de contexte.

    Attributes:
        content: Contenu textuel du document
        metadata: Métadonnées associées (source, date, etc.)
        score: Score de similarité/pertinence (optionnel)
        doc_id: Identifiant unique du document
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    doc_id: str | None = None

    @property
    def source(self) -> str:
        """Retourne la source du document."""
        return self.metadata.get("source", "unknown")

    @property
    def domain(self) -> Domain:
        """Retourne le domaine du document."""
        domain_str = self.metadata.get("domain", "mixed")
        return Domain(domain_str)

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(source={self.source}, score={self.score}, content='{content_preview}')"


@dataclass
class Source:
    """
    Représente une source citée dans la réponse.

    Attributes:
        name: Nom de la source (ex: "Santé Publique France")
        url: URL de la source (optionnel)
        date: Date des données (optionnel)
        reliability: Score de fiabilité (0-1)
    """

    name: str
    url: str | None = None
    date: str | None = None
    reliability: float = 1.0

    def __repr__(self) -> str:
        return f"Source({self.name})"


@dataclass
class RAGResponse:
    """
    Réponse générée par le système RAG.

    Attributes:
        answer: Réponse textuelle générée
        sources: Liste des sources citées
        documents: Documents utilisés comme contexte
        confidence: Score de confiance (0-1)
        metadata: Métadonnées supplémentaires (tokens, latence, etc.)
    """

    answer: str
    sources: list[Source] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        """Retourne la latence en millisecondes."""
        return self.metadata.get("latency_ms", 0.0)

    @property
    def tokens_used(self) -> int:
        """Retourne le nombre de tokens utilisés."""
        return self.metadata.get("tokens_used", 0)

    @property
    def cost_usd(self) -> float:
        """Retourne le coût estimé en USD."""
        return self.metadata.get("cost_usd", 0.0)

    def has_sources(self) -> bool:
        """Vérifie si la réponse a des sources."""
        return len(self.sources) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convertit la réponse en dictionnaire."""
        return {
            "answer": self.answer,
            "sources": [
                {"name": s.name, "url": s.url, "date": s.date}
                for s in self.sources
            ],
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
        }


@dataclass
class RAGMetrics:
    """
    Métriques d'évaluation d'une réponse RAG.

    Utilisé pour les benchmarks comparatifs entre architectures.
    """

    # Retrieval metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain

    # Generation metrics
    rouge_l: float = 0.0
    bert_score: float = 0.0
    factual_consistency: float = 0.0

    # Hallucination metrics
    hallucination_rate: float = 0.0
    citation_accuracy: float = 0.0

    # Performance metrics
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Cost metrics
    total_cost_usd: float = 0.0
    cost_per_query_usd: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convertit les métriques en dictionnaire."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "rouge_l": self.rouge_l,
            "bert_score": self.bert_score,
            "factual_consistency": self.factual_consistency,
            "hallucination_rate": self.hallucination_rate,
            "citation_accuracy": self.citation_accuracy,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "tokens_per_second": self.tokens_per_second,
            "total_cost_usd": self.total_cost_usd,
            "cost_per_query_usd": self.cost_per_query_usd,
        }


class RAGInterface(ABC):
    """
    Interface abstraite pour toutes les implémentations RAG.

    Cette classe définit le contrat que toutes les architectures
    RAG doivent respecter pour permettre une comparaison équitable.

    Attributes:
        rag_type: Type d'architecture RAG
        name: Nom descriptif de l'implémentation
        version: Version de l'implémentation
    """

    def __init__(
        self,
        rag_type: RAGType,
        name: str,
        version: str = "1.0.0",
    ):
        self.rag_type = rag_type
        self.name = name
        self.version = version
        self._initialized = False
        self._created_at = datetime.now()

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialise les ressources nécessaires (modèles, index, etc.).

        Doit être appelé avant toute utilisation.
        Peut lever une exception si l'initialisation échoue.
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> list[Document]:
        """
        Récupère les documents pertinents pour une requête.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            domain: Filtrer par domaine (santé, pollution, ou les deux)

        Returns:
            Liste de documents triés par pertinence décroissante
        """
        pass

    @abstractmethod
    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """
        Génère une réponse à partir de la requête et du contexte.

        Args:
            query: Question de l'utilisateur
            context: Documents récupérés comme contexte

        Returns:
            Réponse structurée avec sources et métadonnées
        """
        pass

    def query(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> RAGResponse:
        """
        Pipeline complet : retrieve + generate.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            domain: Filtrer par domaine

        Returns:
            Réponse structurée avec sources et métadonnées
        """
        if not self._initialized:
            self.initialize()
            self._initialized = True

        # Récupération des documents
        documents = self.retrieve(query, top_k=top_k, domain=domain)

        # Génération de la réponse
        response = self.generate(query, documents)

        return response

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        domain: Domain,
    ) -> int:
        """
        Ajoute des documents à l'index vectoriel.

        Args:
            documents: Documents à indexer
            domain: Domaine des documents

        Returns:
            Nombre de documents ajoutés
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """Retourne les informations sur l'implémentation."""
        return {
            "rag_type": self.rag_type.value,
            "name": self.name,
            "version": self.version,
            "initialized": self._initialized,
            "created_at": self._created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.rag_type.value}, name='{self.name}')"


class RAGFactory:
    """
    Factory pour créer des instances RAG.

    Usage:
        factory = RAGFactory()
        factory.register("baseline", BaselineRAG)
        rag = factory.create("baseline")
    """

    _registry: dict[str, type[RAGInterface]] = {}

    @classmethod
    def register(cls, rag_type: str, rag_class: type[RAGInterface]) -> None:
        """Enregistre une classe RAG."""
        cls._registry[rag_type] = rag_class

    @classmethod
    def create(cls, rag_type: str, **kwargs: Any) -> RAGInterface:
        """
        Crée une instance RAG.

        Args:
            rag_type: Type de RAG (baseline, rag_basic, etc.)
            **kwargs: Arguments passés au constructeur

        Returns:
            Instance de RAGInterface
        """
        if rag_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"RAG type '{rag_type}' inconnu. Disponibles: {available}"
            )

        return cls._registry[rag_type](**kwargs)

    @classmethod
    def list_types(cls) -> list[str]:
        """Liste les types RAG enregistrés."""
        return list(cls._registry.keys())
