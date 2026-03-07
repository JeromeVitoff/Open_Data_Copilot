"""
Reranker sémantique avec CrossEncoder.

Utilise le modèle cross-encoder/ms-marco-MiniLM-L-6-v2 pour
reclasser les candidats issus du retrieval hybride.

Le CrossEncoder prédit directement la pertinence d'une paire
(question, document) en lisant les deux textes en simultané —
plus précis mais plus lent qu'un bi-encoder.

Pipeline :
- Entrée  : top-20 candidats (FAISS + BM25 fusionnés)
- Sortie  : top-5 documents reclassés par pertinence sémantique
"""

import time
from loguru import logger

from src.core.rag_interface import Document
from experiments.rag_optimized.config import RAGOptimizedConfig


class SemanticReranker:
    """
    Reranker CrossEncoder pour la sélection finale des documents.

    Attributes:
        config: Configuration du RAG optimisé
        model: CrossEncoder chargé
    """

    def __init__(self, config: RAGOptimizedConfig):
        self.config = config
        self.model = None

    def initialize(self) -> None:
        """Charge le modèle CrossEncoder."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers non installé: pip install sentence-transformers"
            )

        logger.info(f"⚙️  Chargement du CrossEncoder: {self.config.reranker_model}")
        start = time.time()
        self.model = CrossEncoder(
            self.config.reranker_model,
            max_length=self.config.reranker_max_length,
        )
        elapsed = time.time() - start
        logger.info(f"✅ CrossEncoder chargé en {elapsed:.1f}s")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Reclasse les documents par pertinence sémantique.

        Args:
            query: Requête utilisateur
            documents: Candidats issus du retrieval hybride
            top_k: Nombre de documents à retourner (défaut: config.rerank_top_k)

        Returns:
            Documents reclassés, limités à top_k
        """
        if not documents:
            return []

        if top_k is None:
            top_k = self.config.rerank_top_k

        if self.model is None:
            self.initialize()

        # Construire les paires (query, doc_content)
        pairs = [(query, doc.content) for doc in documents]

        # Prédire les scores de pertinence
        scores = self.model.predict(
            pairs,
            batch_size=self.config.reranker_batch_size,
            show_progress_bar=False,
        )

        # Associer les scores aux documents
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Retourner top_k avec le score CrossEncoder
        result = []
        for score, doc in scored_docs[:top_k]:
            reranked_doc = Document(
                content=doc.content,
                metadata=doc.metadata,
                score=float(score),  # Score CrossEncoder (logit)
                doc_id=doc.doc_id,
            )
            result.append(reranked_doc)

        return result
