"""
OpenDataCopilot - RAG Optimisé
================================

Architecture RAG avancée combinant :
1. Retrieval hybride : FAISS (dense) + BM25 (sparse), alpha=0.6
2. Reranking sémantique : CrossEncoder ms-marco-MiniLM-L-6-v2
3. Génération : GPT-3.5-turbo avec contexte enrichi

Pipeline complet :
    Query → Embedding → FAISS top-20 + BM25 top-20
         → Fusion pondérée → top-20 candidats
         → CrossEncoder rerank → top-5
         → GPT-3.5 generate → RAGResponse
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.rag_interface import (
    RAGInterface,
    RAGType,
    RAGResponse,
    Document,
    Source,
    Domain,
)
from experiments.rag_optimized.config import RAGOptimizedConfig
from experiments.rag_optimized.hybrid_retriever import HybridRetriever
from experiments.rag_optimized.reranker import SemanticReranker


class OptimizedRAG(RAGInterface):
    """
    RAG Optimisé : FAISS + BM25 + CrossEncoder + GPT-3.5.

    Attributes:
        config: Configuration du RAG optimisé
        retriever: HybridRetriever (FAISS + BM25)
        reranker: SemanticReranker (CrossEncoder)
        client: Client OpenAI
    """

    def __init__(self, config: RAGOptimizedConfig | None = None):
        super().__init__(
            rag_type=RAGType.RAG_OPTIMIZED,
            name="RAG Optimisé (FAISS + BM25 + CrossEncoder)",
            version="1.0.0",
        )

        self.config = config or RAGOptimizedConfig()
        self.retriever = HybridRetriever(self.config)
        self.reranker = SemanticReranker(self.config)
        self.client = None

        # Cache LRU pour éviter de retraiter les questions déjà vues
        self._cache: dict[str, RAGResponse] = {}
        self._cache_hits = 0

        # Métriques
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_rerank_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0

    def initialize(self) -> None:
        """
        Initialise tous les composants :
        - Client OpenAI
        - HybridRetriever (FAISS + BM25)
        - SemanticReranker (CrossEncoder)

        Raises:
            ValueError: Si OPENAI_API_KEY manquante
            FileNotFoundError: Si l'index FAISS est absent
        """
        # Client OpenAI
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY non définie dans .env")
            self.client = OpenAI(api_key=api_key)
            logger.info("✅ Client OpenAI initialisé")
        except ImportError:
            raise ImportError("openai non installé: pip install openai")

        # Hybrid Retriever (FAISS + BM25)
        self.retriever.initialize(self.client)

        # Semantic Reranker (CrossEncoder)
        self.reranker.initialize()

        self._initialized = True
        logger.info("✅ OptimizedRAG initialisé (FAISS + BM25 + CrossEncoder)")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> list[Document]:
        """
        Retrieval hybride + reranking sémantique.

        Étape 1 : HybridRetriever → top-20 candidats (FAISS + BM25 fusionnés)
        Étape 2 : SemanticReranker → top-5 documents reclassés

        Args:
            query: Question de l'utilisateur
            top_k: Nombre final de documents (après reranking)
            domain: Filtrer par domaine

        Returns:
            Documents reclassés par pertinence sémantique
        """
        if not self._initialized:
            self.initialize()

        start_total = time.time()

        # Étape 1 : Retrieval hybride → top retrieval_top_k candidats
        start_retrieval = time.time()
        candidates = self.retriever.retrieve(
            query,
            top_k=self.config.retrieval_top_k,
            domain=domain,
        )
        retrieval_time = time.time() - start_retrieval
        self._total_retrieval_time += retrieval_time

        logger.debug(f"Retrieval hybride: {len(candidates)} candidats en {retrieval_time*1000:.0f}ms")

        # Étape 2 : Reranking CrossEncoder → top rerank_top_k
        start_rerank = time.time()
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        rerank_time = time.time() - start_rerank
        self._total_rerank_time += rerank_time

        logger.debug(f"Reranking: {len(reranked)} docs sélectionnés en {rerank_time*1000:.0f}ms")

        return reranked

    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """
        Génère une réponse avec GPT-3.5 basée sur le contexte reranké.

        Args:
            query: Question de l'utilisateur
            context: Documents reranqués (top-5)

        Returns:
            Réponse structurée avec sources et métriques
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Construire le contexte enrichi
        if context:
            context_parts = []
            for i, doc in enumerate(context, 1):
                source = doc.metadata.get("source", "unknown")
                date = doc.metadata.get("date", "date inconnue")
                domain = doc.metadata.get("domain", "")
                context_parts.append(
                    f"[{i}] {doc.content}\n    (Source: {source} | Date: {date} | Domaine: {domain})"
                )
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "Aucun document pertinent trouvé dans la base de données."

        user_prompt = f"""Contexte (sources officielles françaises) :
{context_text}

Question : {query}

Réponds en citant précisément les sources et les dates. Si les données ne permettent pas de répondre précisément, dis-le clairement."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            generation_time = time.time() - start_time
            self._total_generation_time += generation_time

            answer = response.choices[0].message.content or ""
            usage = response.usage

            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = input_tokens + output_tokens

            embedding_cost = 0.00002  # Approximation embedding requête
            llm_cost = (
                (input_tokens / 1000) * self.config.llm_input_cost_per_1k
                + (output_tokens / 1000) * self.config.llm_output_cost_per_1k
            )
            total_cost = embedding_cost + llm_cost

            self._total_tokens += total_tokens
            self._total_cost += total_cost
            self._query_count += 1

            sources = [
                Source(
                    name=doc.metadata.get("source", "unknown"),
                    url="",
                    date=doc.metadata.get("date", ""),
                )
                for doc in context
            ]

            # Confiance basée sur les scores CrossEncoder (sigmoïde approximative)
            import math
            if context:
                avg_score = sum(doc.score for doc in context if doc.score is not None) / len(context)
                # CrossEncoder scores sont des logits → sigmoïde pour [0, 1]
                confidence = min(0.95, 1 / (1 + math.exp(-avg_score / 3)))
            else:
                confidence = 0.0

            avg_retrieval_t = (
                self._total_retrieval_time * 1000 / max(1, self._query_count)
            )
            avg_rerank_t = (
                self._total_rerank_time * 1000 / max(1, self._query_count)
            )

            return RAGResponse(
                answer=answer,
                sources=sources,
                documents=context,
                confidence=confidence,
                metadata={
                    "latency_ms": generation_time * 1000,
                    "retrieval_time_ms": avg_retrieval_t,
                    "rerank_time_ms": avg_rerank_t,
                    "tokens_used": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": total_cost,
                    "model": self.config.llm_model,
                    "num_docs_retrieved": len(context),
                    "avg_relevance_score": confidence,
                    "has_context": len(context) > 0,
                    "pipeline": "hybrid_faiss_bm25 + crossencoder",
                },
            )

        except Exception as e:
            generation_time = time.time() - start_time
            return RAGResponse(
                answer=f"Erreur lors de la génération: {str(e)}",
                sources=[],
                documents=[],
                confidence=0.0,
                metadata={
                    "latency_ms": generation_time * 1000,
                    "error": str(e),
                },
            )

    def query(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
        use_cache: bool = True,
    ) -> RAGResponse:
        """
        Pipeline RAG optimisé complet avec cache.

        Pour les questions déjà vues, retourne la réponse en cache (<1ms).

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents après reranking
            domain: Filtrer par domaine
            use_cache: Activer le cache (défaut: True)

        Returns:
            Réponse complète avec sources et métriques
        """
        if not self._initialized:
            self.initialize()

        # Cache lookup
        if use_cache:
            cache_key = hashlib.md5(f"{query}|{top_k}|{domain}".encode()).hexdigest()
            if cache_key in self._cache:
                self._cache_hits += 1
                logger.debug(f"✅ Cache hit ({self._cache_hits} total)")
                return self._cache[cache_key]

        context = self.retrieve(query, top_k=top_k, domain=domain)
        response = self.generate(query, context)

        # Stocker en cache
        if use_cache:
            self._cache[cache_key] = response

        return response

    def add_documents(
        self,
        documents: list[Document],
        domain: Domain,
    ) -> int:
        """Non implémenté : utiliser data_indexer.py pour reconstruire l'index."""
        logger.warning(
            "add_documents() non implémenté pour OptimizedRAG. "
            "Utilisez experiments/rag_basic/data_indexer.py pour reconstruire l'index FAISS."
        )
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques d'utilisation."""
        index_size = self.retriever.index.ntotal if self.retriever.index else 0
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "avg_retrieval_time_ms": (
                self._total_retrieval_time * 1000 / max(1, self._query_count)
            ),
            "avg_rerank_time_ms": (
                self._total_rerank_time * 1000 / max(1, self._query_count)
            ),
            "avg_generation_time_ms": (
                self._total_generation_time * 1000 / max(1, self._query_count)
            ),
            "model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "reranker_model": self.config.reranker_model,
            "hybrid_alpha": self.config.hybrid_alpha,
            "retrieval_top_k": self.config.retrieval_top_k,
            "rerank_top_k": self.config.rerank_top_k,
            "index_size": index_size,
        }

    def reset_stats(self) -> None:
        """Réinitialise les statistiques."""
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_rerank_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0
        self._cache_hits = 0

    def clear_cache(self) -> None:
        """Vide le cache des réponses."""
        self._cache.clear()
        self._cache_hits = 0
