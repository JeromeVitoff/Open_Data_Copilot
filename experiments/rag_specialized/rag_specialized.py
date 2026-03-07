"""
OpenDataCopilot - RAG Spécialisé Multi-Domaines
=================================================

Étend le RAG Optimisé avec :
1. Détection automatique du domaine (santé / pollution / corrélation)
2. Expansion terminologique multi-domaines avant retrieval
3. Filtrage et scoring de documents par domaine/source
4. Prompts spécialisés adaptés au domaine détecté

Pipeline complet :
    Query
      → DomainDetector → domaine
      → QueryExpander (terminologie) → query enrichie
      → HybridRetriever (FAISS+BM25) top-20
      → ContextualFilter (filtre + scoring domaine)
      → CrossEncoder rerank → top-5
      → GPT-3.5 + prompt spécialisé → RAGResponse
"""

import hashlib
import math
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
from experiments.rag_optimized.hybrid_retriever import HybridRetriever
from experiments.rag_optimized.reranker import SemanticReranker
from experiments.rag_specialized.config import RAGSpecializedConfig
from experiments.rag_specialized.domain_detector import DomainDetector
from experiments.rag_specialized.domain_embeddings import expand_query
from experiments.rag_specialized.contextual_filter import filter_and_score_documents
from experiments.rag_specialized.specialized_prompts import (
    get_system_prompt,
    format_user_prompt,
)


class SpecializedRAG(RAGInterface):
    """
    RAG Spécialisé Multi-Domaines.

    Réutilise l'infrastructure du RAG Optimisé (FAISS + BM25 + CrossEncoder)
    et y ajoute une couche de spécialisation par domaine.
    """

    def __init__(self, config: RAGSpecializedConfig | None = None):
        super().__init__(
            rag_type=RAGType.RAG_SPECIALIZED,
            name="RAG Spécialisé Multi-Domaines (Santé + Pollution)",
            version="1.0.0",
        )

        self.config = config or RAGSpecializedConfig()
        self.retriever = HybridRetriever(self.config)
        self.reranker = SemanticReranker(self.config)
        self.domain_detector = DomainDetector()
        self.client = None

        # Cache
        self._cache: dict[str, RAGResponse] = {}
        self._cache_hits = 0

        # Métriques
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_filter_time = 0.0
        self._total_rerank_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0
        self._domain_counts: dict[str, int] = {
            "health": 0, "environment": 0, "correlation": 0, "general": 0
        }

    def initialize(self) -> None:
        """Initialise client OpenAI, HybridRetriever et CrossEncoder."""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY non définie dans .env")
            self.client = OpenAI(api_key=api_key)
            logger.info("✅ Client OpenAI initialisé")
        except ImportError:
            raise ImportError("openai non installé: pip install openai")

        self.retriever.initialize(self.client)
        self.reranker.initialize()

        self._initialized = True
        logger.info("✅ SpecializedRAG initialisé (FAISS + BM25 + CrossEncoder + DomainDetector)")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> list[Document]:
        """
        Pipeline retrieval spécialisé :
        1. Détection domaine
        2. Expansion terminologique
        3. HybridRetriever (top-20)
        4. Filtrage + scoring domaine
        5. CrossEncoder rerank (top-5)
        """
        if not self._initialized:
            self.initialize()

        # --- 1. Détection domaine ---
        detected = self.domain_detector.detect(query)
        detected_domain = detected.domain
        self._domain_counts[detected_domain] = (
            self._domain_counts.get(detected_domain, 0) + 1
        )
        logger.debug(
            f"Domaine détecté: {detected_domain} "
            f"(santé={detected.health_score}, env={detected.env_score})"
        )

        # --- 2. Expansion terminologique ---
        expanded_query = expand_query(query, detected_domain)
        if expanded_query != query:
            logger.debug(f"Query étendue: +{len(expanded_query)-len(query)} chars")

        # --- 3. Retrieval hybride sur query enrichie ---
        start_retrieval = time.time()
        candidates = self.retriever.retrieve(
            expanded_query,
            top_k=self.config.retrieval_top_k,
            domain=domain,  # Filtre RAG interface (si fourni)
        )
        retrieval_time = time.time() - start_retrieval
        self._total_retrieval_time += retrieval_time
        logger.debug(f"Retrieval hybride: {len(candidates)} candidats en {retrieval_time*1000:.0f}ms")

        # --- 4. Filtrage + scoring domaine ---
        start_filter = time.time()
        scored_docs = filter_and_score_documents(
            candidates,
            domain=detected_domain,
            domain_score_weight=self.config.domain_score_weight,
            diversity_bonus=self.config.domain_diversity_bonus,
            strict=self.config.strict_domain_filter,
        )
        # Trier par score (domain-boosted) décroissant
        scored_docs.sort(key=lambda d: d.score or 0.0, reverse=True)
        filter_time = time.time() - start_filter
        self._total_filter_time += filter_time

        # Prendre top-retrieval_top_k pour le reranking
        candidates_for_rerank = scored_docs[:self.config.retrieval_top_k]

        # --- 5. CrossEncoder rerank ---
        start_rerank = time.time()
        reranked = self.reranker.rerank(
            query,  # Reranker sur la query ORIGINALE (pas étendue)
            candidates_for_rerank,
            top_k=top_k,
        )
        rerank_time = time.time() - start_rerank
        self._total_rerank_time += rerank_time
        logger.debug(f"Reranking: {len(reranked)} docs en {rerank_time*1000:.0f}ms")

        # Annoter les documents avec le domaine détecté
        for doc in reranked:
            doc.metadata["_detected_domain"] = detected_domain

        return reranked

    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """
        Génère avec prompt spécialisé selon le domaine détecté.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Récupérer le domaine annoté par retrieve()
        detected_domain = "general"
        if context:
            detected_domain = context[0].metadata.get("_detected_domain", "general")

        # Construire le contexte formaté
        if context:
            context_parts = []
            for i, doc in enumerate(context, 1):
                source = doc.metadata.get("source", "unknown")
                date = doc.metadata.get("date", "date inconnue")
                domain_meta = doc.metadata.get("domain", "")
                context_parts.append(
                    f"[{i}] {doc.content}\n"
                    f"    (Source: {source} | Date: {date} | Domaine: {domain_meta})"
                )
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "Aucun document pertinent trouvé dans la base de données."

        # Prompt spécialisé selon domaine
        system_prompt = get_system_prompt(detected_domain)
        user_prompt = format_user_prompt(query, context_text, detected_domain)

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
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

            embedding_cost = 0.00002
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

            if context:
                avg_score = sum(
                    doc.score for doc in context if doc.score is not None
                ) / len(context)
                confidence = min(0.95, 1 / (1 + math.exp(-avg_score / 3)))
            else:
                confidence = 0.0

            return RAGResponse(
                answer=answer,
                sources=sources,
                documents=context,
                confidence=confidence,
                metadata={
                    "latency_ms": generation_time * 1000,
                    "retrieval_time_ms": self._total_retrieval_time * 1000 / max(1, self._query_count),
                    "filter_time_ms": self._total_filter_time * 1000 / max(1, self._query_count),
                    "rerank_time_ms": self._total_rerank_time * 1000 / max(1, self._query_count),
                    "tokens_used": total_tokens,
                    "cost_usd": total_cost,
                    "model": self.config.llm_model,
                    "num_docs_retrieved": len(context),
                    "avg_relevance_score": confidence,
                    "detected_domain": detected_domain,
                    "pipeline": "domain_aware_hybrid + crossencoder",
                },
            )

        except Exception as e:
            generation_time = time.time() - start_time
            return RAGResponse(
                answer=f"Erreur lors de la génération: {str(e)}",
                sources=[],
                documents=[],
                confidence=0.0,
                metadata={"latency_ms": generation_time * 1000, "error": str(e)},
            )

    def query(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
        use_cache: bool = True,
    ) -> RAGResponse:
        """Pipeline complet avec cache."""
        if not self._initialized:
            self.initialize()

        if use_cache:
            cache_key = hashlib.md5(f"{query}|{top_k}|{domain}".encode()).hexdigest()
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]

        context = self.retrieve(query, top_k=top_k, domain=domain)
        response = self.generate(query, context)

        if use_cache:
            self._cache[cache_key] = response

        return response

    def add_documents(self, documents: list[Document], domain: Domain) -> int:
        logger.warning("add_documents() non implémenté. Utilisez data_indexer.py.")
        return 0

    def get_stats(self) -> dict[str, Any]:
        index_size = self.retriever.index.ntotal if self.retriever.index else 0
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "domain_distribution": self._domain_counts,
            "avg_retrieval_time_ms": self._total_retrieval_time * 1000 / max(1, self._query_count),
            "avg_filter_time_ms": self._total_filter_time * 1000 / max(1, self._query_count),
            "avg_rerank_time_ms": self._total_rerank_time * 1000 / max(1, self._query_count),
            "avg_generation_time_ms": self._total_generation_time * 1000 / max(1, self._query_count),
            "model": self.config.llm_model,
            "reranker_model": self.config.reranker_model,
            "hybrid_alpha": self.config.hybrid_alpha,
            "retrieval_top_k": self.config.retrieval_top_k,
            "rerank_top_k": self.config.rerank_top_k,
            "domain_score_weight": self.config.domain_score_weight,
            "index_size": index_size,
        }
