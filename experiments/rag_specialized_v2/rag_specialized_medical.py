"""
RAG Spécialisé Médical v2
==========================

Combine :
- CamemBERT-bio (almanach/camembert-bio-base) pour l'embedding des requêtes
  → Index FAISS médical (faiss_medical/) re-indexé avec embeddings 768-dim
- BM25 existant (partagé avec RAG Optimisé) pour la recherche sparse
- CrossEncoder (ms-marco-MiniLM-L-6-v2) pour le reranking sémantique
- GPT-3.5-turbo (ou BioMistral-7B optionnel) pour la génération

Pipeline :
    Query → MedicalEmbeddings.embed_query()
          → FAISS (faiss_medical) + BM25 en parallèle (top-20 chacun)
          → Fusion hybride alpha=0.6
          → CrossEncoder rerank → top-5
          → GPT-3.5 / BioMistral generate → RAGResponse
"""

import concurrent.futures
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.rag_interface import (
    Domain,
    Document,
    RAGInterface,
    RAGResponse,
    RAGType,
    Source,
)
from experiments.rag_optimized.reranker import SemanticReranker
from experiments.rag_specialized_v2.config import RAGSpecializedV2Config
from experiments.rag_specialized_v2.medical_embeddings import MedicalEmbeddings


class MedicalRetriever:
    """
    Retrieval hybride avec embeddings CamemBERT-bio (médical).

    - Index FAISS médical : vecteurs 768-dim, re-indexés avec CamemBERT-bio
    - BM25 : partagé avec RAG Optimisé (même corpus, même pickle)
    - Fusion pondérée : alpha * dense + (1-alpha) * sparse
    """

    def __init__(self, config: RAGSpecializedV2Config):
        self.config = config
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []
        self.bm25 = None
        self.medical_emb: MedicalEmbeddings | None = None

    @property
    def _bm25_path(self) -> Path:
        # Réutilise le pickle BM25 de RAG Optimisé (même corpus)
        return self.config.metadata_path.parent / "bm25_index.pkl"

    def initialize(self) -> None:
        """Charge l'index FAISS médical, les documents et l'index BM25."""

        # --- FAISS médical ---
        if not self.config.medical_index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS médical absent : {self.config.medical_index_path}\n"
                "Lancez d'abord : python -m experiments.rag_specialized_v2.data_indexer_medical"
            )
        self.index = faiss.read_index(str(self.config.medical_index_path))
        logger.info(f"Index FAISS medical charge : {self.index.ntotal:,} vecteurs (dim={self.index.d})")

        # --- Documents (depuis l'index médical) ---
        if not self.config.medical_metadata_path.exists():
            raise FileNotFoundError(
                f"Métadonnées médicales absentes : {self.config.medical_metadata_path}"
            )
        with open(self.config.medical_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.documents = meta["documents"]
        logger.info(f"{len(self.documents):,} documents charges")

        # --- BM25 ---
        if self._bm25_path.exists():
            logger.info(f"Chargement BM25 depuis {self._bm25_path}...")
            t0 = time.time()
            with open(self._bm25_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            logger.info(f"BM25 charge en {time.time()-t0:.1f}s ({data['num_docs']:,} docs)")
        else:
            logger.warning("Index BM25 absent. Construction à la volée (lente)...")
            self._build_bm25()

        # --- Embeddings médicaux ---
        self.medical_emb = MedicalEmbeddings(
            model_name=self.config.medical_embedding_model,
            normalize=self.config.normalize_embeddings,
        )

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)

    def _get_embedding(self, query: str) -> np.ndarray:
        """Encode la requête avec CamemBERT-bio (pas d'API OpenAI)."""
        return self.medical_emb.embed_query(query).astype(np.float32)

    def _faiss_search(self, query_emb: np.ndarray, top_k: int) -> dict[int, float]:
        """FAISS inner product search → scores [0, 1]."""
        scores, indices = self.index.search(query_emb.reshape(1, -1), top_k)
        result = {}
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.documents):
                result[int(idx)] = float((score + 1.0) / 2.0)
        return result

    def _bm25_search(self, query: str, top_k: int) -> dict[int, float]:
        """BM25 sparse search → scores normalisés [0, 1]."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[-top_k:][::-1]
        max_score = float(scores[top_idx[0]]) if len(top_idx) > 0 else 1.0
        if max_score <= 0:
            max_score = 1.0
        return {int(i): float(scores[i]) / max_score for i in top_idx if scores[i] > 0}

    def retrieve(self, query: str, top_k: int = 20) -> list[Document]:
        """
        Retrieval hybride parallèle (FAISS médical + BM25).

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de candidats à retourner avant reranking

        Returns:
            Liste de Documents triés par score fusionné décroissant
        """
        search_k = top_k * 2

        # Paralléliser FAISS + BM25
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            def dense_search():
                emb = self._get_embedding(query)
                return self._faiss_search(emb, search_k)

            def sparse_search():
                return self._bm25_search(query, search_k)

            f_dense = executor.submit(dense_search)
            f_sparse = executor.submit(sparse_search)
            dense_scores = f_dense.result()
            sparse_scores = f_sparse.result()

        # Fusion pondérée
        alpha = self.config.hybrid_alpha
        all_idx = set(dense_scores) | set(sparse_scores)
        fused: dict[int, float] = {}
        for idx in all_idx:
            d = dense_scores.get(idx, 0.0)
            s = sparse_scores.get(idx, 0.0)
            fused[idx] = alpha * d + (1.0 - alpha) * s

        # Trier et construire la liste de Documents
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[Document] = []
        for idx, score in ranked:
            doc = self.documents[idx]
            results.append(Document(
                content=doc["text"],
                metadata=doc.get("metadata", {}),
                score=score,
                doc_id=str(idx),
            ))
        return results


class SpecializedMedicalRAG(RAGInterface):
    """
    RAG Spécialisé v2 — Embeddings médicaux CamemBERT-bio + GPT-3.5.

    Utilise un index FAISS re-indexé avec CamemBERT-bio (768-dim)
    pour un meilleur retrieval sur la terminologie médicale française.
    """

    def __init__(self, config: RAGSpecializedV2Config | None = None):
        super().__init__(
            rag_type=RAGType.RAG_SPECIALIZED,
            name="RAG Spécialisé Médical v2 (CamemBERT-bio + CrossEncoder)",
            version="2.0.0",
        )
        self.config = config or RAGSpecializedV2Config()
        self.retriever = MedicalRetriever(self.config)
        self.reranker = SemanticReranker(self.config)
        self.client = None

        self._cache: dict[str, RAGResponse] = {}
        self._cache_hits = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_rerank_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0

    def initialize(self) -> None:
        """Initialise le client OpenAI, le MedicalRetriever et le SemanticReranker."""
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY non définie dans .env")
        self.client = OpenAI(api_key=api_key)
        logger.info("Client OpenAI initialise")

        self.retriever.initialize()
        self.reranker.initialize()

        self._initialized = True
        logger.info("SpecializedMedicalRAG initialise (CamemBERT-bio + BM25 + CrossEncoder)")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> list[Document]:
        """Retrieval hybride médical + reranking CrossEncoder."""
        if not self._initialized:
            self.initialize()

        t0 = time.time()
        candidates = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
        self._total_retrieval_time += time.time() - t0

        t1 = time.time()
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        self._total_rerank_time += time.time() - t1

        return reranked

    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """Génère une réponse GPT-3.5 basée sur les documents médicaux."""
        if not self._initialized:
            self.initialize()

        t0 = time.time()

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

        user_prompt = (
            f"Contexte (sources officielles françaises) :\n{context_text}\n\n"
            f"Question : {query}\n\n"
            "Réponds en citant précisément les sources et les dates. "
            "Si les données ne permettent pas de répondre précisément, dis-le clairement."
        )

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

            gen_time = time.time() - t0
            self._total_generation_time += gen_time

            answer = response.choices[0].message.content or ""
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            cost = (in_tok / 1000) * self.config.llm_input_cost_per_1k + (
                out_tok / 1000
            ) * self.config.llm_output_cost_per_1k

            self._total_tokens += in_tok + out_tok
            self._total_cost += cost
            self._query_count += 1

            sources = [
                Source(
                    name=doc.metadata.get("source", "unknown"),
                    date=doc.metadata.get("date", ""),
                )
                for doc in context
            ]

            import math
            confidence = 0.0
            if context:
                avg_score = sum(d.score for d in context if d.score is not None) / len(context)
                confidence = min(0.95, 1 / (1 + math.exp(-avg_score / 3)))

            return RAGResponse(
                answer=answer,
                sources=sources,
                documents=context,
                confidence=confidence,
                metadata={
                    "latency_ms": gen_time * 1000,
                    "retrieval_time_ms": self._total_retrieval_time * 1000 / max(1, self._query_count),
                    "rerank_time_ms": self._total_rerank_time * 1000 / max(1, self._query_count),
                    "tokens_used": in_tok + out_tok,
                    "cost_usd": cost,
                    "model": self.config.llm_model,
                    "embedding_model": self.config.medical_embedding_model,
                    "num_docs_retrieved": len(context),
                    "avg_relevance_score": confidence,
                    "pipeline": "camembert-bio + bm25 + crossencoder",
                },
            )

        except Exception as e:
            gen_time = time.time() - t0
            return RAGResponse(
                answer=f"Erreur lors de la génération: {e}",
                sources=[],
                documents=[],
                confidence=0.0,
                metadata={"latency_ms": gen_time * 1000, "error": str(e)},
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
        logger.warning("add_documents() non implémenté — utilisez data_indexer_medical.py")
        return 0

    def get_stats(self) -> dict[str, Any]:
        idx_size = self.retriever.index.ntotal if self.retriever.index else 0
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "avg_retrieval_time_ms": self._total_retrieval_time * 1000 / max(1, self._query_count),
            "avg_rerank_time_ms": self._total_rerank_time * 1000 / max(1, self._query_count),
            "avg_generation_time_ms": self._total_generation_time * 1000 / max(1, self._query_count),
            "model": self.config.llm_model,
            "embedding_model": self.config.medical_embedding_model,
            "reranker_model": self.config.reranker_model,
            "hybrid_alpha": self.config.hybrid_alpha,
            "retrieval_top_k": self.config.retrieval_top_k,
            "rerank_top_k": self.config.rerank_top_k,
            "index_size": idx_size,
        }
