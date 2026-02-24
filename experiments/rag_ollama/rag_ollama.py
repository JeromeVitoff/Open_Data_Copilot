"""
OpenDataCopilot - RAG avec Ollama
===================================

Implémentation RAG qui réutilise EXACTEMENT le retrieval de BasicRAG
(index FAISS + embeddings OpenAI) et remplace uniquement le LLM de
génération par Ollama local (Mistral 7B ou Llama3 8B).

Cela garantit une comparaison équitable :
- Retrieval identique → différence = LLM uniquement
- Même prompt système → même format de réponse attendu
- Même dataset → mêmes 50 questions
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
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
from experiments.rag_ollama.config import OllamaConfig


class OllamaRAG(RAGInterface):
    """
    RAG Ollama : retrieval FAISS/OpenAI + génération Ollama local.

    Le retrieval est IDENTIQUE à BasicRAG (même FAISS, mêmes embeddings).
    Seule la génération change : GPT-3.5 → Ollama (Mistral/Llama3).

    Attributes:
        config: Configuration Ollama
        index: Index FAISS (partagé avec BasicRAG)
        documents: Documents indexés (partagés avec BasicRAG)
        openai_client: Client OpenAI pour les embeddings uniquement
    """

    def __init__(self, config: OllamaConfig | None = None):
        """
        Args:
            config: Configuration Ollama (modèle, URL, params)
        """
        super().__init__(
            rag_type=RAGType.RAG_BASIC,   # Même type de retrieval
            name=f"RAG Ollama ({config.model_name if config else 'mistral:7b'})",
            version="1.0.0",
        )

        self.config = config or OllamaConfig()
        self.index = None
        self.documents: list[dict] = []
        self.openai_client = None   # Embeddings seulement

        # Métriques
        self._total_output_tokens = 0
        self._total_input_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0

    # ─────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Charge l'index FAISS et initialise les clients."""
        # 1. Vérifier Ollama
        self._check_ollama()

        # 2. Charger l'index FAISS (partagé avec BasicRAG)
        if not self.config.index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS non trouvé: {self.config.index_path}\n"
                "Lancez d'abord: python -m experiments.rag_basic.data_indexer"
            )

        try:
            import faiss
            self.index = faiss.read_index(str(self.config.index_path))
            logger.info(f"✅ Index FAISS chargé: {self.index.ntotal:,} vecteurs")
        except ImportError:
            raise ImportError("faiss non installé: pip install faiss-cpu")

        # 3. Charger les métadonnées
        if not self.config.metadata_path.exists():
            raise FileNotFoundError(
                f"Métadonnées non trouvées: {self.config.metadata_path}"
            )

        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.documents = metadata["documents"]
            logger.info(f"✅ {len(self.documents):,} documents chargés")

        # 4. Initialiser OpenAI pour les embeddings
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY non définie dans .env")
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("✅ Client OpenAI initialisé (embeddings uniquement)")
        except ImportError:
            raise ImportError("openai non installé: pip install openai")

        self._initialized = True
        logger.info(f"✅ OllamaRAG initialisé avec {self.config.model_name}")

    def _check_ollama(self) -> None:
        """Vérifie qu'Ollama est disponible et que le modèle est chargé."""
        try:
            resp = requests.get(self.config.tags_url, timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.config.model_name not in models:
                available = ", ".join(models)
                raise ValueError(
                    f"Modèle '{self.config.model_name}' non disponible dans Ollama.\n"
                    f"Disponibles: {available}\n"
                    f"Installez avec: ollama pull {self.config.model_name}"
                )
            logger.info(f"✅ Ollama OK — modèle {self.config.model_name} disponible")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Ollama non joignable sur {self.config.ollama_base_url}\n"
                "Démarrez Ollama avec: ollama serve"
            )

    # ─────────────────────────────────────────────────────────
    # Retrieval — IDENTIQUE à BasicRAG
    # ─────────────────────────────────────────────────────────

    def _get_embedding(self, text: str) -> np.ndarray:
        """Génère l'embedding OpenAI d'un texte (identique à BasicRAG)."""
        response = self.openai_client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        import faiss
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> list[Document]:
        """
        Retrieval FAISS identique à BasicRAG.
        Copie exacte du code pour garantir la comparabilité.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        query_embedding = self._get_embedding(query)

        search_k = top_k * 3 if domain else top_k
        scores, indices = self.index.search(query_embedding.reshape(1, -1), search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            if domain:
                doc_domain = doc["metadata"].get("domain", "")
                domain_str = domain.value if hasattr(domain, "value") else str(domain)
                if doc_domain != domain_str:
                    continue

            if score < self.config.min_relevance_score:
                continue

            results.append(Document(
                content=doc["text"],
                metadata=doc["metadata"],
                score=float(score),
            ))

            if len(results) >= top_k:
                break

        retrieval_time = time.time() - start_time
        self._total_retrieval_time += retrieval_time

        return results

    # ─────────────────────────────────────────────────────────
    # Génération — Ollama local
    # ─────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> dict:
        """
        Appelle l'API Ollama pour générer une réponse.

        Args:
            prompt: Prompt complet (système + contexte + question)

        Returns:
            Dict avec 'response', 'eval_count', 'prompt_eval_count'
        """
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        for attempt in range(self.config.num_retries + 1):
            try:
                resp = requests.post(
                    self.config.generate_url,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.Timeout:
                if attempt < self.config.num_retries:
                    logger.warning(
                        f"Timeout Ollama (tentative {attempt + 1}/"
                        f"{self.config.num_retries + 1}), retry..."
                    )
                    time.sleep(2)
                else:
                    raise TimeoutError(
                        f"Ollama timeout après {self.config.timeout_seconds}s"
                    )
            except requests.RequestException as e:
                raise ConnectionError(f"Erreur Ollama: {e}")

    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """
        Génère une réponse avec Ollama au lieu de GPT-3.5.

        Le prompt est IDENTIQUE à BasicRAG pour garantir l'équité
        de comparaison. Seul le LLM change.
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Construire le contexte — IDENTIQUE à BasicRAG
        if context:
            context_parts = []
            for i, doc in enumerate(context, 1):
                source = doc.metadata.get("source", "unknown")
                date = doc.metadata.get("date", "date inconnue")
                context_parts.append(f"[{i}] {doc.content} (Date: {date})")
            context_text = "\n\n".join(context_parts)
        else:
            context_text = "Aucun document pertinent trouvé dans la base de données."

        # Prompt utilisateur — IDENTIQUE à BasicRAG
        user_prompt = (
            f"Contexte (sources officielles) :\n{context_text}\n\n"
            f"Question : {query}\n\n"
            "Réponds en citant les sources et les dates. "
            "Si les données ne permettent pas de répondre précisément, dis-le clairement."
        )

        # Ollama n'a pas de séparation système/utilisateur explicite dans
        # /api/generate, on préfixe avec le prompt système
        full_prompt = (
            f"[SYSTEM]\n{self.config.system_prompt}\n\n"
            f"[USER]\n{user_prompt}\n\n"
            "[ASSISTANT]\n"
        )

        try:
            result = self._call_ollama(full_prompt)

            generation_time = time.time() - start_time
            self._total_generation_time += generation_time

            answer = result.get("response", "").strip()
            output_tokens = result.get("eval_count", 0)
            input_tokens = result.get("prompt_eval_count", 0)
            total_tokens = input_tokens + output_tokens

            # Coût : embeddings OpenAI seulement (Ollama est gratuit)
            embedding_cost = 0.00002
            total_cost = embedding_cost  # LLM Ollama = $0.00

            self._total_output_tokens += output_tokens
            self._total_input_tokens += input_tokens
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

            avg_score = (
                sum(doc.score for doc in context) / len(context) if context else 0.0
            )

            return RAGResponse(
                answer=answer,
                sources=sources,
                documents=context,
                confidence=min(0.95, avg_score),
                metadata={
                    "latency_ms": generation_time * 1000,
                    "retrieval_time_ms": (
                        self._total_retrieval_time * 1000 / max(1, self._query_count)
                    ),
                    "tokens_used": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": total_cost,
                    "model": self.config.model_name,
                    "num_docs_retrieved": len(context),
                    "avg_relevance_score": avg_score,
                    "has_context": len(context) > 0,
                    "ollama_tokens_per_second": (
                        output_tokens / generation_time if generation_time > 0 else 0
                    ),
                },
            )

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Erreur génération Ollama: {e}")
            return RAGResponse(
                answer=f"Erreur Ollama ({self.config.model_name}): {str(e)}",
                sources=[],
                documents=[],
                confidence=0.0,
                metadata={
                    "latency_ms": generation_time * 1000,
                    "error": str(e),
                    "model": self.config.model_name,
                },
            )

    def query(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> RAGResponse:
        """Pipeline complet : retrieve (OpenAI) + generate (Ollama)."""
        if not self._initialized:
            self.initialize()

        context = self.retrieve(query, top_k=top_k, domain=domain)
        response = self.generate(query, context)
        return response

    def add_documents(self, documents: list[Document], domain: Domain) -> int:
        logger.warning("add_documents() non implémenté pour OllamaRAG.")
        return 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "model": self.config.model_name,
            "embedding_model": self.config.embedding_model,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": self._total_cost,
            "total_queries": self._query_count,
            "avg_retrieval_time_ms": (
                self._total_retrieval_time * 1000 / max(1, self._query_count)
            ),
            "avg_generation_time_ms": (
                self._total_generation_time * 1000 / max(1, self._query_count)
            ),
            "index_size": self.index.ntotal if self.index else 0,
        }
