"""
OpenDataCopilot - RAG Basique avec FAISS
=========================================

Implémentation d'un système RAG simple qui utilise FAISS pour
le retrieval et GPT-3.5 pour la génération.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from loguru import logger

# Charger les variables d'environnement
load_dotenv()

# Import du projet
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
from experiments.rag_basic.config import RAGBasicConfig


class BasicRAG(RAGInterface):
    """
    RAG Basique avec FAISS et OpenAI.

    Utilise un index FAISS pour le retrieval dense et GPT-3.5
    pour la génération de réponses contextualisées.

    Attributes:
        config: Configuration du RAG
        index: Index FAISS
        documents: Liste des documents indexés
        client: Client OpenAI
    """

    def __init__(self, config: RAGBasicConfig | None = None):
        """
        Initialise le RAG basique.

        Args:
            config: Configuration (utilise les défauts si None)
        """
        super().__init__(
            rag_type=RAGType.RAG_BASIC,
            name="RAG Basique (FAISS)",
            version="1.0.0",
        )

        self.config = config or RAGBasicConfig()
        self.index = None
        self.documents: list[dict] = []
        self.client = None

        # Métriques
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0

    def initialize(self) -> None:
        """
        Initialise le RAG : charge l'index FAISS et le client OpenAI.

        Raises:
            FileNotFoundError: Si l'index FAISS n'existe pas
            ValueError: Si la clé API OpenAI n'est pas définie
        """
        # Charger l'index FAISS
        if not self.config.index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS non trouvé: {self.config.index_path}\n"
                "Lancez d'abord: python -m experiments.rag_basic.data_indexer"
            )

        try:
            import faiss
            self.index = faiss.read_index(str(self.config.index_path))
            logger.info(f"✅ Index FAISS chargé: {self.index.ntotal} vecteurs")
        except ImportError:
            raise ImportError("faiss non installé: pip install faiss-cpu")

        # Charger les métadonnées
        if not self.config.metadata_path.exists():
            raise FileNotFoundError(f"Métadonnées non trouvées: {self.config.metadata_path}")

        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.documents = metadata["documents"]
            logger.info(f"✅ {len(self.documents)} documents chargés")

        # Initialiser OpenAI
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY non définie dans .env")
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ Client OpenAI initialisé")
        except ImportError:
            raise ImportError("openai non installé: pip install openai")

        self._initialized = True

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Génère l'embedding d'un texte.

        Args:
            text: Texte à encoder

        Returns:
            Vecteur d'embedding normalisé
        """
        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        # Normaliser pour cosine similarity
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
        Recherche les documents les plus pertinents.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            domain: Filtrer par domaine (sante/pollution)

        Returns:
            Liste des documents pertinents avec scores
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Générer l'embedding de la query
        query_embedding = self._get_embedding(query)

        # Rechercher dans FAISS (récupérer plus si on filtre par domaine)
        search_k = top_k * 3 if domain else top_k
        scores, indices = self.index.search(query_embedding.reshape(1, -1), search_k)

        # Construire les résultats
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            # Filtrer par domaine si spécifié
            if domain:
                doc_domain = doc["metadata"].get("domain", "")
                domain_str = domain.value if hasattr(domain, "value") else str(domain)
                if doc_domain != domain_str:
                    continue

            # Filtrer par score minimum
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

    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """
        Génère une réponse basée sur le contexte.

        Args:
            query: Question de l'utilisateur
            context: Documents récupérés

        Returns:
            Réponse avec sources et métadonnées
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        # Construire le contexte
        if context:
            context_parts = []
            for i, doc in enumerate(context, 1):
                source = doc.metadata.get("source", "unknown")
                date = doc.metadata.get("date", "date inconnue")
                context_parts.append(f"[{i}] {doc.content} (Date: {date})")

            context_text = "\n\n".join(context_parts)
        else:
            context_text = "Aucun document pertinent trouvé dans la base de données."

        # Construire le prompt
        user_prompt = f"""Contexte (sources officielles) :
{context_text}

Question : {query}

Réponds en citant les sources et les dates. Si les données ne permettent pas de répondre précisément, dis-le clairement."""

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

            # Extraire la réponse
            answer = response.choices[0].message.content or ""
            usage = response.usage

            # Calculer les coûts
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = input_tokens + output_tokens

            embedding_cost = 0.00002  # Approximation pour la query
            llm_cost = (
                (input_tokens / 1000) * self.config.llm_input_cost_per_1k
                + (output_tokens / 1000) * self.config.llm_output_cost_per_1k
            )
            total_cost = embedding_cost + llm_cost

            # Mettre à jour les totaux
            self._total_tokens += total_tokens
            self._total_cost += total_cost
            self._query_count += 1

            # Construire les sources
            sources = [
                Source(
                    name=doc.metadata.get("source", "unknown"),
                    url="",
                    date=doc.metadata.get("date", ""),
                )
                for doc in context
            ]

            # Calculer la confiance basée sur les scores de retrieval
            avg_score = sum(doc.score for doc in context) / len(context) if context else 0
            confidence = min(0.95, avg_score)

            return RAGResponse(
                answer=answer,
                sources=sources,
                documents=context,
                confidence=confidence,
                metadata={
                    "latency_ms": generation_time * 1000,
                    "retrieval_time_ms": self._total_retrieval_time * 1000 / max(1, self._query_count),
                    "tokens_used": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": total_cost,
                    "model": self.config.llm_model,
                    "num_docs_retrieved": len(context),
                    "avg_relevance_score": avg_score,
                    "has_context": len(context) > 0,
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
    ) -> RAGResponse:
        """
        Pipeline RAG complet : retrieve puis generate.

        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à récupérer
            domain: Filtrer par domaine

        Returns:
            Réponse complète avec sources
        """
        if not self._initialized:
            self.initialize()

        # Étape 1: Retrieval
        context = self.retrieve(query, top_k=top_k, domain=domain)

        # Étape 2: Generation
        response = self.generate(query, context)

        return response

    def add_documents(
        self,
        documents: list[Document],
        domain: Domain,
    ) -> int:
        """
        Ajoute de nouveaux documents à l'index.

        Note: Pour le RAG basique, cette méthode nécessite
        de reconstruire l'index complet.

        Args:
            documents: Documents à ajouter
            domain: Domaine des documents

        Returns:
            Nombre de documents ajoutés
        """
        logger.warning(
            "add_documents() non implémenté pour BasicRAG. "
            "Utilisez data_indexer.py pour reconstruire l'index."
        )
        return 0

    def get_stats(self) -> dict[str, Any]:
        """
        Retourne les statistiques d'utilisation.

        Returns:
            Dictionnaire avec métriques
        """
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "total_queries": self._query_count,
            "avg_retrieval_time_ms": (
                self._total_retrieval_time * 1000 / max(1, self._query_count)
            ),
            "avg_generation_time_ms": (
                self._total_generation_time * 1000 / max(1, self._query_count)
            ),
            "model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "index_size": self.index.ntotal if self.index else 0,
        }

    def reset_stats(self) -> None:
        """Réinitialise les statistiques."""
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_retrieval_time = 0.0
        self._total_generation_time = 0.0
        self._query_count = 0
