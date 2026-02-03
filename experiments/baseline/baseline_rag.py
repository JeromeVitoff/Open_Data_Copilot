"""
OpenDataCopilot - Baseline RAG Implementation
==============================================

Implémentation de la baseline SANS RAG.
Le LLM répond directement sans contexte documentaire.

Cette baseline sert de référence pour mesurer l'apport du RAG.
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.core.rag_interface import (
    RAGInterface,
    RAGType,
    RAGResponse,
    Document,
    Source,
    Domain,
)
from .config import BaselineConfig

# Charger les variables d'environnement
load_dotenv()


class BaselineRAG(RAGInterface):
    """
    Baseline sans RAG - Le LLM répond sans contexte documentaire.

    Cette implémentation sert de référence pour mesurer :
    - Le taux d'hallucination sans contexte
    - Les limites des connaissances du LLM
    - L'apport du RAG en comparaison

    Attributes:
        config: Configuration de la baseline
        client: Client OpenAI
    """

    def __init__(self, config: BaselineConfig | None = None):
        """
        Initialise la baseline.

        Args:
            config: Configuration (utilise les défauts si None)
        """
        super().__init__(
            rag_type=RAGType.BASELINE,
            name="Baseline (Sans RAG)",
            version="1.0.0",
        )

        self.config = config or BaselineConfig()
        self.client: OpenAI | None = None
        self._total_tokens = 0
        self._total_cost = 0.0

    def initialize(self) -> None:
        """Initialise le client OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY non définie dans .env")

        self.client = OpenAI(api_key=api_key)
        self._initialized = True

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Domain | None = None,
    ) -> list[Document]:
        """
        Baseline: ne récupère aucun document.

        Returns:
            Liste vide (pas de contexte)
        """
        # Baseline = pas de retrieval
        return []

    def generate(
        self,
        query: str,
        context: list[Document],
    ) -> RAGResponse:
        """
        Génère une réponse SANS contexte documentaire.

        Le LLM utilise uniquement ses connaissances internes.

        Args:
            query: Question de l'utilisateur
            context: Ignoré (toujours vide pour baseline)

        Returns:
            Réponse avec métadonnées (tokens, coût, latence)
        """
        if not self._initialized:
            self.initialize()

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": query},
                ],
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extraire les informations de la réponse
            answer = response.choices[0].message.content or ""
            usage = response.usage

            # Calculer les tokens et coûts
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            total_tokens = input_tokens + output_tokens

            cost = (
                (input_tokens / 1000) * self.config.input_cost_per_1k
                + (output_tokens / 1000) * self.config.output_cost_per_1k
            )

            # Mettre à jour les totaux
            self._total_tokens += total_tokens
            self._total_cost += cost

            return RAGResponse(
                answer=answer,
                sources=[],  # Pas de sources (baseline)
                documents=[],  # Pas de documents
                confidence=0.5,  # Confiance moyenne (pas de contexte)
                metadata={
                    "latency_ms": latency_ms,
                    "tokens_used": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost,
                    "model": self.config.model,
                    "has_context": False,
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RAGResponse(
                answer=f"Erreur lors de la génération: {str(e)}",
                sources=[],
                documents=[],
                confidence=0.0,
                metadata={
                    "latency_ms": latency_ms,
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
        Pipeline baseline: génère directement sans retrieval.

        Args:
            query: Question de l'utilisateur
            top_k: Ignoré (pas de retrieval)
            domain: Ignoré (pas de filtrage)

        Returns:
            Réponse générée sans contexte
        """
        if not self._initialized:
            self.initialize()
            self._initialized = True

        # Pas de retrieval, génération directe
        return self.generate(query, context=[])

    def add_documents(
        self,
        documents: list[Document],
        domain: Domain,
    ) -> int:
        """
        Baseline: n'ajoute pas de documents (pas d'index).

        Returns:
            0 (aucun document ajouté)
        """
        # Baseline = pas d'indexation
        return 0

    def get_stats(self) -> dict[str, Any]:
        """
        Retourne les statistiques d'utilisation.

        Returns:
            Dictionnaire avec tokens totaux et coût total
        """
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
            "model": self.config.model,
        }

    def reset_stats(self) -> None:
        """Réinitialise les statistiques."""
        self._total_tokens = 0
        self._total_cost = 0.0
