"""
OpenDataCopilot - Baseline RAG (Sans RAG)
==========================================

Ce module implémente la baseline sans RAG pour comparaison.
Le LLM répond directement sans contexte documentaire.
"""

from .baseline_rag import BaselineRAG
from .config import BaselineConfig

__all__ = ["BaselineRAG", "BaselineConfig"]
