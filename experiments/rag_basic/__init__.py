"""
OpenDataCopilot - RAG Basique avec FAISS
=========================================

Implémentation d'un système RAG simple utilisant FAISS pour
le retrieval de documents sur les données ouvertes françaises.
"""

from .rag_basic import BasicRAG
from .config import RAGBasicConfig

__all__ = ["BasicRAG", "RAGBasicConfig"]
