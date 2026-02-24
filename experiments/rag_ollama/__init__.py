"""
OpenDataCopilot - RAG avec Ollama (LLM local)
===============================================

Implémentation RAG utilisant l'index FAISS existant pour le retrieval
(OpenAI text-embedding-3-small) et Ollama comme LLM local pour la génération.

Modèles disponibles :
- mistral:7b  (~4.4 GB, Q4_K_M)
- llama3:8b   (~4.7 GB, Q4_0)

Usage:
    from experiments.rag_ollama import OllamaRAG, OllamaConfig

    config = OllamaConfig(model_name="mistral:7b")
    rag = OllamaRAG(config)
    rag.initialize()
    response = rag.query("Question ?")
"""

from .rag_ollama import OllamaRAG
from .config import OllamaConfig

__all__ = ["OllamaRAG", "OllamaConfig"]
