"""
OpenDataCopilot - RAG Optimisé
================================

Architecture RAG avancée avec :
- Retrieval hybride : FAISS (dense) + BM25 (sparse)
- Reranking sémantique : CrossEncoder ms-marco-MiniLM-L-6-v2
- Pipeline : Hybrid top-20 → Rerank → top-5 → GPT-3.5
"""
