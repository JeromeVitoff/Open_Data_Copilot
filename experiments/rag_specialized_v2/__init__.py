"""
RAG Spécialisé v2 - Embeddings médicaux + LLM médical
======================================================

Architecture avancée combinant :
- CamemBERT-bio (almanach/camembert-bio-base) pour les embeddings médicaux français
- BioMistral-7B pour la génération spécialisée biomédical
- Hybrid retrieval FAISS + BM25 + CrossEncoder reranking
"""
