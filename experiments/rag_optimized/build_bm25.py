#!/usr/bin/env python3
"""
Construction et persistance de l'index BM25.

À exécuter UNE SEULE FOIS pour pré-calculer l'index BM25 sur 1.2M documents.
L'index est sauvegardé en pickle et rechargé en ~1-2s au démarrage.

Usage:
    python -m experiments.rag_optimized.build_bm25
    python experiments/rag_optimized/build_bm25.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_optimized.config import RAGOptimizedConfig


def build_and_save_bm25_index(config: RAGOptimizedConfig | None = None) -> Path:
    """
    Construit l'index BM25Okapi sur le corpus complet et le sauvegarde.

    Args:
        config: Configuration du RAG optimisé

    Returns:
        Chemin vers le fichier pickle sauvegardé
    """
    if config is None:
        config = RAGOptimizedConfig()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank-bm25 non installé: pip install rank-bm25")

    output_path = config.metadata_path.parent / "bm25_index.pkl"

    # --- Chargement des documents ---
    if not config.metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées non trouvées: {config.metadata_path}")

    print(f"📂 Chargement des métadonnées depuis {config.metadata_path}...")
    t0 = time.time()
    with open(config.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    documents = metadata["documents"]
    n_docs = len(documents)
    print(f"   ✅ {n_docs:,} documents chargés en {time.time()-t0:.1f}s")

    # --- Tokenisation ---
    print(f"\n⚙️  Tokenisation de {n_docs:,} documents...")
    t1 = time.time()
    tokenized_corpus = [doc["text"].lower().split() for doc in documents]
    print(f"   ✅ Tokenisation terminée en {time.time()-t1:.1f}s")

    # --- Construction BM25 ---
    print(f"\n⚙️  Construction de l'index BM25 (k1={config.bm25_k1}, b={config.bm25_b})...")
    t2 = time.time()
    bm25 = BM25Okapi(tokenized_corpus, k1=config.bm25_k1, b=config.bm25_b)
    build_time = time.time() - t2
    print(f"   ✅ Index BM25 construit en {build_time:.1f}s")

    # --- Sauvegarde ---
    print(f"\n💾 Sauvegarde dans {output_path}...")
    t3 = time.time()
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "num_docs": n_docs,
                "k1": config.bm25_k1,
                "b": config.bm25_b,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    save_time = time.time() - t3
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"   ✅ Sauvegardé en {save_time:.1f}s — {size_mb:.0f} MB")

    total_time = time.time() - t0
    print(f"\n✅ Index BM25 prêt : {n_docs:,} docs | {size_mb:.0f} MB | {total_time:.1f}s total")
    print(f"   → Chargement futur : ~1-2s au lieu de ~{build_time:.0f}s")

    return output_path


if __name__ == "__main__":
    build_and_save_bm25_index()
