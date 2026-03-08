#!/usr/bin/env python3
"""
OpenDataCopilot - Indexation médicale avec CamemBERT-bio
=========================================================

Ré-indexe les 1.2M documents avec des embeddings CamemBERT-bio (768 dims)
au lieu des embeddings OpenAI génériques (1536 dims).

L'index résultant est sauvegardé dans data/vectorstore/faiss_medical/.

Usage:
    python -m experiments.rag_specialized_v2.data_indexer_medical
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import psutil
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_specialized_v2.config import RAGSpecializedV2Config
from experiments.rag_specialized_v2.medical_embeddings import MedicalEmbeddings

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


def load_existing_documents(config: RAGSpecializedV2Config) -> list[dict]:
    """
    Charge les documents déjà indexés depuis metadata.json (index OpenAI existant).
    Évite de re-parser tous les CSV — on réutilise le texte et les métadonnées déjà extraits.

    Returns:
        Liste de dicts {'text': str, 'metadata': dict}
    """
    metadata_path = config.metadata_path  # faiss/metadata.json (index OpenAI)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Index OpenAI introuvable : {metadata_path}\n"
            "Lancez d'abord : python -m experiments.rag_basic.data_indexer"
        )

    logger.info(f"Chargement documents depuis {metadata_path}...")
    t0 = time.time()
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    logger.info(f"  {len(documents):,} documents charges en {time.time()-t0:.1f}s")
    return documents


def generate_medical_embeddings(
    documents: list[dict],
    model: MedicalEmbeddings,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Génère les embeddings CamemBERT-bio pour tous les documents.

    Args:
        documents: Liste de dicts avec clé 'text'
        model: Instance MedicalEmbeddings
        batch_size: Taille batch GPU (32 = bon compromis mémoire/vitesse)

    Returns:
        np.ndarray float32 de shape (n_docs, 768)
    """
    n = len(documents)
    logger.info(f"Generation embeddings CamemBERT-bio ({n:,} docs, batch={batch_size})...")
    logger.info(f"  RAM disponible : {psutil.virtual_memory().available / 1024**3:.1f} GB")

    texts = [doc["text"] for doc in documents]
    t0 = time.time()

    # Traitement en macro-batches pour le monitoring
    macro_batch = 10_000
    all_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, n, macro_batch), desc="Macro-batches", unit="batch"):
        chunk = texts[start:start + macro_batch]
        embs = model.embed_documents(chunk, batch_size=batch_size, show_progress=False)
        all_embeddings.append(embs.astype(np.float32))

        elapsed = time.time() - t0
        speed = (start + len(chunk)) / elapsed if elapsed > 0 else 0
        eta = (n - start - len(chunk)) / speed if speed > 0 else 0
        ram = psutil.virtual_memory().used / 1024**3
        logger.info(
            f"  {start + len(chunk):,}/{n:,} docs | "
            f"{speed:.0f} docs/s | ETA {eta/60:.1f}min | RAM {ram:.1f}GB"
        )

    embeddings = np.vstack(all_embeddings)
    total_time = time.time() - t0
    logger.info(f"  Embeddings calcules : {embeddings.shape} en {total_time/60:.1f} min")
    logger.info(f"  Vitesse moyenne : {n/total_time:.0f} docs/s")
    return embeddings


def build_faiss_index(
    embeddings: np.ndarray,
    config: RAGSpecializedV2Config,
    documents: list[dict],
) -> None:
    """
    Crée l'index FAISS IndexFlatIP et sauvegarde l'index + métadonnées.

    Args:
        embeddings: np.ndarray (n_docs, dim) float32
        config: Configuration avec les chemins de sauvegarde
        documents: Métadonnées des documents
    """
    dim = embeddings.shape[1]
    logger.info(f"Construction index FAISS (dim={dim}, {len(documents):,} vecteurs)...")

    # Normaliser pour cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"  Index FAISS : {index.ntotal:,} vecteurs")

    # Sauvegarder l'index
    faiss.write_index(index, str(config.medical_index_path))
    logger.info(f"  Index sauvegarde : {config.medical_index_path}")

    # Sauvegarder les métadonnées
    metadata = {
        "documents": documents,
        "created_at": datetime.now().isoformat(),
        "num_documents": len(documents),
        "embedding_model": config.medical_embedding_model,
        "embedding_dim": dim,
    }
    with open(config.medical_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"  Metadonnees sauvegardees : {config.medical_metadata_path}")

    # Taille sur disque
    idx_size_gb = config.medical_index_path.stat().st_size / 1024**3
    logger.info(f"  Taille index : {idx_size_gb:.2f} GB")


def main():
    print("=" * 70)
    print("Indexation medicale - CamemBERT-bio")
    print("=" * 70)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = RAGSpecializedV2Config()

    # 1. Charger les documents depuis l'index existant
    documents = load_existing_documents(config)

    # 2. Charger le modèle d'embeddings médical
    model = MedicalEmbeddings(
        model_name=config.medical_embedding_model,
        normalize=config.normalize_embeddings,
    )

    # 3. Générer les embeddings
    t_start = time.time()
    embeddings = generate_medical_embeddings(
        documents, model, batch_size=config.embedding_batch_size
    )

    # 4. Construire et sauvegarder l'index FAISS
    build_faiss_index(embeddings, config, documents)

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print("INDEXATION TERMINEE")
    print("=" * 70)
    print(f"Documents      : {len(documents):,}")
    print(f"Dimension      : {embeddings.shape[1]}")
    print(f"Temps total    : {total_time/60:.1f} min")
    print(f"Index FAISS    : {config.medical_index_path}")
    print(f"Metadonnees    : {config.medical_metadata_path}")
    print(f"Cout           : $0.00 (modele local)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
