"""
Retrieval hybride : FAISS (dense) + BM25 (sparse).

Combine les avantages :
- FAISS : similarité sémantique, robuste aux synonymes
- BM25  : correspondance exacte, excellent sur dates/codes/chiffres

Pipeline optimisé :
1. BM25 chargé depuis pickle pré-construit (1-2s au démarrage, 0ms par requête)
2. FAISS + BM25 exécutés en PARALLÈLE (ThreadPoolExecutor)
3. Fusion pondérée : score = alpha * dense + (1-alpha) * sparse
4. Tri par score fusionné → top-K candidats
"""

import concurrent.futures
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.core.rag_interface import Document, Domain
from experiments.rag_optimized.config import RAGOptimizedConfig


class HybridRetriever:
    """
    Retrieval hybride FAISS + BM25 avec fusion de scores.

    Optimisations :
    - BM25 chargé depuis index pré-construit (pickle) — évite 9s de reconstruction
    - FAISS et BM25 lancés en parallèle — économise max(t_faiss, t_bm25)

    Attributes:
        config: Configuration du RAG optimisé
        index: Index FAISS chargé
        documents: Liste des documents (métadonnées + texte)
        bm25: Index BM25Okapi (chargé depuis pickle ou construit)
        client: Client OpenAI pour les embeddings
    """

    def __init__(self, config: RAGOptimizedConfig):
        self.config = config
        self.index = None
        self.documents: list[dict] = []
        self.bm25 = None
        self.client = None

    @property
    def _bm25_path(self) -> Path:
        return self.config.metadata_path.parent / "bm25_index.pkl"

    def initialize(self, client: Any) -> None:
        """
        Charge l'index FAISS, les documents et l'index BM25.
        Si le pickle BM25 existe, le charge en ~1-2s.
        Sinon, construit et sauvegarde automatiquement.

        Args:
            client: Client OpenAI déjà initialisé
        """
        self.client = client

        # --- FAISS ---
        try:
            import faiss
            if not self.config.index_path.exists():
                raise FileNotFoundError(
                    f"Index FAISS non trouvé: {self.config.index_path}"
                )
            self.index = faiss.read_index(str(self.config.index_path))
            logger.info(f"✅ FAISS chargé: {self.index.ntotal:,} vecteurs")
        except ImportError:
            raise ImportError("faiss non installé: pip install faiss-cpu")

        # --- Documents ---
        if not self.config.metadata_path.exists():
            raise FileNotFoundError(
                f"Métadonnées non trouvées: {self.config.metadata_path}"
            )
        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.documents = metadata["documents"]
        logger.info(f"✅ {len(self.documents):,} documents chargés")

        # --- BM25 : pickle ou construction ---
        if self._bm25_path.exists():
            self._load_bm25_from_pickle()
        else:
            logger.warning(
                f"Index BM25 pré-construit non trouvé: {self._bm25_path}\n"
                "   Construction à la volée (lente). Exécutez build_bm25.py pour l'éviter."
            )
            self._build_and_save_bm25()

    def _load_bm25_from_pickle(self) -> None:
        """Charge l'index BM25 depuis le fichier pickle pré-construit."""
        logger.info(f"⚙️  Chargement index BM25 depuis {self._bm25_path}...")
        start = time.time()
        with open(self._bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        elapsed = time.time() - start
        logger.info(f"✅ Index BM25 chargé en {elapsed:.1f}s ({data['num_docs']:,} docs)")

    def _build_and_save_bm25(self) -> None:
        """Construit et sauvegarde l'index BM25 (fallback si pickle absent)."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 non installé: pip install rank-bm25")

        logger.info("⚙️  Construction de l'index BM25...")
        start = time.time()
        tokenized = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)
        elapsed = time.time() - start
        logger.info(f"✅ Index BM25 construit en {elapsed:.1f}s")

        # Sauvegarde pour les prochains lancements
        try:
            with open(self._bm25_path, "wb") as f:
                pickle.dump(
                    {"bm25": self.bm25, "num_docs": len(self.documents),
                     "k1": self.config.bm25_k1, "b": self.config.bm25_b},
                    f, protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"✅ Index BM25 sauvegardé: {self._bm25_path}")
        except Exception as e:
            logger.warning(f"⚠️  Impossible de sauvegarder l'index BM25: {e}")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Génère et normalise l'embedding d'une requête."""
        import faiss

        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

    def _faiss_search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> dict[int, float]:
        """Recherche FAISS → {doc_idx: score} normalisé [0, 1]."""
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k
        )
        result = {}
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.documents):
                # IndexFlatIP cosine scores dans [-1, 1] → normaliser dans [0, 1]
                result[int(idx)] = float((score + 1.0) / 2.0)
        return result

    def _bm25_search(self, query: str, top_k: int) -> dict[int, float]:
        """Recherche BM25 → {doc_idx: score} normalisé [0, 1]."""
        query_tokens = query.lower().split()
        raw_scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argpartition(raw_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(raw_scores[top_indices])[::-1]]

        max_score = raw_scores[top_indices[0]] if len(top_indices) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0

        return {
            int(idx): float(raw_scores[idx] / max_score)
            for idx in top_indices
            if raw_scores[idx] > 0
        }

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        domain: Domain | None = None,
    ) -> list[Document]:
        """
        Retrieval hybride parallèle : FAISS + BM25 simultanés.

        FAISS (embedding API ~700ms) et BM25 (CPU ~100ms) s'exécutent en
        parallèle via ThreadPoolExecutor → latence = max(t_faiss, t_bm25).

        Args:
            query: Requête utilisateur
            top_k: Nombre de documents à retourner
            domain: Filtrer par domaine

        Returns:
            Documents triés par score fusionné décroissant
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k

        search_k = top_k * 2 if domain else top_k

        # Lancer FAISS et BM25 en parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # FAISS : génère embedding puis cherche
            def dense_search():
                emb = self._get_embedding(query)
                return self._faiss_search(emb, search_k), emb

            # BM25 : directement sur le texte brut
            def sparse_search():
                return self._bm25_search(query, search_k)

            future_dense = executor.submit(dense_search)
            future_sparse = executor.submit(sparse_search)

            faiss_scores, _ = future_dense.result()
            bm25_scores = future_sparse.result()

        # Fusion pondérée
        all_indices = set(faiss_scores.keys()) | set(bm25_scores.keys())
        alpha = self.config.hybrid_alpha
        fused: list[tuple[int, float]] = [
            (idx, alpha * faiss_scores.get(idx, 0.0) + (1 - alpha) * bm25_scores.get(idx, 0.0))
            for idx in all_indices
        ]
        fused.sort(key=lambda x: x[1], reverse=True)

        # Construire les Documents
        results: list[Document] = []
        for idx, score in fused:
            if len(results) >= top_k:
                break

            doc = self.documents[idx]

            if domain:
                doc_domain = doc["metadata"].get("domain", "")
                domain_str = domain.value if hasattr(domain, "value") else str(domain)
                if doc_domain != domain_str:
                    continue

            sparse = bm25_scores.get(idx, 0.0)
            faiss_score_raw = faiss_scores.get(idx, 0.0) * 2.0 - 1.0
            if faiss_score_raw < self.config.min_relevance_score and faiss_score_raw > 0:
                if sparse < 0.5:
                    continue

            results.append(Document(
                content=doc["text"],
                metadata=doc["metadata"],
                score=score,
                doc_id=str(idx),
            ))

        return results
