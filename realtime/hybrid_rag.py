"""
RAG Hybride — fusion index historique FAISS + données temps réel (APIs).

Réutilise l'infrastructure existante du projet :
  - faiss.read_index() + metadata.json  (comme HybridRetriever)
  - OpenAI embeddings + chat             (comme OptimizedRAG)
  - TemporalDetector                     (Jour 1)
  - API wrappers SPF / Airparif / OpenAQ (Jour 1)
  - CacheManager                         (Jour 2)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI

from realtime.temporal_detector import TemporalDetector
from realtime.api_wrappers.spf_api import SPFRealtimeAPI
from realtime.api_wrappers.airparif_api import AirparifRealtimeAPI
from realtime.api_wrappers.openaq_api import OpenAQRealtimeAPI
from realtime.cache_manager import CacheManager

# Chemins par défaut (relatifs à la racine du projet)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INDEX = str(_PROJECT_ROOT / "data/vectorstore/faiss/index.faiss")
_DEFAULT_META  = str(_PROJECT_ROOT / "data/vectorstore/faiss/metadata.json")


class HybridRAG:
    """
    RAG combinant données historiques (FAISS 1.2M docs) et temps réel (APIs).

    Usage ::
        rag = HybridRAG()
        result = rag.query("Qualité de l'air à Paris aujourd'hui")
        print(result["answer"])
    """

    def __init__(
        self,
        faiss_index_path: str = _DEFAULT_INDEX,
        metadata_path: str = _DEFAULT_META,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
    ) -> None:
        print("🔧 Initialisation HybridRAG...")

        # ── OpenAI ────────────────────────────────────────────────────────────
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.llm_model = os.getenv("OPENAI_MODEL", llm_model)

        # ── Index FAISS ───────────────────────────────────────────────────────
        print(f"📚 Chargement index FAISS : {faiss_index_path}")
        import faiss as _faiss
        self._faiss = _faiss
        self.index = _faiss.read_index(faiss_index_path)
        print(f"   ✅ {self.index.ntotal:,} vecteurs")

        # ── Métadonnées ───────────────────────────────────────────────────────
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.documents: List[Dict] = meta["documents"]
        print(f"   ✅ {len(self.documents):,} documents")

        # ── Détecteur temporalité ─────────────────────────────────────────────
        self.temporal_detector = TemporalDetector()

        # ── APIs temps réel ───────────────────────────────────────────────────
        self.spf_api: Optional[SPFRealtimeAPI] = None
        self.airparif_api: Optional[AirparifRealtimeAPI] = None
        self.openaq_api: Optional[OpenAQRealtimeAPI] = None

        try:
            self.spf_api = SPFRealtimeAPI()
            print("   ✅ SPF API")
        except Exception as exc:
            print(f"   ⚠️  SPF non disponible : {exc}")

        try:
            self.airparif_api = AirparifRealtimeAPI()
            if self.airparif_api.api_key:
                print("   ✅ Airparif API")
            else:
                print("   ⚠️  Airparif : clé absente")
                self.airparif_api = None
        except Exception as exc:
            print(f"   ⚠️  Airparif non disponible : {exc}")

        try:
            self.openaq_api = OpenAQRealtimeAPI()
            if self.openaq_api.api_key:
                print("   ✅ OpenAQ API")
            else:
                print("   ⚠️  OpenAQ : clé absente")
                self.openaq_api = None
        except Exception as exc:
            print(f"   ⚠️  OpenAQ non disponible : {exc}")

        # ── Cache ─────────────────────────────────────────────────────────────
        self.cache = CacheManager()

        print("✅ HybridRAG initialisé\n")

    # ── Interface principale ──────────────────────────────────────────────────

    def query(
        self,
        question: str,
        k_historical: int = 5,
        k_realtime: int = 5,
    ) -> Dict:
        """
        Répond à une question en fusionnant sources historiques + temps réel.

        Returns ::
            {
                "answer": str,
                "sources": List[Dict],
                "temporal_analysis": Dict,
                "num_historical": int,
                "num_realtime": int,
            }
        """
        print(f"\n{'='*70}")
        print(f"❓ Question : {question}")
        print("="*70)

        # 1. Analyse temporalité
        temporal_info = self.temporal_detector.detect(question)
        print(f"\n🕐 Temporalité : {temporal_info['type']}")
        print(f"   Score RT : {temporal_info['realtime_score']:.2f}  "
              f"| Besoin données fraîches : {temporal_info['needs_realtime_data']}")

        # 2. Données historiques (toujours)
        print("\n📚 Recherche historique (FAISS)…")
        historical_docs = self._search_faiss(question, k=k_historical)
        print(f"   ✅ {len(historical_docs)} documents historiques")
        for doc in historical_docs:
            doc["metadata"]["source_type"] = "historical"

        # 3. Données temps réel (si besoin)
        realtime_docs: List[Dict] = []
        if temporal_info["needs_realtime_data"]:
            print("\n🔴 Recherche temps réel (APIs)…")
            realtime_docs = self._fetch_realtime_data(question)
            print(f"   ✅ {len(realtime_docs)} documents temps réel")

        all_docs = historical_docs + realtime_docs

        # 4. Génération
        print("\n🤖 Génération réponse LLM…")
        answer = self._generate_answer(question, all_docs)

        return {
            "answer": answer,
            "sources": all_docs,
            "temporal_analysis": temporal_info,
            "num_historical": len(historical_docs),
            "num_realtime": len(realtime_docs),
        }

    # ── Retrieval FAISS ───────────────────────────────────────────────────────

    def _search_faiss(self, query: str, k: int) -> List[Dict]:
        """Cherche les k documents les plus proches dans l'index FAISS."""
        # Embedding de la requête
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
        self._faiss.normalize_L2(vec)

        distances, indices = self.index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append({
                "text": doc["text"],
                "metadata": {
                    **doc.get("metadata", {}),
                    "faiss_score": float(dist),
                    "source_type": "historical",
                },
            })
        return results

    # ── Récupération temps réel ───────────────────────────────────────────────

    def _fetch_realtime_data(self, question: str) -> List[Dict]:
        """Sélectionne et appelle les APIs pertinentes selon la question."""
        docs: List[Dict] = []
        q = question.lower()

        # SPF — COVID / santé
        if any(kw in q for kw in ["covid", "hospitali", "santé", "maladie", "épidém"]):
            docs.extend(self._fetch_spf())

        # Pollution — Airparif + OpenAQ
        if any(kw in q for kw in ["pollution", "qualité", "air", "no2", "pm10", "pm2.5", "o3", "atmo"]):
            docs.extend(self._fetch_airparif())
            docs.extend(self._fetch_openaq())

        # Si aucun mot-clé précis mais données temps réel demandées → OpenAQ par défaut
        if not docs:
            docs.extend(self._fetch_openaq())

        return docs

    def _fetch_spf(self) -> List[Dict]:
        if not self.spf_api:
            return []
        cache_key = {"api": "spf", "days": 30}
        cached = self.cache.get("spf", cache_key)
        if cached:
            return cached
        try:
            df = self.spf_api.get_covid_hospitalizations_recent(days=30)
            if df.empty:
                print("      ⚠️  SPF : aucune donnée récente (30j)")
                return []
            docs = self.spf_api.format_for_rag(df.head(5))
            for d in docs:
                d["metadata"]["source_type"] = "realtime"
            self.cache.set("spf", cache_key, docs)
            return docs
        except Exception as exc:
            print(f"      ⚠️  SPF error : {exc}")
            return []

    def _fetch_airparif(self) -> List[Dict]:
        if not self.airparif_api:
            return []
        cache_key = {"city": "Paris", "insee": "75056"}
        cached = self.cache.get("airparif", cache_key)
        if cached:
            return [cached]
        try:
            data = self.airparif_api.get_current_pollution(city="Paris", insee_code="75056")
            if not data:
                return []
            doc = self.airparif_api.format_for_rag(data)
            if doc:
                doc["metadata"]["source_type"] = "realtime"
                self.cache.set("airparif", cache_key, doc)
                return [doc]
        except Exception as exc:
            print(f"      ⚠️  Airparif error : {exc}")
        return []

    def _fetch_openaq(self) -> List[Dict]:
        if not self.openaq_api:
            return []
        cache_key = {"parameter": "no2", "region": "idf"}
        cached = self.cache.get("openaq", cache_key)
        if cached:
            return cached
        try:
            measures = self.openaq_api.get_latest_measurements(
                city="Paris", parameter="no2", limit=5
            )
            if not measures:
                return []
            docs = self.openaq_api.format_for_rag(measures[:3])
            for d in docs:
                d["metadata"]["source_type"] = "realtime"
            self.cache.set("openaq", cache_key, docs)
            return docs
        except Exception as exc:
            print(f"      ⚠️  OpenAQ error : {exc}")
            return []

    # ── Génération LLM ────────────────────────────────────────────────────────

    def _generate_answer(self, question: str, documents: List[Dict]) -> str:
        if not documents:
            return "Je n'ai pas trouvé de données pertinentes pour répondre à cette question."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            stype = doc["metadata"].get("source_type", "realtime")
            source = doc["metadata"].get("source", "Inconnu")
            date = doc["metadata"].get("date", "N/A")
            label = (
                f"[Doc {i}] {'Historique' if stype == 'historical' else 'Temps réel'}"
                f" — {source} — {date}"
            )
            context_parts.append(f"{label}\n{doc['text']}\n")

        context = "\n".join(context_parts)

        prompt = (
            "Tu es un assistant expert en santé publique et pollution atmosphérique.\n"
            "Réponds à la question en utilisant UNIQUEMENT les documents fournis.\n"
            "Cite TOUJOURS les sources (numéro du document, source, date).\n"
            "Si les documents contiennent des données temps réel ET historiques, "
            "mentionne-le explicitement.\n\n"
            f"Documents :\n{context}\n"
            f"Question : {question}\n\n"
            "Réponse (avec sources et dates) :"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"❌ Erreur LLM : {exc}")
            return f"Erreur lors de la génération de la réponse : {exc}"
