#!/usr/bin/env python3
"""
OpenDataCopilot - Indexation des données pour RAG Basique
=========================================================

Ce script indexe les données CSV dans un vectorstore FAISS
pour permettre le retrieval sémantique.

Usage:
    python -m experiments.rag_basic.data_indexer
    python experiments/rag_basic/data_indexer.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_basic.config import RAGBasicConfig

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


class DataIndexer:
    """
    Indexeur de données pour le RAG basique.

    Charge les fichiers CSV, les convertit en chunks textuels,
    génère des embeddings et crée un index FAISS.
    """

    def __init__(self, config: RAGBasicConfig | None = None):
        """
        Initialise l'indexeur.

        Args:
            config: Configuration (utilise les défauts si None)
        """
        self.config = config or RAGBasicConfig()
        self.documents: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.client = None
        self.total_cost = 0.0
        self.start_time = None
        self.embedding_time = 0.0

    def _init_openai(self):
        """Initialise le client OpenAI."""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY non définie dans .env")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai non installé: pip install openai")

    def _load_csv(self, filepath: Path, domain: str) -> pd.DataFrame | None:
        """
        Charge un fichier CSV.

        Args:
            filepath: Chemin du fichier
            domain: Domaine (sante/pollution)

        Returns:
            DataFrame ou None si erreur
        """
        if not filepath.exists():
            logger.warning(f"Fichier non trouvé: {filepath}")
            return None

        try:
            # Essayer plusieurs encodages et séparateurs
            for sep in [",", ";"]:
                for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
                    try:
                        df = pd.read_csv(filepath, sep=sep, encoding=encoding, low_memory=False)
                        if len(df.columns) > 1:
                            return df
                    except Exception:
                        continue

            logger.warning(f"Impossible de parser: {filepath}")
            return None

        except Exception as e:
            logger.error(f"Erreur chargement {filepath}: {e}")
            return None

    def _row_to_text(self, row: pd.Series, source: str, domain: str) -> str:
        """
        Convertit une ligne CSV en texte lisible.

        Args:
            row: Ligne du DataFrame
            source: Nom du fichier source
            domain: Domaine (sante/pollution)

        Returns:
            Texte formaté pour l'indexation
        """
        parts = []

        # Identifier les colonnes clés selon le fichier
        if "covid_hospitalisations" in source:
            parts.append(f"Hospitalisations COVID-19")
            if "jour" in row.index and pd.notna(row.get("jour")):
                parts.append(f"Date: {row['jour']}")
            if "dep" in row.index and pd.notna(row.get("dep")):
                parts.append(f"Département: {row['dep']}")
            if "hosp" in row.index and pd.notna(row.get("hosp")):
                parts.append(f"Hospitalisations: {row['hosp']}")
            if "rea" in row.index and pd.notna(row.get("rea")):
                parts.append(f"Réanimation: {row['rea']}")
            if "rad" in row.index and pd.notna(row.get("rad")):
                parts.append(f"Retours à domicile: {row['rad']}")
            if "dc" in row.index and pd.notna(row.get("dc")):
                parts.append(f"Décès: {row['dc']}")

        elif "covid_tests" in source:
            parts.append(f"Tests COVID-19")
            if "jour" in row.index and pd.notna(row.get("jour")):
                parts.append(f"Date: {row['jour']}")
            if "dep" in row.index and pd.notna(row.get("dep")):
                parts.append(f"Département: {row['dep']}")
            if "P" in row.index and pd.notna(row.get("P")):
                parts.append(f"Tests positifs: {row['P']}")
            if "T" in row.index and pd.notna(row.get("T")):
                parts.append(f"Tests total: {row['T']}")

        elif "sursaud_urgences" in source:
            parts.append(f"Passages aux urgences")
            if "date_de_passage" in row.index and pd.notna(row.get("date_de_passage")):
                parts.append(f"Date: {row['date_de_passage']}")
            if "dep" in row.index and pd.notna(row.get("dep")):
                parts.append(f"Département: {row['dep']}")
            if "nbre_pass_tot" in row.index and pd.notna(row.get("nbre_pass_tot")):
                parts.append(f"Passages totaux: {row['nbre_pass_tot']}")
            if "nbre_hospit" in row.index and pd.notna(row.get("nbre_hospit")):
                parts.append(f"Hospitalisations: {row['nbre_hospit']}")

        elif "professionnels_sante" in source:
            parts.append(f"Démographie médicale")
            if "departement" in row.index and pd.notna(row.get("departement")):
                parts.append(f"Département: {row['departement']}")
            if "libelle_departement" in row.index and pd.notna(row.get("libelle_departement")):
                parts.append(f"Nom: {row['libelle_departement']}")
            if "effectif" in row.index and pd.notna(row.get("effectif")):
                parts.append(f"Effectif médecins: {row['effectif']}")
            if "profession" in row.index and pd.notna(row.get("profession")):
                parts.append(f"Profession: {row['profession']}")

        elif "airparif" in source:
            parts.append(f"Qualité de l'air Île-de-France")
            if "date_ech" in row.index and pd.notna(row.get("date_ech")):
                # Convertir timestamp en date
                try:
                    ts = int(row["date_ech"]) / 1000
                    date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    parts.append(f"Date: {date_str}")
                except Exception:
                    pass
            if "lib_zone" in row.index and pd.notna(row.get("lib_zone")):
                parts.append(f"Zone: {row['lib_zone']}")
            if "qualif" in row.index and pd.notna(row.get("qualif")):
                parts.append(f"Qualité: {row['qualif']}")
            if "val_no2" in row.index and pd.notna(row.get("val_no2")):
                parts.append(f"NO2: {row['val_no2']}")
            if "val_pm10" in row.index and pd.notna(row.get("val_pm10")):
                parts.append(f"PM10: {row['val_pm10']}")
            if "val_pm25" in row.index and pd.notna(row.get("val_pm25")):
                parts.append(f"PM2.5: {row['val_pm25']}")
            if "val_o3" in row.index and pd.notna(row.get("val_o3")):
                parts.append(f"O3: {row['val_o3']}")

        elif "openaq" in source:
            parts.append(f"Qualité de l'air (OpenAQ)")
            if "city" in row.index and pd.notna(row.get("city")):
                parts.append(f"Ville: {row['city']}")
            if "parameter" in row.index and pd.notna(row.get("parameter")):
                parts.append(f"Polluant: {row['parameter']}")
            if "value" in row.index and pd.notna(row.get("value")):
                parts.append(f"Valeur: {row['value']}")
            if "unit" in row.index and pd.notna(row.get("unit")):
                parts.append(f"Unité: {row['unit']}")
            if "date" in row.index and pd.notna(row.get("date")):
                parts.append(f"Date: {str(row['date'])[:10]}")

        else:
            # Format générique
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col}: {val}")

        parts.append(f"Source: {source}")
        return " | ".join(parts)

    def _extract_metadata(self, row: pd.Series, source: str, domain: str) -> dict:
        """
        Extrait les métadonnées d'une ligne.

        Args:
            row: Ligne du DataFrame
            source: Nom du fichier source
            domain: Domaine (sante/pollution)

        Returns:
            Dictionnaire de métadonnées
        """
        metadata = {
            "source": source,
            "domain": domain,
        }

        # Extraire la date
        date_cols = ["jour", "date", "date_de_passage", "date_ech", "fetch_date"]
        for col in date_cols:
            if col in row.index and pd.notna(row.get(col)):
                val = row[col]
                if col == "date_ech":
                    try:
                        ts = int(val) / 1000
                        metadata["date"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    except Exception:
                        metadata["date"] = str(val)
                else:
                    metadata["date"] = str(val)[:10]
                break

        # Extraire le département/zone
        if "dep" in row.index and pd.notna(row.get("dep")):
            metadata["departement"] = str(row["dep"])
        elif "departement" in row.index and pd.notna(row.get("departement")):
            metadata["departement"] = str(row["departement"])
        elif "lib_zone" in row.index and pd.notna(row.get("lib_zone")):
            metadata["zone"] = str(row["lib_zone"])
        elif "city" in row.index and pd.notna(row.get("city")):
            metadata["city"] = str(row["city"])

        # Extraire les métriques clés
        metrics = {}
        metric_cols = {
            "hosp": "hospitalisations",
            "rea": "reanimation",
            "dc": "deces",
            "P": "tests_positifs",
            "T": "tests_total",
            "val_no2": "no2",
            "val_pm10": "pm10",
            "val_pm25": "pm25",
            "val_o3": "o3",
            "effectif": "effectif_medecins",
        }
        for col, name in metric_cols.items():
            if col in row.index and pd.notna(row.get(col)):
                try:
                    metrics[name] = float(row[col])
                except (ValueError, TypeError):
                    pass

        if metrics:
            metadata["metrics"] = metrics

        return metadata

    def load_all_data(self, sample_size: int | None = None) -> int:
        """
        Charge tous les fichiers de données.

        Args:
            sample_size: Nombre de lignes par fichier (None = tout)

        Returns:
            Nombre total de documents chargés
        """
        logger.info("📂 Chargement des données...")
        self.documents = []

        for domain, files in self.config.data_files.items():
            base_dir = self.config.sante_dir if domain == "sante" else self.config.pollution_dir

            for filename in files:
                filepath = base_dir / filename
                logger.info(f"   📄 {filename}...")

                df = self._load_csv(filepath, domain)
                if df is None:
                    continue

                # Échantillonnage si demandé
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    logger.info(f"      → Échantillon: {sample_size} lignes")

                # Convertir en documents
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"      {filename}", leave=False):
                    text = self._row_to_text(row, filename, domain)
                    metadata = self._extract_metadata(row, filename, domain)

                    self.documents.append({
                        "text": text,
                        "metadata": metadata,
                    })

                logger.info(f"      ✅ {len(df)} lignes chargées")

        logger.info(f"\n📊 Total: {len(self.documents)} documents")
        return len(self.documents)

    def generate_embeddings(self, batch_size: int = 1000) -> np.ndarray:
        """
        Génère les embeddings pour tous les documents avec monitoring complet.

        Args:
            batch_size: Taille des batches pour l'API (défaut: 1000 pour VM puissante)

        Returns:
            Matrice d'embeddings (n_docs, embedding_dim)
        """
        if not self.documents:
            raise ValueError("Aucun document chargé")

        if self.client is None:
            self._init_openai()

        logger.info(f"\n🧮 Génération des embeddings ({len(self.documents):,} documents)...")
        logger.info(f"   📦 Batch size: {batch_size}")
        logger.info(f"   💻 CPU disponibles: {psutil.cpu_count()}")
        logger.info(f"   💾 RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

        all_embeddings = []
        texts = [doc["text"] for doc in self.documents]

        embedding_start = time.time()
        num_batches = (len(texts) + batch_size - 1) // batch_size

        process = psutil.Process()

        with tqdm(total=len(texts), desc="   📊 Embeddings", unit="docs", ncols=100) as pbar:
            for batch_idx, i in enumerate(range(0, len(texts), batch_size), 1):
                batch_start = time.time()
                batch = texts[i:i + batch_size]

                try:
                    response = self.client.embeddings.create(
                        model=self.config.embedding_model,
                        input=batch,
                    )

                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)

                    # Calculer le coût du batch
                    batch_tokens = sum(len(t.split()) * 1.3 for t in batch)
                    batch_cost = (batch_tokens / 1000) * self.config.embedding_cost_per_1k
                    self.total_cost += batch_cost

                    # Temps et stats
                    batch_time = time.time() - batch_start
                    elapsed_total = time.time() - embedding_start
                    docs_processed = min(i + batch_size, len(texts))
                    avg_time_per_batch = elapsed_total / batch_idx
                    eta_seconds = avg_time_per_batch * (num_batches - batch_idx)

                    # RAM utilisée
                    ram_used_gb = process.memory_info().rss / (1024**3)
                    ram_avail_gb = psutil.virtual_memory().available / (1024**3)

                    # Mise à jour progress bar avec stats détaillées
                    pbar.update(len(batch))
                    pbar.set_postfix({
                        'batch': f'{batch_idx}/{num_batches}',
                        'time': f'{batch_time:.1f}s',
                        'cost': f'${self.total_cost:.3f}',
                        'RAM': f'{ram_used_gb:.1f}GB',
                        'ETA': f'{eta_seconds/60:.0f}m'
                    }, refresh=True)

                    # Log détaillé tous les 10 batches
                    if batch_idx % 10 == 0:
                        logger.info(f"   📊 Batch {batch_idx}/{num_batches} | "
                                   f"{docs_processed:,}/{len(texts):,} docs | "
                                   f"Coût: ${self.total_cost:.4f} | "
                                   f"RAM: {ram_used_gb:.1f}/{ram_avail_gb:.1f} GB | "
                                   f"ETA: {eta_seconds/60:.1f} min")

                except Exception as e:
                    logger.error(f"❌ Erreur API batch {batch_idx}: {e}")
                    raise

        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        self.embedding_time = time.time() - embedding_start

        logger.info(f"\n   ✅ Embeddings générés: {self.embeddings.shape}")
        logger.info(f"   ⏱️  Temps total: {self.embedding_time/60:.1f} minutes")
        logger.info(f"   💰 Coût total: ${self.total_cost:.4f}")
        logger.info(f"   ⚡ Vitesse: {len(texts)/self.embedding_time:.0f} docs/sec")

        return self.embeddings

    def create_faiss_index(self) -> None:
        """Crée et sauvegarde l'index FAISS."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss non installé: pip install faiss-cpu")

        if self.embeddings is None:
            raise ValueError("Embeddings non générés")

        logger.info("\n📦 Création de l'index FAISS...")

        # Créer l'index (Inner Product pour cosine similarity avec vecteurs normalisés)
        dimension = self.embeddings.shape[1]

        # Normaliser les vecteurs pour cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Index avec Inner Product (équivalent à cosine après normalisation)
        index = faiss.IndexFlatIP(dimension)
        index.add(self.embeddings)

        # Sauvegarder l'index
        faiss.write_index(index, str(self.config.index_path))
        logger.info(f"   ✅ Index sauvegardé: {self.config.index_path}")

        # Sauvegarder les métadonnées
        metadata = {
            "documents": self.documents,
            "created_at": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "embedding_model": self.config.embedding_model,
            "embedding_dim": dimension,
        }

        with open(self.config.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"   ✅ Métadonnées sauvegardées: {self.config.metadata_path}")

    def print_stats(self) -> None:
        """Affiche les statistiques complètes d'indexation."""
        if not self.documents:
            return

        logger.info("\n" + "=" * 80)
        logger.info("📊 STATISTIQUES D'INDEXATION COMPLÈTE")
        logger.info("=" * 80)

        # Compter par source et domaine
        sources = {}
        domains = {"sante": 0, "pollution": 0}
        dates_by_year = {}
        dates_by_month = {}

        for doc in self.documents:
            source = doc["metadata"]["source"]
            domain = doc["metadata"]["domain"]
            sources[source] = sources.get(source, 0) + 1
            domains[domain] += 1

            # Analyse temporelle
            if "date" in doc["metadata"]:
                date_str = doc["metadata"]["date"]
                try:
                    if len(date_str) >= 4:
                        year = date_str[:4]
                        dates_by_year[year] = dates_by_year.get(year, 0) + 1
                    if len(date_str) >= 7:
                        month = date_str[:7]
                        dates_by_month[month] = dates_by_month.get(month, 0) + 1
                except Exception:
                    pass

        logger.info(f"\n📁 VOLUME TOTAL")
        logger.info(f"   Documents indexés: {len(self.documents):,}")

        logger.info(f"\n📂 RÉPARTITION PAR DOMAINE")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            pct = count / len(self.documents) * 100
            bar = "█" * int(pct / 2)
            logger.info(f"   {domain:12} : {count:,} ({pct:5.1f}%) {bar}")

        logger.info(f"\n📄 RÉPARTITION PAR SOURCE")
        for source, count in sorted(sources.items(), key=lambda x: -x[1]):
            pct = count / len(self.documents) * 100
            logger.info(f"   {source:40} : {count:,} ({pct:5.1f}%)")

        if dates_by_year:
            logger.info(f"\n📅 RÉPARTITION TEMPORELLE PAR ANNÉE")
            for year in sorted(dates_by_year.keys(), reverse=True)[:10]:
                count = dates_by_year[year]
                pct = count / len(self.documents) * 100
                bar = "█" * int(pct / 2)
                logger.info(f"   {year} : {count:,} ({pct:5.1f}%) {bar}")

        if dates_by_month:
            logger.info(f"\n📆 DERNIERS MOIS (Top 10)")
            for month in sorted(dates_by_month.keys(), reverse=True)[:10]:
                count = dates_by_month[month]
                logger.info(f"   {month} : {count:,} documents")

        if self.embeddings is not None:
            size_mb = self.embeddings.nbytes / (1024 * 1024)
            size_gb = size_mb / 1024
            logger.info(f"\n💾 EMBEDDINGS")
            logger.info(f"   Dimension: {self.embeddings.shape[1]}")
            logger.info(f"   Taille mémoire: {size_mb:.1f} MB ({size_gb:.2f} GB)")
            logger.info(f"   Format: float32")

        # Taille de l'index FAISS
        if self.config.index_path.exists():
            index_size_mb = self.config.index_path.stat().st_size / (1024 * 1024)
            logger.info(f"\n📦 INDEX FAISS")
            logger.info(f"   Fichier: {self.config.index_path.name}")
            logger.info(f"   Taille: {index_size_mb:.1f} MB")

        if self.config.metadata_path.exists():
            metadata_size_mb = self.config.metadata_path.stat().st_size / (1024 * 1024)
            logger.info(f"\n📋 MÉTADONNÉES")
            logger.info(f"   Fichier: {self.config.metadata_path.name}")
            logger.info(f"   Taille: {metadata_size_mb:.1f} MB")

        logger.info(f"\n💰 COÛTS")
        logger.info(f"   Coût embeddings: ${self.total_cost:.4f}")
        logger.info(f"   Modèle: {self.config.embedding_model}")
        logger.info(f"   Prix/1K tokens: ${self.config.embedding_cost_per_1k}")

        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"\n⏱️  PERFORMANCE")
            logger.info(f"   Temps total: {total_time/60:.1f} minutes ({total_time/3600:.2f} heures)")
            if self.embedding_time > 0:
                logger.info(f"   Temps embeddings: {self.embedding_time/60:.1f} minutes")
                logger.info(f"   Vitesse moyenne: {len(self.documents)/self.embedding_time:.0f} docs/sec")
            logger.info(f"   RAM max utilisée: {psutil.Process().memory_info().rss / (1024**3):.1f} GB")

        logger.info("\n" + "=" * 80)


def main():
    """Point d'entrée principal - Indexation COMPLÈTE sur VM puissante."""
    logger.info("=" * 80)
    logger.info("🚀 OpenDataCopilot - INDEXATION COMPLÈTE (VM 128 CPU / 376 GB RAM)")
    logger.info("=" * 80)

    # Info système
    logger.info(f"\n💻 Configuration système:")
    logger.info(f"   CPU: {psutil.cpu_count()} cœurs")
    logger.info(f"   RAM totale: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"   RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    config = RAGBasicConfig()
    indexer = DataIndexer(config)
    indexer.start_time = time.time()

    # Charger les données - INDEXATION COMPLÈTE (453K documents)
    logger.info(f"\n{'='*80}")
    logger.info("📂 PHASE 1 : Chargement des données")
    logger.info(f"{'='*80}")
    num_docs = indexer.load_all_data(sample_size=None)

    if num_docs == 0:
        logger.error("❌ Aucun document chargé")
        return 1

    # Générer les embeddings - BATCHES DE 1000
    logger.info(f"\n{'='*80}")
    logger.info("🧮 PHASE 2 : Génération des embeddings")
    logger.info(f"{'='*80}")
    indexer.generate_embeddings(batch_size=1000)

    # Créer l'index FAISS
    logger.info(f"\n{'='*80}")
    logger.info("📦 PHASE 3 : Création de l'index FAISS")
    logger.info(f"{'='*80}")
    indexer.create_faiss_index()

    # Afficher les stats complètes
    indexer.print_stats()

    total_time = time.time() - indexer.start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ INDEXATION TERMINÉE EN {total_time/60:.1f} MINUTES!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
