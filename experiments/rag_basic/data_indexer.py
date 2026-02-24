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

        # ── ODISSE : COVID-19 synthèse par département
        elif "covid-19-synthese" in source or "covid_19_synthese" in source:
            parts.append("COVID-19 Suivi pandémie (ODISSE/SPF)")
            if "date" in row.index and pd.notna(row.get("date")):
                parts.append(f"Date: {row['date']}")
            if "lib_dep" in row.index and pd.notna(row.get("lib_dep")):
                parts.append(f"Département: {row['lib_dep']}")
            elif "dep" in row.index and pd.notna(row.get("dep")):
                parts.append(f"Département: {row['dep']}")
            if "hosp" in row.index and pd.notna(row.get("hosp")):
                parts.append(f"Hospitalisations: {row['hosp']}")
            if "rea" in row.index and pd.notna(row.get("rea")):
                parts.append(f"Réanimation: {row['rea']}")
            if "dchosp" in row.index and pd.notna(row.get("dchosp")):
                parts.append(f"Décès hôpital: {row['dchosp']}")
            if "rad" in row.index and pd.notna(row.get("rad")):
                parts.append(f"Retours domicile: {row['rad']}")
            if "tx_incid" in row.index and pd.notna(row.get("tx_incid")):
                parts.append(f"Taux incidence: {row['tx_incid']}")
            if "tx_pos" in row.index and pd.notna(row.get("tx_pos")):
                parts.append(f"Taux positivité: {row['tx_pos']}")
            if "lib_reg" in row.index and pd.notna(row.get("lib_reg")):
                parts.append(f"Région: {row['lib_reg']}")

        # ── ODISSE : Arboviroses
        elif "arboviroses" in source:
            parts.append("Arboviroses - Déclarations obligatoires (ODISSE/SPF)")
            if "mois" in row.index and pd.notna(row.get("mois")):
                parts.append(f"Période: {row['mois']}")
            if "libgeo" in row.index and pd.notna(row.get("libgeo")):
                parts.append(f"Département: {row['libgeo']}")
            elif "dep" in row.index and pd.notna(row.get("dep")):
                parts.append(f"Département: {row['dep']}")
            if "arbo" in row.index and pd.notna(row.get("arbo")):
                parts.append(f"Arbovirose: {row['arbo']}")
            if "nbcas_imp" in row.index and pd.notna(row.get("nbcas_imp")):
                parts.append(f"Cas importés: {row['nbcas_imp']}")
            if "nbcas_autoch" in row.index and pd.notna(row.get("nbcas_autoch")):
                parts.append(f"Cas autochtones: {row['nbcas_autoch']}")
            if "reglib" in row.index and pd.notna(row.get("reglib")):
                parts.append(f"Région: {row['reglib']}")

        # ── ODISSE : Infections respiratoires aiguës
        elif "infections-respiratoires" in source or "infections_respiratoires" in source:
            parts.append("Infections Respiratoires Aiguës - Urgences et SOS Médecins (ODISSE/SPF)")
            if "date_complet" in row.index and pd.notna(row.get("date_complet")):
                parts.append(f"Date: {row['date_complet']}")
            elif "semaine" in row.index and pd.notna(row.get("semaine")):
                parts.append(f"Semaine: {row['semaine']}")
            if "reglib" in row.index and pd.notna(row.get("reglib")):
                parts.append(f"Région: {row['reglib']}")
            elif "region" in row.index and pd.notna(row.get("region")):
                parts.append(f"Région: {row['region']}")
            if "sursaud_cl_age_gene" in row.index and pd.notna(row.get("sursaud_cl_age_gene")):
                parts.append(f"Tranche d'âge: {row['sursaud_cl_age_gene']}")
            if "taux_passages_ira_sau" in row.index and pd.notna(row.get("taux_passages_ira_sau")):
                parts.append(f"Taux passages urgences IRA: {row['taux_passages_ira_sau']:.1f}")
            if "taux_hospit_ira_sau" in row.index and pd.notna(row.get("taux_hospit_ira_sau")):
                parts.append(f"Taux hospitalisations IRA: {row['taux_hospit_ira_sau']:.1f}")
            if "taux_actes_ira_sos" in row.index and pd.notna(row.get("taux_actes_ira_sos")):
                parts.append(f"Taux actes SOS Médecins IRA: {row['taux_actes_ira_sos']:.1f}")

        # ── ODISSE : Données génériques (IST, VIH, antibiotiques, cardiovasculaires, etc.)
        elif any(kw in source for kw in [
            "infections-sexuellement", "vih-depistages", "antibiotiques",
            "maladies-cardio", "gestes-auto", "maladie-veineuse", "traumatisme",
            "legionellose", "couvertures-vaccinales",
        ]) or source.endswith(".csv") and "sante_odisse" in source:
            # Détecter automatiquement le thème depuis le nom de fichier
            if "vih" in source or "sida" in source:
                parts.append("VIH/SIDA - Dépistages (ODISSE/SPF)")
            elif "infections-sexuellement" in source or "ist" in source:
                parts.append("Infections Sexuellement Transmissibles (ODISSE/SPF)")
            elif "antibiotiques" in source:
                parts.append("Consommation antibiotiques (ODISSE/SPF)")
            elif "cardio" in source or "vasculaire" in source:
                parts.append("Maladies cardio-neuro-vasculaires (ODISSE/SPF)")
            elif "gestes-auto" in source or "suicid" in source:
                parts.append("Gestes auto-infligés / Santé mentale (ODISSE/SPF)")
            elif "veineuse" in source or "thrombo" in source:
                parts.append("Maladie veineuse thrombo-embolique (ODISSE/SPF)")
            elif "traumatisme" in source:
                parts.append("Traumatismes - Urgences (ODISSE/SPF)")
            elif "legionellose" in source:
                parts.append("Légionellose - Déclarations obligatoires (ODISSE/SPF)")
            elif "vaccination" in source or "vaccinale" in source:
                parts.append("Couvertures vaccinales (ODISSE/SPF)")
            else:
                parts.append(f"Données santé publique ODISSE/SPF")

            # Colonnes communes ODISSE
            date_cols = ["date", "annee", "mois", "semaine", "date_complet", "periode"]
            for col in date_cols:
                if col in row.index and pd.notna(row.get(col)):
                    parts.append(f"Période: {row[col]}")
                    break

            geo_cols = ["lib_dep", "libgeo", "dep", "reglib", "lib_reg", "region", "commune"]
            for col in geo_cols:
                if col in row.index and pd.notna(row.get(col)):
                    parts.append(f"Zone: {row[col]}")
                    break

            # Valeurs numériques clés
            val_cols = ["valeur", "value", "nb", "n", "effectif", "taux", "indicateur",
                        "nbcas", "nbre", "count", "nombre"]
            for col in row.index:
                col_lower = col.lower()
                if any(kw in col_lower for kw in val_cols) and pd.notna(row.get(col)):
                    try:
                        val = float(row[col])
                        parts.append(f"{col}: {val:.2f}")
                    except (ValueError, TypeError):
                        if str(row[col]).strip():
                            parts.append(f"{col}: {row[col]}")

        # ── Airparif historiques (mesures par station)
        elif "airparif_" in source and any(
            yr in source for yr in ["2020", "2021", "2022", "2023", "2024"]
        ):
            parts.append("Qualité de l'air Île-de-France - Mesures historiques (Airparif)")
            # Détecter l'année depuis le nom de fichier
            import re as _re
            yr_match = _re.search(r'(20[12]\d)', source)
            if yr_match:
                parts.append(f"Année: {yr_match.group(1)}")
            # Première colonne souvent = date/timestamp
            first_col = row.index[0] if len(row.index) > 0 else None
            if first_col and pd.notna(row.get(first_col)):
                val = str(row[first_col])
                if any(c in val for c in ['-', '/']):
                    parts.append(f"Date: {val[:16]}")
            # Colonnes de polluants
            pollutant_map = {
                "no2": "NO2 (µg/m³)", "no": "NO (µg/m³)", "nox": "NOx (µg/m³)",
                "pm10": "PM10 (µg/m³)", "pm25": "PM2.5 (µg/m³)", "pm2_5": "PM2.5 (µg/m³)",
                "o3": "O3 (µg/m³)", "co": "CO (mg/m³)", "so2": "SO2 (µg/m³)",
            }
            found_pollutants = []
            for col in row.index:
                col_lower = col.lower().replace(" ", "").replace(".", "").replace("-", "")
                for key, label in pollutant_map.items():
                    if key in col_lower and pd.notna(row.get(col)):
                        try:
                            val = float(row[col])
                            found_pollutants.append(f"{label}: {val:.1f}")
                        except (ValueError, TypeError):
                            pass
                        break
            if found_pollutants:
                parts.extend(found_pollutants[:6])
            # Nom de la station si disponible
            station_cols = ["station", "nom_station", "lib_zone", "site"]
            for col in station_cols:
                if col in row.index and pd.notna(row.get(col)):
                    parts.append(f"Station: {row[col]}")
                    break

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

        # Extraire la date (colonnes standard + ODISSE)
        date_cols = ["jour", "date", "date_de_passage", "date_ech", "fetch_date",
                     "date_complet", "mois", "semaine", "annee", "periode"]
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

        # Extraire le département/zone (standard + ODISSE)
        if "dep" in row.index and pd.notna(row.get("dep")):
            metadata["departement"] = str(row["dep"])
        elif "departement" in row.index and pd.notna(row.get("departement")):
            metadata["departement"] = str(row["departement"])
        elif "lib_dep" in row.index and pd.notna(row.get("lib_dep")):
            metadata["departement"] = str(row["lib_dep"])
        elif "libgeo" in row.index and pd.notna(row.get("libgeo")):
            metadata["departement"] = str(row["libgeo"])
        elif "lib_zone" in row.index and pd.notna(row.get("lib_zone")):
            metadata["zone"] = str(row["lib_zone"])
        elif "city" in row.index and pd.notna(row.get("city")):
            metadata["city"] = str(row["city"])
        # Région ODISSE
        if "reglib" in row.index and pd.notna(row.get("reglib")):
            metadata["region"] = str(row["reglib"])
        elif "lib_reg" in row.index and pd.notna(row.get("lib_reg")):
            metadata["region"] = str(row["lib_reg"])

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

    def _load_directory(
        self,
        directory: Path,
        domain: str,
        sample_size: int | None = None,
        label: str = "",
    ) -> int:
        """
        Charge tous les CSV d'un répertoire.

        Args:
            directory: Répertoire à scanner
            domain: Domaine (sante/pollution)
            sample_size: Lignes max par fichier (None = tout)
            label: Label pour les logs

        Returns:
            Nombre de documents chargés depuis ce répertoire
        """
        if not directory.exists():
            logger.debug(f"   Répertoire absent, skip: {directory}")
            return 0

        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            logger.debug(f"   Aucun CSV dans {directory}")
            return 0

        logger.info(f"\n   📁 {label or directory.name} ({len(csv_files)} fichiers CSV)")
        count = 0

        for filepath in csv_files:
            if filepath.name.startswith("metadata"):
                continue
            logger.info(f"      📄 {filepath.name}...")

            df = self._load_csv(filepath, domain)
            if df is None:
                continue

            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"         → Échantillon: {sample_size} lignes")

            source_name = filepath.name
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"         {filepath.name[:40]}", leave=False):
                text = self._row_to_text(row, source_name, domain)
                metadata = self._extract_metadata(row, source_name, domain)
                self.documents.append({"text": text, "metadata": metadata})

            logger.info(f"         ✅ {len(df):,} lignes chargées")
            count += len(df)

        return count

    def load_all_data(self, sample_size: int | None = None) -> int:
        """
        Charge tous les fichiers de données.

        Scanne les répertoires suivants :
        - data/raw/sante/         (sources existantes)
        - data/raw/sante_odisse/  (ODISSE - Santé publique France)
        - data/raw/pollution/     (Airparif indices + OpenAQ)
        - data/raw/pollution_airparif_hist/  (Airparif historiques 2020-2023)

        Args:
            sample_size: Nombre de lignes par fichier (None = tout)

        Returns:
            Nombre total de documents chargés
        """
        logger.info("📂 Chargement des données...")
        self.documents = []

        # ── Sources existantes (hardcodées dans config.data_files)
        for domain, files in self.config.data_files.items():
            base_dir = self.config.sante_dir if domain == "sante" else self.config.pollution_dir

            for filename in files:
                filepath = base_dir / filename
                logger.info(f"   📄 {filename}...")

                df = self._load_csv(filepath, domain)
                if df is None:
                    continue

                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    logger.info(f"      → Échantillon: {sample_size} lignes")

                for _, row in tqdm(df.iterrows(), total=len(df),
                                   desc=f"      {filename}", leave=False):
                    text = self._row_to_text(row, filename, domain)
                    metadata = self._extract_metadata(row, filename, domain)
                    self.documents.append({"text": text, "metadata": metadata})

                logger.info(f"      ✅ {len(df):,} lignes chargées")

        # ── Nouveaux répertoires (scan dynamique)
        new_dirs = [
            (self.config.sante_odisse_dir, "sante",
             "ODISSE (Santé publique France)"),
            (self.config.airparif_hist_dir, "pollution",
             "Airparif Historiques 2020-2023"),
        ]

        for directory, domain, label in new_dirs:
            self._load_directory(directory, domain, sample_size, label)

        logger.info(f"\n📊 Total: {len(self.documents):,} documents")
        return len(self.documents)

    def _est_tokens(self, text: str) -> float:
        """Estimation conservative du nombre de tokens.

        Utilise len(chars)/3 comme proxy fiable pour le français :
        le tokenizer BPE donne ~3-4 chars/token (accents, diacritiques inclus).
        On ajoute 20% de marge de sécurité.
        """
        return (len(text) / 3.0) * 1.2

    def _truncate_text(self, text: str, max_tokens: int | None = None) -> str:
        """Tronque un texte pour respecter la limite de tokens OpenAI.

        Approximation conservative : 1 token ≈ 3 chars (texte français).
        """
        if max_tokens is None:
            max_tokens = self.config.max_chunk_size
        max_chars = int(max_tokens * 3)
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def _make_token_aware_batches(self, texts: list[str], max_tokens_per_batch: int = 180_000, max_inputs: int = 2048) -> list[list[str]]:
        """Découpe les textes en batches qui respectent les limites OpenAI.

        Limites OpenAI embeddings : 300K tokens/req ET 2048 inputs/req.
        On utilise 180K tokens (60% de la limite) comme marge de sécurité.
        """
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0.0

        for text in texts:
            est_tokens = self._est_tokens(text)
            if current_batch and (current_tokens + est_tokens > max_tokens_per_batch or len(current_batch) >= max_inputs):
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = est_tokens
            else:
                current_batch.append(text)
                current_tokens += est_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def generate_embeddings(self, batch_size: int = 500) -> np.ndarray:
        """
        Génère les embeddings pour tous les documents avec monitoring complet.

        Args:
            batch_size: Ignoré, remplacé par un batching basé sur les tokens.

        Returns:
            Matrice d'embeddings (n_docs, embedding_dim)
        """
        if not self.documents:
            raise ValueError("Aucun document chargé")

        if self.client is None:
            self._init_openai()

        logger.info(f"\n🧮 Génération des embeddings ({len(self.documents):,} documents)...")
        logger.info(f"   💻 CPU disponibles: {psutil.cpu_count()}")
        logger.info(f"   💾 RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

        # Tronquer les textes AVANT de créer les batches
        texts = [self._truncate_text(doc["text"]) for doc in self.documents]

        # Batching par tokens pour respecter la limite OpenAI (300K tokens/req)
        batches = self._make_token_aware_batches(texts)
        num_batches = len(batches)
        logger.info(f"   📦 Batches token-aware: {num_batches} (max 250K tokens/batch)")

        all_embeddings = []
        embedding_start = time.time()
        process = psutil.Process()
        docs_done = 0

        with tqdm(total=len(texts), desc="   📊 Embeddings", unit="docs", ncols=100) as pbar:
            for batch_idx, batch in enumerate(batches, 1):
                batch_start = time.time()

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
                    docs_done += len(batch)
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
                                   f"{docs_done:,}/{len(texts):,} docs | "
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
