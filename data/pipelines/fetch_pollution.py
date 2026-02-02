"""
OpenDataCopilot - Pipeline de récupération des données de Pollution
====================================================================

Ce script récupère les données de pollution atmosphérique depuis :
- API Airparif (Île-de-France)
- API OpenAQ (données mondiales)

Les données sont mises en cache pour éviter les appels API répétés.

Usage:
    python -m data.pipelines.fetch_pollution
    python -m data.pipelines.fetch_pollution --source airparif --days 7
    python -m data.pipelines.fetch_pollution --source openaq --city Paris
    python -m data.pipelines.fetch_pollution --test-connection
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

# Charger les variables d'environnement
load_dotenv()

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pollution"
CACHE_DIR = PROJECT_ROOT / ".cache" / "pollution"
METADATA_FILE = DATA_RAW_DIR / "metadata.json"

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# ═══════════════════════════════════════════════════════════════
# Configuration des APIs
# ═══════════════════════════════════════════════════════════════

# Airparif - données via data.gouv.fr et ArcGIS Hub
AIRPARIF_CONFIG = {
    "download_urls": {
        # Indices quotidiens Île-de-France (data.gouv.fr)
        "indices_idf": "https://www.data.gouv.fr/fr/datasets/r/c17f815e-ce06-4fbb-baab-b6519a5409a6",
        # Mesures horaires (ArcGIS Feature Service - GeoJSON)
        "mesures_recentes": "https://services8.arcgis.com/gtmasQsdfwbDAQSQ/arcgis/rest/services/ind_idf_agglo/FeatureServer/0/query?where=1%3D1&outFields=*&f=json",
        # Historique mesures (data.gouv.fr)
        "historique_no2": "https://www.data.gouv.fr/fr/datasets/r/f1a5c3ab-5e6d-4e6b-8c7a-8f9b0c1d2e3f",
    },
    "pollutants": ["no2", "pm25", "pm10", "o3", "so2", "co"],
    "region": "Île-de-France",
}

# OpenAQ utilise maintenant l'API v3
OPENAQ_CONFIG = {
    "base_url": "https://api.openaq.org/v3",
    "endpoints": {
        "measurements": "/measurements",
        "locations": "/locations",
        "latest": "/locations",  # v3 utilise locations pour latest
        "countries": "/countries",
        "parameters": "/parameters",
    },
    "default_country": "FR",
    "pollutants": ["pm25", "pm10", "no2", "o3", "so2", "co"],
}

# Villes françaises avec coordonnées pour OpenAQ
FRENCH_CITIES = {
    "Paris": {"lat": 48.8566, "lon": 2.3522, "radius": 25000},
    "Lyon": {"lat": 45.7640, "lon": 4.8357, "radius": 15000},
    "Marseille": {"lat": 43.2965, "lon": 5.3698, "radius": 15000},
    "Montpellier": {"lat": 43.6108, "lon": 3.8767, "radius": 15000},
    "Bordeaux": {"lat": 44.8378, "lon": -0.5792, "radius": 15000},
    "Toulouse": {"lat": 43.6047, "lon": 1.4442, "radius": 15000},
    "Nice": {"lat": 43.7102, "lon": 7.2620, "radius": 15000},
    "Strasbourg": {"lat": 48.5734, "lon": 7.7521, "radius": 15000},
    "Nantes": {"lat": 47.2184, "lon": -1.5536, "radius": 15000},
    "Lille": {"lat": 50.6292, "lon": 3.0573, "radius": 15000},
}

# ═══════════════════════════════════════════════════════════════
# GÉOD'AIR - API nationale de qualité de l'air
# ═══════════════════════════════════════════════════════════════
GEODAIR_CONFIG = {
    "base_url": "https://www.geodair.fr/api-ext",
    "endpoints": {
        "polluants": "/polluant/export",
        "types_donnees": "/type-donnees/export",
        "statistiques": "/statistique/export",
        "stations": "/station/export",
        "download": "/download",
    },
    # Codes des polluants principaux
    "polluants": {
        "NO2": "03",
        "PM10": "24",
        "PM2.5": "39",
        "O3": "08",
        "SO2": "01",
        "CO": "04",
    },
    # Types de données
    "types_donnees": {
        "horaire": "a1",      # Moyenne horaire
        "journaliere": "a2",   # Moyenne journalière
        "annuelle": "a7",      # Moyenne annuelle
    },
    # Codes départements pour les 10 villes majeures
    "villes": {
        "Paris": {"departement": "75", "region": "11"},
        "Lyon": {"departement": "69", "region": "84"},
        "Marseille": {"departement": "13", "region": "93"},
        "Toulouse": {"departement": "31", "region": "76"},
        "Bordeaux": {"departement": "33", "region": "75"},
        "Lille": {"departement": "59", "region": "32"},
        "Montpellier": {"departement": "34", "region": "76"},
        "Strasbourg": {"departement": "67", "region": "44"},
        "Nantes": {"departement": "44", "region": "52"},
        "Nice": {"departement": "06", "region": "93"},
    },
    # Rate limit: 15 requêtes par heure
    "rate_limit": 15,
    "rate_limit_window": 3600,  # secondes
}


def get_api_key(source: str) -> str | None:
    """Récupère la clé API depuis les variables d'environnement."""
    if source == "airparif":
        return os.getenv("AIRPARIF_API_KEY")
    elif source == "openaq":
        return os.getenv("OPENAQ_API_KEY")
    elif source == "geodair":
        return os.getenv("GEODAIR_API_KEY")
    return None


def load_metadata() -> dict[str, Any]:
    """Charge les métadonnées des téléchargements précédents."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"downloads": {}, "api_calls": [], "last_run": None}


def save_metadata(metadata: dict[str, Any]) -> None:
    """Sauvegarde les métadonnées."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata["last_run"] = datetime.now().isoformat()
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def test_api_connection(source: str) -> tuple[bool, str]:
    """
    Teste la connexion à une API.

    Args:
        source: 'airparif' ou 'openaq'

    Returns:
        Tuple (success, message)
    """
    api_key = get_api_key(source)

    if source == "airparif":
        # Airparif utilise l'Open Data - pas besoin de clé API
        try:
            # Test avec l'API ArcGIS Feature Service
            response = httpx.get(
                AIRPARIF_CONFIG["download_urls"]["mesures_recentes"],
                follow_redirects=True,
                timeout=15.0,
            )
            if response.status_code == 200:
                data = response.json()
                features = data.get("features", [])
                return True, f"Connexion OK - {len(features)} mesures Airparif"
            else:
                return False, f"Erreur HTTP {response.status_code}"
        except httpx.RequestError as e:
            return False, f"Erreur de connexion: {e}"
        except Exception as e:
            return False, f"Erreur: {e}"

    elif source == "openaq":
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        try:
            # Test avec l'API v3
            response = httpx.get(
                f"{OPENAQ_CONFIG['base_url']}/countries",
                params={"limit": 1},
                headers=headers,
                timeout=10.0,
            )
            if response.status_code == 200:
                return True, "Connexion OK - API OpenAQ v3 accessible"
            elif response.status_code == 403:
                return False, "Clé API requise ou invalide"
            else:
                return False, f"Erreur HTTP {response.status_code}"
        except httpx.RequestError as e:
            return False, f"Erreur de connexion: {e}"

    elif source == "geodair":
        if not api_key:
            return False, "Clé API GEODAIR_API_KEY non configurée"

        headers = {
            "accept": "text/csv; charset=UTF-8",
            "apikey": api_key,
            "User-Agent": "OpenDataCopilot/1.0",
        }

        try:
            # Test avec la liste des polluants
            response = httpx.get(
                f"{GEODAIR_CONFIG['base_url']}/polluant/export",
                headers=headers,
                timeout=15.0,
            )
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                return True, f"Connexion OK - {len(lines)-1} polluants disponibles"
            elif response.status_code == 403:
                return False, "Clé API invalide ou non autorisée"
            else:
                return False, f"Erreur HTTP {response.status_code}"
        except httpx.RequestError as e:
            return False, f"Erreur de connexion: {e}"

    return False, f"Source inconnue: {source}"


# ═══════════════════════════════════════════════════════════════
# Fonctions Airparif (via Open Data ArcGIS)
# ═══════════════════════════════════════════════════════════════


def fetch_airparif_indices() -> pd.DataFrame | None:
    """
    Récupère les indices de qualité de l'air Airparif via ArcGIS Feature Service.

    Returns:
        DataFrame avec les indices ou None
    """
    try:
        logger.info("  Téléchargement indices Airparif...")
        response = httpx.get(
            AIRPARIF_CONFIG["download_urls"]["mesures_recentes"],
            follow_redirects=True,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        features = data.get("features", [])
        if not features:
            return None

        # Extraire les attributs de chaque feature
        records = [f.get("attributes", {}) for f in features]
        df = pd.DataFrame(records)

        # Ajouter la date de récupération
        df["fetch_date"] = datetime.now().isoformat()

        return df

    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP Airparif indices: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Erreur Airparif indices: {e}")
        return None


def fetch_airparif_mesures_csv() -> pd.DataFrame | None:
    """
    Récupère les mesures Airparif depuis data.gouv.fr (CSV).

    Returns:
        DataFrame avec les mesures ou None
    """
    try:
        logger.info("  Téléchargement mesures Airparif (data.gouv.fr)...")
        response = httpx.get(
            AIRPARIF_CONFIG["download_urls"]["indices_idf"],
            follow_redirects=True,
            timeout=120.0,
        )
        response.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text), sep=";")
        return df

    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP Airparif mesures: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Erreur Airparif mesures: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# Fonctions OpenAQ (API v3)
# ═══════════════════════════════════════════════════════════════


def fetch_openaq_locations(
    city: str | None = None,
    country: str = "FR",
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """Récupère les stations de mesure OpenAQ (API v3)."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        # API v3 utilise 'iso' pour le code pays
        params = {
            "iso": country,
            "limit": 100,
        }

        # Pour les villes, utiliser les coordonnées avec radius
        if city and city in FRENCH_CITIES:
            coords = FRENCH_CITIES[city]
            params["coordinates"] = f"{coords['lat']},{coords['lon']}"
            params["radius"] = coords["radius"]
            del params["iso"]  # Pas besoin du pays avec les coordonnées

        response = httpx.get(
            f"{OPENAQ_CONFIG['base_url']}/locations",
            headers=headers,
            params=params,
            timeout=30.0,
        )

        if response.status_code != 200:
            logger.debug(f"OpenAQ locations status: {response.status_code}")
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            return None

        df = pd.DataFrame(results)
        return df

    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP OpenAQ locations: {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Erreur OpenAQ locations: {e}")
        return None


def fetch_openaq_measurements(
    city: str | None = None,
    country: str = "FR",
    date_start: datetime | None = None,
    date_end: datetime | None = None,
    parameters: list[str] | None = None,
    api_key: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame | None:
    """
    Récupère les mesures OpenAQ via API v3.

    Args:
        city: Nom de la ville (optionnel)
        country: Code pays (défaut: FR)
        date_start: Date de début
        date_end: Date de fin
        parameters: Liste des polluants
        api_key: Clé API OpenAQ
        limit: Nombre max de résultats par requête

    Returns:
        DataFrame avec les mesures ou None
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    # D'abord récupérer les locations pour cette ville/pays
    locations_df = fetch_openaq_locations(city=city, country=country, api_key=api_key)

    if locations_df is None or len(locations_df) == 0:
        logger.warning(f"Aucune station trouvée pour {city or country}")
        return None

    all_data = []
    location_ids = locations_df["id"].tolist()[:10]  # Limiter à 10 stations

    for loc_id in tqdm(location_ids, desc=f"Stations {city or 'FR'}", leave=False):
        try:
            params = {
                "limit": limit,
            }

            if date_start:
                params["date_from"] = date_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            if date_end:
                params["date_to"] = date_end.strftime("%Y-%m-%dT%H:%M:%SZ")

            response = httpx.get(
                f"{OPENAQ_CONFIG['base_url']}/locations/{loc_id}/measurements",
                headers=headers,
                params=params,
                timeout=60.0,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    for item in data["results"]:
                        item["location_id"] = loc_id
                    all_data.extend(data["results"])

        except Exception as e:
            logger.debug(f"Erreur location {loc_id}: {e}")
            continue

    if not all_data:
        return None

    # Normaliser les données
    records = []
    for item in all_data:
        record = {
            "location_id": item.get("location_id"),
            "parameter": item.get("parameter", {}).get("name") if isinstance(item.get("parameter"), dict) else item.get("parameter"),
            "value": item.get("value"),
            "unit": item.get("unit", {}).get("name") if isinstance(item.get("unit"), dict) else item.get("unit"),
            "datetime_utc": item.get("datetime", {}).get("utc") if isinstance(item.get("datetime"), dict) else item.get("datetime"),
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def fetch_openaq_latest(
    city: str | None = None,
    country: str = "FR",
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Récupère les dernières mesures OpenAQ en utilisant l'endpoint measurements.

    L'API v3 ne fournit pas toujours les données 'latest' dans les locations,
    donc on récupère les mesures récentes et on prend la plus récente par station/polluant.
    """
    # Récupérer les mesures des 7 derniers jours
    date_end = datetime.now()
    date_start = date_end - timedelta(days=7)

    df_measurements = fetch_openaq_measurements(
        city=city,
        country=country,
        date_start=date_start,
        date_end=date_end,
        api_key=api_key,
        limit=1000,
    )

    if df_measurements is None or len(df_measurements) == 0:
        # Fallback: essayer de récupérer les infos des locations
        locations_df = fetch_openaq_locations(city=city, country=country, api_key=api_key)
        if locations_df is None or len(locations_df) == 0:
            return None

        # Extraire les infos de base des locations (sans valeurs)
        records = []
        for _, row in locations_df.iterrows():
            sensors = row.get("sensors", [])
            if isinstance(sensors, list):
                for sensor in sensors:
                    if isinstance(sensor, dict):
                        param = sensor.get("parameter", {})
                        latest = sensor.get("latest", {})
                        record = {
                            "location_id": row.get("id"),
                            "location": row.get("name"),
                            "city": city,
                            "country": country,
                            "parameter": param.get("name") if isinstance(param, dict) else None,
                            "last_value": latest.get("value") if isinstance(latest, dict) else None,
                            "last_updated": latest.get("datetime", {}).get("utc") if isinstance(latest, dict) else None,
                        }
                        records.append(record)

        if not records:
            return None
        return pd.DataFrame(records)

    # Récupérer les noms des locations
    locations_df = fetch_openaq_locations(city=city, country=country, api_key=api_key)
    location_names = {}
    if locations_df is not None:
        for _, row in locations_df.iterrows():
            location_names[row.get("id")] = row.get("name")

    # Prendre la mesure la plus récente par location/parameter
    df_measurements["datetime_utc"] = pd.to_datetime(df_measurements["datetime_utc"], errors="coerce")

    # Grouper et prendre la dernière valeur
    idx = df_measurements.groupby(["location_id", "parameter"])["datetime_utc"].idxmax()
    df_latest = df_measurements.loc[idx].copy()

    # Renommer et réorganiser
    df_latest = df_latest.rename(columns={
        "value": "last_value",
        "datetime_utc": "last_updated"
    })

    # Ajouter les noms de location et la ville
    df_latest["location"] = df_latest["location_id"].map(location_names)
    df_latest["city"] = city
    df_latest["country"] = country

    # Sélectionner les colonnes finales
    columns = ["location_id", "location", "city", "country", "parameter", "last_value", "last_updated"]
    df_latest = df_latest[[c for c in columns if c in df_latest.columns]]

    return df_latest.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# Fonctions GÉOD'AIR (API nationale qualité de l'air)
# ═══════════════════════════════════════════════════════════════

def _geodair_request(endpoint: str, params: dict | None = None, api_key: str | None = None) -> str | None:
    """
    Effectue une requête à l'API GÉOD'AIR.

    Args:
        endpoint: Endpoint de l'API (ex: '/polluant/export')
        params: Paramètres de la requête
        api_key: Clé API

    Returns:
        Contenu de la réponse ou None en cas d'erreur
    """
    if not api_key:
        api_key = get_api_key("geodair")
    if not api_key:
        logger.error("Clé API GEODAIR_API_KEY non configurée")
        return None

    headers = {
        "accept": "text/csv; charset=UTF-8",
        "apikey": api_key,
        "User-Agent": "OpenDataCopilot/1.0",  # Requis par le serveur GÉOD'AIR
    }

    url = f"{GEODAIR_CONFIG['base_url']}{endpoint}"

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=60.0)

        if response.status_code == 200:
            return response.text
        elif response.status_code == 403:
            logger.error("GÉOD'AIR: Clé API invalide ou accès refusé")
            return None
        elif response.status_code == 429:
            logger.error("GÉOD'AIR: Rate limit atteint (429). Limite: 15 requêtes/heure.")
            logger.info("  → Attendez quelques minutes avant de réessayer.")
            return None
        else:
            logger.warning(f"GÉOD'AIR: Erreur HTTP {response.status_code}")
            return None
    except httpx.RequestError as e:
        logger.error(f"GÉOD'AIR: Erreur de connexion - {e}")
        return None


def _geodair_download_file(file_id: str, api_key: str | None = None, max_retries: int = 10) -> str | None:
    """
    Télécharge un fichier généré par GÉOD'AIR.

    L'API génère les fichiers de façon asynchrone, il faut parfois
    attendre plusieurs secondes avant que le fichier soit prêt.

    Args:
        file_id: Identifiant du fichier retourné par /statistique/export
        api_key: Clé API
        max_retries: Nombre max de tentatives

    Returns:
        Contenu CSV du fichier ou None
    """
    if not api_key:
        api_key = get_api_key("geodair")

    headers = {
        "accept": "text/csv; charset=UTF-8",
        "apikey": api_key,
        "User-Agent": "OpenDataCopilot/1.0",
    }

    url = f"{GEODAIR_CONFIG['base_url']}/download"
    import time

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, headers=headers, params={"id": file_id}, timeout=90.0)

            if response.status_code == 200:
                content = response.text.strip()
                if content:
                    # Vérifier que c'est du CSV valide (contient des séparateurs ; et plusieurs lignes)
                    lines = content.split("\n")
                    if len(lines) > 1 and ";" in lines[0]:
                        return response.text
                    elif "error" in content.lower() or "erreur" in content.lower():
                        logger.warning(f"GÉOD'AIR: Réponse d'erreur - {content[:200]}")
                        return None
                    else:
                        # Fichier peut-être pas encore prêt
                        time.sleep(3)
                        continue
                else:
                    # Réponse vide, fichier pas encore prêt
                    time.sleep(3)
                    continue

            elif response.status_code == 429:
                # Rate limit atteint (15 requêtes/heure)
                logger.error("GÉOD'AIR: Rate limit atteint (429). Limite: 15 requêtes/heure.")
                logger.info("  → Réessayez dans quelques minutes ou attendez 1 heure pour le reset complet.")
                return None

            elif response.status_code == 406:
                # Fichier pas encore prêt (réponse documentée)
                time.sleep(3)
                continue

            elif response.status_code == 404:
                # Fichier expiré ou invalide
                logger.warning("GÉOD'AIR: Fichier non trouvé (404)")
                return None

            else:
                time.sleep(2)

        except httpx.RequestError as e:
            logger.debug(f"GÉOD'AIR download error: {e}")
            time.sleep(2)

    logger.warning(f"GÉOD'AIR: Échec après {max_retries} tentatives")
    return None


def fetch_geodair_mesures(
    departement: str | None = None,
    polluant: str = "03",  # NO2 par défaut
    type_donnee: str = "a2",  # Moyenne journalière
    date_debut: datetime | None = None,
    date_fin: datetime | None = None,
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Récupère les mesures de pollution depuis GÉOD'AIR.

    Args:
        departement: Code département (ex: '75' pour Paris)
        polluant: Code polluant ('03'=NO2, '24'=PM10, '39'=PM2.5, '08'=O3)
        type_donnee: Type de statistique ('a1'=horaire, 'a2'=journalière)
        date_debut: Date de début
        date_fin: Date de fin
        api_key: Clé API GÉOD'AIR

    Returns:
        DataFrame avec les mesures ou None
    """
    if not api_key:
        api_key = get_api_key("geodair")

    if date_fin is None:
        date_fin = datetime.now()
    if date_debut is None:
        date_debut = date_fin - timedelta(days=30)

    # Formater les dates
    date_debut_str = date_debut.strftime("%d/%m/%Y 00:00")
    date_fin_str = date_fin.strftime("%d/%m/%Y 23:59")

    params = {
        "date_debut": date_debut_str,
        "date_fin": date_fin_str,
        "type_donnee": type_donnee,
        "polluant": polluant,
    }

    if departement:
        params["departement"] = departement

    # Étape 1: Demander la génération du fichier
    logger.debug(f"GÉOD'AIR: Requête statistique avec params: {params}")
    file_id = _geodair_request("/statistique/export", params=params, api_key=api_key)

    if not file_id:
        logger.warning("GÉOD'AIR: Pas de file_id retourné")
        return None

    file_id = file_id.strip()
    logger.debug(f"GÉOD'AIR: file_id reçu: {file_id[:80]}...")

    # Étape 2: Télécharger le fichier (attendre que la génération soit terminée)
    import time
    time.sleep(5)  # Attendre plus longtemps que le fichier soit généré

    csv_content = _geodair_download_file(file_id, api_key=api_key)

    if not csv_content:
        logger.warning(f"GÉOD'AIR: Impossible de télécharger: {file_id[:60]}...")
        return None

    # Parser le CSV
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content), sep=";", encoding="utf-8-sig")
        return df
    except Exception as e:
        logger.error(f"Erreur parsing CSV GÉOD'AIR: {e}")
        return None


def fetch_geodair_ville(
    ville: str,
    polluants: list[str] | None = None,
    days: int = 30,
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """
    Récupère les mesures de pollution pour une ville.

    Args:
        ville: Nom de la ville (ex: 'Paris', 'Lyon')
        polluants: Liste des codes polluants (défaut: NO2, PM10, PM2.5, O3)
        days: Nombre de jours de données
        api_key: Clé API

    Returns:
        DataFrame avec toutes les mesures ou None
    """
    if ville not in GEODAIR_CONFIG["villes"]:
        logger.warning(f"Ville non supportée: {ville}")
        return None

    ville_config = GEODAIR_CONFIG["villes"][ville]
    departement = ville_config["departement"]

    if polluants is None:
        polluants = ["03", "24", "39", "08"]  # NO2, PM10, PM2.5, O3

    date_fin = datetime.now()
    date_debut = date_fin - timedelta(days=days)

    all_data = []

    for polluant in polluants:
        logger.debug(f"  Récupération {ville} - polluant {polluant}...")

        df = fetch_geodair_mesures(
            departement=departement,
            polluant=polluant,
            date_debut=date_debut,
            date_fin=date_fin,
            api_key=api_key,
        )

        if df is not None and len(df) > 0:
            all_data.append(df)

    if not all_data:
        return None

    # Combiner tous les DataFrames
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def fetch_geodair_all_cities(
    cities: list[str] | None = None,
    polluants: list[str] | None = None,
    days: int = 30,
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Récupère les mesures pour plusieurs villes.

    ATTENTION: Rate limit de 15 requêtes/heure !
    Chaque ville x polluant = 1 requête

    Args:
        cities: Liste des villes (défaut: toutes les 10 villes)
        polluants: Liste des codes polluants
        days: Nombre de jours
        api_key: Clé API

    Returns:
        Dictionnaire {ville: DataFrame}
    """
    if cities is None:
        cities = list(GEODAIR_CONFIG["villes"].keys())

    if polluants is None:
        # Limiter à 2 polluants par défaut pour respecter le rate limit
        polluants = ["03", "39"]  # NO2, PM2.5

    results = {}

    # Calculer le nombre de requêtes nécessaires
    total_requests = len(cities) * len(polluants)
    logger.info(f"GÉOD'AIR: {total_requests} requêtes prévues (limit: 15/heure)")

    if total_requests > 15:
        logger.warning(f"⚠️ Dépassement rate limit prévu! Réduction des villes...")
        # Prioriser les plus grandes villes
        cities = cities[:min(len(cities), 15 // len(polluants))]

    for ville in tqdm(cities, desc="Villes GÉOD'AIR"):
        df = fetch_geodair_ville(
            ville=ville,
            polluants=polluants,
            days=days,
            api_key=api_key,
        )

        if df is not None and len(df) > 0:
            results[ville] = df
            logger.success(f"  {ville}: {len(df)} mesures")
        else:
            logger.warning(f"  {ville}: Aucune donnée")

    return results


# ═══════════════════════════════════════════════════════════════
# Fonctions principales
# ═══════════════════════════════════════════════════════════════


def fetch_pollution_data(
    source: str = "all",
    days: int = 30,
    city: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Récupère les données de pollution.

    Args:
        source: 'airparif', 'openaq', 'geodair' ou 'all'
        days: Nombre de jours de données à récupérer
        city: Ville spécifique (pour OpenAQ/GÉOD'AIR)
        force: Force le re-téléchargement

    Returns:
        Dictionnaire avec les résultats
    """
    results = {
        "success": [],
        "failed": [],
        "total_records": 0,
    }

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()

    date_end = datetime.now()
    date_start = date_end - timedelta(days=days)

    # ─────────────────────────────────────────────────────────────
    # Airparif (via Open Data - pas besoin de clé API)
    # ─────────────────────────────────────────────────────────────
    if source in ("airparif", "all"):
        logger.info("=" * 50)
        logger.info("RÉCUPÉRATION DONNÉES AIRPARIF (Open Data)")
        logger.info("=" * 50)

        # Indices qualité de l'air via ArcGIS
        logger.info("Récupération des indices (ArcGIS)...")
        df_indices = fetch_airparif_indices()
        if df_indices is not None and len(df_indices) > 0:
            filepath = DATA_RAW_DIR / "airparif_indices.csv"
            df_indices.to_csv(filepath, index=False)
            logger.success(f"  Indices: {len(df_indices)} enregistrés")
            results["success"].append({
                "source": "airparif",
                "type": "indices",
                "records": len(df_indices),
                "file": str(filepath),
            })
            results["total_records"] += len(df_indices)
        else:
            logger.warning("  Aucun indice récupéré depuis ArcGIS")

        # Mesures depuis data.gouv.fr
        logger.info("Récupération des mesures (data.gouv.fr)...")
        df_mesures = fetch_airparif_mesures_csv()
        if df_mesures is not None and len(df_mesures) > 0:
            filepath = DATA_RAW_DIR / "airparif_mesures.csv"
            df_mesures.to_csv(filepath, index=False)
            logger.success(f"  Mesures: {len(df_mesures)} enregistrées")
            results["success"].append({
                "source": "airparif",
                "type": "mesures",
                "records": len(df_mesures),
                "file": str(filepath),
            })
            results["total_records"] += len(df_mesures)
        else:
            logger.warning("  Aucune mesure récupérée depuis data.gouv.fr")

    # ─────────────────────────────────────────────────────────────
    # OpenAQ
    # ─────────────────────────────────────────────────────────────
    if source in ("openaq", "all"):
        api_key = get_api_key("openaq")

        logger.info("")
        logger.info("=" * 50)
        logger.info("RÉCUPÉRATION DONNÉES OPENAQ")
        logger.info("=" * 50)

        cities_to_fetch = [city] if city else list(FRENCH_CITIES.keys())[:3]  # Par défaut: 3 villes

        for city_name in cities_to_fetch:
            logger.info(f"\nVille: {city_name}")

            # Mesures historiques
            logger.info(f"  Récupération des mesures ({days} jours)...")
            df_measurements = fetch_openaq_measurements(
                city=city_name,
                date_start=date_start,
                date_end=date_end,
                api_key=api_key,
            )

            if df_measurements is not None and len(df_measurements) > 0:
                filepath = DATA_RAW_DIR / f"openaq_{city_name.lower()}_mesures.csv"
                df_measurements.to_csv(filepath, index=False)
                logger.success(f"    Mesures: {len(df_measurements)} enregistrées")
                results["success"].append({
                    "source": "openaq",
                    "city": city_name,
                    "type": "mesures",
                    "records": len(df_measurements),
                    "file": str(filepath),
                })
                results["total_records"] += len(df_measurements)
            else:
                logger.warning(f"    Aucune mesure pour {city_name}")

            # Dernières mesures
            df_latest = fetch_openaq_latest(city=city_name, api_key=api_key)
            if df_latest is not None and len(df_latest) > 0:
                filepath = DATA_RAW_DIR / f"openaq_{city_name.lower()}_latest.csv"
                df_latest.to_csv(filepath, index=False)
                logger.success(f"    Dernières mesures: {len(df_latest)}")

    # ─────────────────────────────────────────────────────────────
    # GÉOD'AIR (API nationale - RECOMMANDÉ)
    # ─────────────────────────────────────────────────────────────
    if source in ("geodair", "all"):
        api_key = get_api_key("geodair")

        if not api_key:
            logger.warning("Clé API GEODAIR_API_KEY non configurée - GÉOD'AIR ignoré")
        else:
            logger.info("")
            logger.info("=" * 50)
            logger.info("RÉCUPÉRATION DONNÉES GÉOD'AIR (National)")
            logger.info("=" * 50)

            # Déterminer les villes à récupérer
            if city and city in GEODAIR_CONFIG["villes"]:
                cities_to_fetch = [city]
            else:
                # Par défaut: 5 grandes villes (pour respecter rate limit)
                cities_to_fetch = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux"]

            # Polluants: NO2 et PM2.5 (2 requêtes par ville = 10 requêtes pour 5 villes)
            polluants = ["03", "39"]  # NO2, PM2.5

            for ville in cities_to_fetch:
                logger.info(f"\nVille: {ville}")

                df = fetch_geodair_ville(
                    ville=ville,
                    polluants=polluants,
                    days=days,
                    api_key=api_key,
                )

                if df is not None and len(df) > 0:
                    # Sauvegarder le fichier
                    month_str = datetime.now().strftime("%Y-%m")
                    filepath = DATA_RAW_DIR / f"geodair_{ville.lower()}_{month_str}.csv"
                    df.to_csv(filepath, index=False, encoding="utf-8-sig")
                    logger.success(f"  {ville}: {len(df)} mesures sauvegardées")

                    results["success"].append({
                        "source": "geodair",
                        "city": ville,
                        "type": "mesures",
                        "records": len(df),
                        "file": str(filepath),
                    })
                    results["total_records"] += len(df)
                else:
                    logger.warning(f"  {ville}: Aucune donnée")
                    results["failed"].append({
                        "source": "geodair",
                        "city": ville,
                        "reason": "Pas de données",
                    })

    # Mettre à jour les métadonnées
    metadata["downloads"]["pollution"] = {
        "date": datetime.now().isoformat(),
        "date_start": date_start.isoformat(),
        "date_end": date_end.isoformat(),
        "source": source,
        "city": city,
        "total_records": results["total_records"],
    }
    save_metadata(metadata)

    # Résumé
    logger.info("")
    logger.info("=" * 50)
    logger.info("RÉSUMÉ POLLUTION")
    logger.info("=" * 50)
    logger.info(f"  Fichiers créés: {len(results['success'])}")
    logger.info(f"  Total enregistrements: {results['total_records']}")
    if results["failed"]:
        logger.warning(f"  Échecs: {len(results['failed'])}")

    return results


def main() -> dict[str, Any] | None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Récupère les données de pollution depuis GÉOD'AIR, Airparif et OpenAQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m data.pipelines.fetch_pollution --test-connection
  python -m data.pipelines.fetch_pollution --source geodair --city Paris
  python -m data.pipelines.fetch_pollution --source airparif --days 7
  python -m data.pipelines.fetch_pollution --all --days 30

Sources disponibles:
  - geodair  : API nationale GÉOD'AIR (recommandé, 10 villes)
  - airparif : Île-de-France uniquement (données détaillées)
  - openaq   : Données mondiales (limité pour la France)
  - all      : Toutes les sources

Villes GÉOD'AIR: Paris, Lyon, Marseille, Toulouse, Bordeaux, Lille, Montpellier, Strasbourg, Nantes, Nice
        """,
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["airparif", "openaq", "geodair", "all"],
        default="all",
        help="Source de données (défaut: all)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Nombre de jours de données (défaut: 30)",
    )
    parser.add_argument(
        "--city", "-c",
        type=str,
        choices=list(GEODAIR_CONFIG["villes"].keys()),
        help="Ville spécifique (pour GÉOD'AIR/OpenAQ)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force le re-téléchargement",
    )
    parser.add_argument(
        "--test-connection", "-t",
        action="store_true",
        help="Teste uniquement la connexion aux APIs",
    )

    args = parser.parse_args()

    # Test de connexion
    if args.test_connection:
        logger.info("=" * 50)
        logger.info("TEST DE CONNEXION AUX APIs")
        logger.info("=" * 50)

        for source in ["geodair", "airparif", "openaq"]:
            success, message = test_api_connection(source)
            if success:
                logger.success(f"{source.upper()}: {message}")
            else:
                logger.error(f"{source.upper()}: {message}")

        return None

    # Récupération des données
    return fetch_pollution_data(
        source=args.source,
        days=args.days,
        city=args.city,
        force=args.force,
    )


if __name__ == "__main__":
    main()
