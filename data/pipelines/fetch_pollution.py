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
}


def get_api_key(source: str) -> str | None:
    """Récupère la clé API depuis les variables d'environnement."""
    if source == "airparif":
        return os.getenv("AIRPARIF_API_KEY")
    elif source == "openaq":
        return os.getenv("OPENAQ_API_KEY")
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
    """Récupère les dernières mesures OpenAQ via les locations."""
    # En API v3, on utilise les locations qui contiennent les dernières mesures
    locations_df = fetch_openaq_locations(city=city, country=country, api_key=api_key)

    if locations_df is None or len(locations_df) == 0:
        return None

    # Extraire les infos pertinentes
    records = []
    for _, row in locations_df.iterrows():
        sensors = row.get("sensors", [])
        if isinstance(sensors, list):
            for sensor in sensors:
                if isinstance(sensor, dict):
                    record = {
                        "location_id": row.get("id"),
                        "location": row.get("name"),
                        "city": city,
                        "country": country,
                        "parameter": sensor.get("parameter", {}).get("name") if isinstance(sensor.get("parameter"), dict) else None,
                        "last_value": sensor.get("latest", {}).get("value") if isinstance(sensor.get("latest"), dict) else None,
                        "last_updated": sensor.get("latest", {}).get("datetime", {}).get("utc") if isinstance(sensor.get("latest"), dict) else None,
                    }
                    records.append(record)

    if not records:
        return None

    df = pd.DataFrame(records)
    return df


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
        source: 'airparif', 'openaq', ou 'all'
        days: Nombre de jours de données à récupérer
        city: Ville spécifique (pour OpenAQ)
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
        description="Récupère les données de pollution depuis Airparif et OpenAQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m data.pipelines.fetch_pollution --test-connection
  python -m data.pipelines.fetch_pollution --source openaq --city Paris
  python -m data.pipelines.fetch_pollution --source airparif --days 7
  python -m data.pipelines.fetch_pollution --all --days 30

Villes disponibles: Paris, Lyon, Marseille, Montpellier, Bordeaux, Toulouse, Nice, Strasbourg
        """,
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["airparif", "openaq", "all"],
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
        choices=list(FRENCH_CITIES.keys()),
        help="Ville spécifique (pour OpenAQ)",
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

        for source in ["airparif", "openaq"]:
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
