"""
Wrapper API OpenAQ v3 — pollution internationale temps réel.

Authentification : Header X-API-Key (compte gratuit sur openaq.org)
Documentation : https://docs.openaq.org/docs
Licence : CC BY 4.0
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import requests


class OpenAQRealtimeAPI:
    """Client OpenAQ v3 pour les mesures de pollution récentes."""

    BASE_URL = "https://api.openaq.org/v3"

    # Mapping nom de polluant → ID paramètre OpenAQ v3
    PARAMETER_IDS: Dict[str, int] = {
        "pm10":  1,
        "pm25":  2,
        "pm2.5": 2,
        "o3":    3,
        "co":    4,
        "no2":   5,
        "so2":   6,
    }

    # Bounding boxes prédéfinies (lon_min, lat_min, lon_max, lat_max)
    BBOX: Dict[str, str] = {
        "idf":        "1.4,48.1,3.6,49.2",
        "lyon":       "4.7,45.6,5.0,45.9",
        "marseille":  "5.2,43.2,5.6,43.5",
        "toulouse":   "1.3,43.5,1.6,43.7",
        "france":     "-5.5,41.3,10.2,51.2",
    }

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10) -> None:
        self.api_key = api_key or os.getenv("OPENAQ_API_KEY")
        if not self.api_key:
            print("⚠️  OPENAQ_API_KEY non trouvée — les requêtes retourneront 401")

        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})

    # ── Récupération ────────────────────────────────────────────────────────

    def get_latest_measurements(
        self,
        city: str = "Paris",
        parameter: str = "no2",
        country_code: str = "FR",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Retourne les dernières mesures pour la région parisienne.

        Stratégie (2 étapes) :
        1. GET /locations?bbox=IDF  → liste des stations avec datetimeLast.
        2. GET /locations/{id}/latest  → mesures récentes, filtré par polluant.

        Args:
            city: Nom de la ville (pour affichage).
            parameter: Polluant parmi {pm10, pm25, no2, o3, so2, co}.
            country_code: Non utilisé (filtrage bbox prioritaire).
            limit: Nombre max de résultats retournés.

        Returns:
            Liste de dicts avec champs location, date, parameter, value, unit.
        """
        param_key = parameter.lower()
        if param_key not in self.PARAMETER_IDS:
            raise ValueError(
                f"Paramètre invalide : {parameter}. "
                f"Choix : {set(self.PARAMETER_IDS)}"
            )

        from datetime import timedelta

        try:
            # Étape 1 : stations IDF avec données récentes (≤ 30 jours)
            resp = self.session.get(
                f"{self.BASE_URL}/locations",
                params={"bbox": self.BBOX["idf"], "limit": 100},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            stations = resp.json().get("results", [])

            cutoff = datetime.now() - timedelta(days=30)
            recent_stations = [
                s for s in stations
                if ((s.get("datetimeLast") or {}).get("utc", "") or "")[:10]
                >= cutoff.strftime("%Y-%m-%d")
            ]

            # Étape 2 : pour chaque station, extraire les sensor IDs correspondant
            # au polluant (disponibles dans station["sensors"]), puis croiser
            # avec /locations/{id}/latest.
            measurements = []
            for station in recent_stations:
                if len(measurements) >= limit:
                    break

                # IDs sensors du polluant voulu dans cette station
                target_ids = {
                    s["id"] for s in station.get("sensors", [])
                    if s.get("parameter", {}).get("name", "").lower() == param_key
                }
                if not target_ids:
                    continue

                r2 = self.session.get(
                    f"{self.BASE_URL}/locations/{station['id']}/latest",
                    timeout=self.timeout,
                )
                if r2.status_code != 200:
                    continue

                for reading in r2.json().get("results", []):
                    if reading.get("sensorsId") in target_ids:
                        # Enrichir avec infos paramètre depuis station["sensors"]
                        sensor_meta = next(
                            (s for s in station.get("sensors", [])
                             if s["id"] == reading["sensorsId"]),
                            {}
                        )
                        reading["parameter"] = sensor_meta.get("parameter", {})
                        reading["_location_name"] = station.get("name", "Inconnu")
                        measurements.append(reading)

            print(f"✅ OpenAQ v3: {len(measurements)} mesures {parameter} région parisienne")
            return measurements[:limit]

        except Exception as exc:
            print(f"❌ Erreur OpenAQ v3: {exc}")
            return []

    def get_france_locations(
        self,
        bbox: str = "1.4,48.1,3.6,49.2",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Retourne les stations dans une bounding box.

        Args:
            bbox: "lon_min,lat_min,lon_max,lat_max" (défaut = Île-de-France).
            limit: Nombre max de stations.
        """
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/locations",
                params={"bbox": bbox, "limit": limit},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            print(f"✅ OpenAQ v3: {len(results)} stations")
            return results

        except Exception as exc:
            print(f"❌ Erreur locations: {exc}")
            return []

    # ── Formatage RAG ────────────────────────────────────────────────────────

    def format_for_rag(self, measurements: List[Dict]) -> List[Dict]:
        """
        Convertit les mesures OpenAQ v3 (via /locations/{id}/latest) en documents RAG.

        Structure sensor :
        {
            "datetime": {"utc": "...", "local": "..."},
            "value": 12.5,
            "parameter": {"id": 5, "name": "no2", "units": "µg/m³", "displayName": "NO₂"},
            "coordinates": {"latitude": 48.8, "longitude": 2.3},
            "_location_name": "Place de l'Opéra",   # ajouté par get_latest_measurements
        }
        """
        documents = []
        for measure in measurements:
            parameter = measure.get("parameter", {})
            location_name = measure.get("_location_name", "Station inconnue")

            date_raw = measure.get("datetime", {}).get("utc", datetime.now().isoformat())
            date_str = date_raw[:10]

            param_name = parameter.get("name", "N/A")
            param_display = parameter.get("displayName") or param_name.upper()
            unit = parameter.get("units", "")
            value = measure.get("value")

            text = (
                f"Station {location_name} (Île-de-France), "
                f"{date_str}: {param_display} = {value} {unit}"
            )
            documents.append({
                "text": text,
                "metadata": {
                    "source": "OpenAQ",
                    "date": date_str,
                    "location": location_name,
                    "country": "France",
                    "parameter": param_name,
                    "value": value,
                    "unit": unit,
                    "is_realtime": True,
                },
            })
        return documents
