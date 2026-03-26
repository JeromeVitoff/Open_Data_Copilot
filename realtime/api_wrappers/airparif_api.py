"""
Wrapper API Airparif temps réel.
Documentation : https://api.airparif.fr/docs

Authentification : Header X-API-Key
Licence : ODbL
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import requests


class AirparifRealtimeAPI:
    """Client API Airparif pour la pollution Île-de-France temps réel."""

    BASE_URL = "https://api.airparif.fr"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10) -> None:
        self.api_key = api_key or os.getenv("AIRPARIF_API_KEY")
        if not self.api_key:
            print("⚠️  AIRPARIF_API_KEY non trouvée — les requêtes retourneront 401")

        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        })

    # ── Récupération ────────────────────────────────────────────────────────

    def get_current_pollution(
        self,
        city: str = "Paris",
        insee_code: str = "75056",
    ) -> Dict:
        """
        Récupère les prévisions pollution pour une commune.

        Args:
            city: Nom de la ville (pour affichage).
            insee_code: Code INSEE (Paris = 75056, Nanterre = 92050, …).

        Returns:
            Dict avec indice, qualificatifs par polluant, prévisions J+1.
            Dict vide si l'API est inaccessible.
        """
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/indices/prevision/commune",
                params={"insee": insee_code},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            # Format : {"75056": [{"date": "2026-03-26", "indice": ..., ...}, ...]}
            previsions = data.get(insee_code, [])
            if not previsions:
                print(f"⚠️  Airparif : aucune donnée pour {city} ({insee_code})")
                return {}

            today = previsions[0]
            result = {
                "city": city,
                "insee_code": insee_code,
                "date": today.get("date"),
                "indice": today.get("indice"),
                "qualificatifs": {
                    "NO2":  today.get("no2"),
                    "O3":   today.get("o3"),
                    "PM10": today.get("pm10"),
                    "PM2.5": today.get("pm25"),
                    "SO2":  today.get("so2"),
                },
                "source": "Airparif",
                "previsions": previsions,
            }
            print(f"✅ Airparif : données récupérées pour {city}")
            return result

        except requests.exceptions.RequestException as exc:
            print(f"❌ Erreur Airparif API: {exc}")
            return {}
        except Exception as exc:
            print(f"❌ Erreur parsing Airparif: {exc}")
            return {}

    def get_episode_pollution(self) -> Dict:
        """
        Récupère les alertes épisodes de pollution en cours et prévus.

        Returns:
            {"actif": bool, "jour": {...}, "demain": {...}, "message": {...}}
        """
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/episodes/en-cours-et-prevus",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            print("✅ Airparif : épisodes pollution récupérés")
            return resp.json()
        except Exception as exc:
            print(f"❌ Erreur épisodes pollution: {exc}")
            return {}

    def get_bulletin_prevision(self) -> str:
        """Récupère le bulletin de prévision texte des prévisionnistes."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/indices/prevision/bulletin",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            print("✅ Airparif : bulletin récupéré")
            return resp.text if "text/plain" in ct else str(resp.json())
        except Exception as exc:
            print(f"❌ Erreur bulletin: {exc}")
            return ""

    def get_multiple_cities(self, cities: List[tuple]) -> List[Dict]:
        """
        Récupère la pollution pour plusieurs villes.

        Args:
            cities: [(nom_ville, code_insee), …]
        """
        results = []
        for city_name, insee_code in cities:
            data = self.get_current_pollution(city_name, insee_code)
            if data:
                results.append(data)
        return results

    # ── Formatage RAG ────────────────────────────────────────────────────────

    def format_for_rag(self, data: Dict) -> Dict:
        """Convertit la réponse Airparif en document texte pour le RAG."""
        if not data:
            return {}

        date = data.get("date") or datetime.now().strftime("%Y-%m-%d")
        city = data.get("city", "Ville")
        indice = data.get("indice", "N/A")
        qualifs = data.get("qualificatifs", {})

        polluants_desc = [
            f"{pol} {qual}" for pol, qual in qualifs.items() if qual
        ]

        parts = [f"Le {date}, à {city}", f"indice de qualité de l'air : {indice}"]
        if polluants_desc:
            parts.append(", ".join(polluants_desc))

        return {
            "text": ": ".join(parts) + ".",
            "metadata": {
                "source": "Airparif",
                "date": date,
                "city": city,
                "insee_code": data.get("insee_code"),
                "type": "pollution_prevision",
                "is_realtime": True,
                "indice": indice,
            },
        }


# Codes INSEE communes Île-de-France (référence)
CODES_INSEE_IDF: Dict[str, str] = {
    "Paris":        "75056",
    "Nanterre":     "92050",
    "Aubervilliers": "93008",
    "Créteil":      "94028",
    "Argenteuil":   "95018",
    "Versailles":   "78646",
    "Meaux":        "77284",
    "Melun":        "77288",
}
