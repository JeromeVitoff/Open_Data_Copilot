"""
Wrapper API Santé Publique France — données hospitalières récentes.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests


class SPFRealtimeAPI:
    """Client pour les données hospitalisations COVID de data.gouv.fr."""

    # CSV direct depuis data.gouv.fr (jeu maintenu par SPF)
    COVID_HOSP_URL = (
        "https://www.data.gouv.fr/fr/datasets/r/"
        "6fadff46-9efd-4c53-942a-54aca783c30c"
    )

    def __init__(self, timeout: int = 30) -> None:
        self.session = requests.Session()
        self.timeout = timeout

    # ── Récupération ────────────────────────────────────────────────────────

    def get_covid_hospitalizations_recent(
        self,
        days: int = 30,
        department: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retourne les hospitalisations COVID des `days` derniers jours.

        Args:
            days: Fenêtre temporelle en jours.
            department: Code département (ex. '75' pour Paris). None = France entière.

        Returns:
            DataFrame filtré, colonnes : jour, dep, hosp, rea, rad, dc.
            DataFrame vide si l'API est inaccessible.
        """
        try:
            resp = self.session.get(self.COVID_HOSP_URL, timeout=self.timeout)
            resp.raise_for_status()

            df = pd.read_csv(self.COVID_HOSP_URL, sep=";")
            df["jour"] = pd.to_datetime(df["jour"], errors="coerce")
            df = df.dropna(subset=["jour"])

            cutoff = datetime.now() - timedelta(days=days)
            df = df[df["jour"] >= cutoff]

            if department:
                df = df[df["dep"].astype(str) == str(department)]

            print(f"✅ SPF COVID: {len(df)} enregistrements (dep={department or 'tous'})")
            return df.reset_index(drop=True)

        except Exception as exc:
            print(f"❌ Erreur SPF API: {exc}")
            return pd.DataFrame()

    # ── Formatage RAG ────────────────────────────────────────────────────────

    def format_for_rag(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convertit un DataFrame SPF en liste de documents texte pour le RAG.

        Returns:
            Liste de dicts {'text': str, 'metadata': dict}.
        """
        documents = []
        for _, row in df.iterrows():
            date_str = row["jour"].strftime("%Y-%m-%d")
            text = (
                f"Le {date_str}, département {row['dep']} : "
                f"{row.get('hosp', 'N/A')} hospitalisations COVID, "
                f"{row.get('rea', 'N/A')} en réanimation, "
                f"{row.get('dc', 'N/A')} décès."
            )
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": "SPF",
                        "date": date_str,
                        "departement": str(row["dep"]),
                        "type": "covid_hospitalisation",
                        "is_realtime": True,
                    },
                }
            )
        return documents
