"""
Détecte si une question nécessite des données temps réel.
"""
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class TemporalDetector:
    """Détecte les références temporelles dans les questions utilisateur."""

    # Mots-clés indiquant un besoin de données récentes/actuelles
    REALTIME_KEYWORDS = [
        r"\baujourd['\u2019]?hui\b",
        r"\bce jour\b",
        r"\bactuellement\b",
        r"\bmaintenant\b",
        r"\bcette semaine\b",
        r"\bce mois\b",
        r"\bcette ann[eé]e\b",
        r"\br[eé]cemment\b",
        r"\br[eé]cent(e?s?)\b",
        r"\bces derniers jours\b",
        r"\bces derni[eè]res semaines\b",
        r"\bdernier\b",
        r"\bderni[eè]re\b",
        r"\b2024\b",
        r"\b2025\b",
        r"\b2026\b",
    ]

    # Mots-clés indiquant des données historiques
    HISTORICAL_KEYWORDS = [
        r"\b(2019|2020|2021|2022|2023)\b",
        r"\ben \d{4}\b",
        r"\bavant\b",
        r"\bpendant\b",
        r"\blors de\b",
    ]

    def __init__(self) -> None:
        self.realtime_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.REALTIME_KEYWORDS
        ]
        self.historical_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HISTORICAL_KEYWORDS
        ]

    def detect(self, query: str) -> Dict:
        """
        Détecte la temporalité de la question.

        Returns:
            {
                'type': 'realtime' | 'historical' | 'mixed' | 'unspecified',
                'realtime_score': float (0-1),
                'historical_score': float (0-1),
                'time_references': List[str],
                'suggested_timerange': Optional[Tuple[date, date]],
                'needs_realtime_data': bool,
            }
        """
        realtime_matches: List[str] = []
        for pattern in self.realtime_patterns:
            realtime_matches.extend(pattern.findall(query))

        historical_matches: List[str] = []
        for pattern in self.historical_patterns:
            historical_matches.extend(pattern.findall(query))

        realtime_score = min(len(realtime_matches) / 2, 1.0)
        historical_score = min(len(historical_matches) / 2, 1.0)

        if realtime_score > 0.3 and historical_score > 0.3:
            temporal_type = "mixed"
        elif realtime_score > 0.3:
            temporal_type = "realtime"
        elif historical_score > 0.3:
            temporal_type = "historical"
        else:
            temporal_type = "unspecified"

        time_refs = realtime_matches + [
            m if isinstance(m, str) else m[0] for m in historical_matches
        ]

        suggested_range = self._suggest_timerange(query, temporal_type, time_refs)

        return {
            "type": temporal_type,
            "realtime_score": realtime_score,
            "historical_score": historical_score,
            "time_references": time_refs,
            "suggested_timerange": suggested_range,
            "needs_realtime_data": realtime_score > 0.3,
        }

    def _suggest_timerange(
        self, query: str, temp_type: str, refs: List[str]
    ) -> Optional[Tuple]:
        """Suggère une plage temporelle pour la recherche."""
        now = datetime.now()
        q = query.lower()

        if temp_type == "realtime":
            if "aujourd'hui" in q or "ce jour" in q:
                return (now.date(), now.date())
            if "cette semaine" in q:
                start = now - timedelta(days=now.weekday())
                return (start.date(), now.date())
            if "ce mois" in q:
                start = now.replace(day=1)
                return (start.date(), now.date())
            # Par défaut : 7 derniers jours
            return ((now - timedelta(days=7)).date(), now.date())

        if temp_type == "historical":
            years = [
                int(r) for r in refs if isinstance(r, str) and r.isdigit() and 2000 <= int(r) <= 2030
            ]
            if years:
                year = max(years)
                return (
                    datetime(year, 1, 1).date(),
                    datetime(year, 12, 31).date(),
                )

        return None


# ─── Démonstration rapide ────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = TemporalDetector()

    test_cases = [
        "Pollution à Paris aujourd'hui ?",
        "Hospitalisations COVID en mars 2021",
        "Évolution grippe cette semaine",
        "Taux vaccination en France",
        "Qualité air Paris actuellement",
        "Lien pollution santé 2020-2023",
        "Données COVID récentes",
    ]

    print("TEST DÉTECTEUR TEMPORALITÉ")
    print("=" * 70)

    for query in test_cases:
        result = detector.detect(query)
        print(f"\nQuery: {query}")
        print(f"  Type:      {result['type']}")
        print(f"  Score RT:  {result['realtime_score']:.2f}")
        print(f"  Score Hist:{result['historical_score']:.2f}")
        print(f"  Needs RT:  {result['needs_realtime_data']}")
        print(f"  Timerange: {result['suggested_timerange']}")
