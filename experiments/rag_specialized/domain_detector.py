"""
Détection automatique du domaine d'une question.

Identifie si la question porte sur :
- 'health'       : Santé publique uniquement
- 'environment'  : Pollution / qualité de l'air uniquement
- 'correlation'  : Croisement santé × pollution
- 'general'      : Indéterminé

Utilisé pour adapter le retrieval, les filtres et les prompts.
"""

import re
from dataclasses import dataclass


HEALTH_KEYWORDS = [
    # COVID / pandémie
    "covid", "coronavirus", "sars-cov-2", "pandémie", "épidémie",
    "confinement", "vaccination", "dose", "vaccin", "immunité",
    # Hospitalisation / urgences
    "hospitalisation", "réanimation", "rea", "urgences", "décès",
    "hospitalisé", "retour domicile", "sos médecins",
    # Maladies respiratoires
    "ira", "infection respiratoire", "grippe", "bronchite", "pneumonie",
    # Professionnels de santé
    "médecin", "généraliste", "dermatologue", "ophtalmologue",
    "désert médical", "patient", "spécialiste",
    # Maladies chroniques / autres
    "diabète", "cardiovasculaire", "cardio", "neuro",
    "arbovirose", "dengue", "chikungunya", "zika",
    "vih", "sida", "ist", "dépistage",
    "antibiotique", "antibiorésistance",
    "geste auto-infligé", "suicide", "traumatisme",
    # Indicateurs épidémio
    "taux incidence", "taux positivité", "tx_pos", "tx_incid",
    "cas confirmés", "santé publique", "spf", "odisse", "sursaud",
]

ENV_KEYWORDS = [
    # Polluants
    "no2", "dioxyde d'azote", "pm10", "pm2.5", "pm2,5",
    "particule", "o3", "ozone", "so2", "dioxyde soufre",
    "monoxyde carbone",  # "co" retiré : trop court, matche dans "covid"
    # Qualité air
    "qualité de l'air", "qualité air", "indice atmo",
    "pollution", "polluant", "atmosphérique",
    # Organismes / sources
    "airparif", "openaq",
    # Mesures
    "concentration", "µg/m", "ug/m", "µg/m³",
    "seuil alerte", "pic pollution", "épisode pollution",
    "station mesure", "capteur",
    # Zones
    "boulevard périphérique", "station trafic", "fond urbain",
]

CORRELATION_TRIGGERS = [
    "impact", "lien", "corrélation", "corrélé", "relation",
    "influence", "effet", "conséquence", "cause", "causé",
    "associé", "en même temps", "simultanément",
    "pendant", "durant", "lors de",
    "et la qualité", "et la pollution", "et les hospitalisations",
    "respiratoire", "pulmonaire",  # mot pivot entre les deux domaines
]


@dataclass
class DomainResult:
    domain: str                 # 'health' | 'environment' | 'correlation' | 'general'
    health_score: int
    env_score: int
    corr_score: int
    health_matches: list[str]
    env_matches: list[str]


class DomainDetector:
    """
    Classifie les requêtes par domaine à partir d'une analyse lexicale.

    Méthode :
    1. Compte les mots-clés santé et environnement dans la question
    2. Si les deux domaines sont présents → 'correlation'
    3. Sinon → domaine majoritaire
    """

    @staticmethod
    def _match(kw: str, text: str) -> bool:
        """
        Correspondance robuste : utilise les frontières de mot pour les
        mots-clés courts (≤4 chars) afin d'éviter les faux positifs
        (ex: "co" dans "covid", "rea" dans "réanimation").
        """
        if len(kw) <= 4:
            return bool(re.search(r'(?<![a-zàâäéèêëîïôùûüÿæœ])' + re.escape(kw) + r'(?![a-zàâäéèêëîïôùûüÿæœ])', text))
        return kw in text

    def detect(self, query: str) -> DomainResult:
        q = query.lower()

        health_matches = [kw for kw in HEALTH_KEYWORDS if self._match(kw, q)]
        env_matches = [kw for kw in ENV_KEYWORDS if self._match(kw, q)]
        corr_matches = [kw for kw in CORRELATION_TRIGGERS if self._match(kw, q)]

        h = len(health_matches)
        e = len(env_matches)
        c = len(corr_matches)

        # Corrélation : les deux domaines présents simultanément
        if h > 0 and e > 0:
            domain = "correlation"
        elif h > 0 and c > 0 and e == 0:
            # Trigger corrélation mais sans mots env → santé avec dimension contextuelle
            domain = "health"
        elif h > e:
            domain = "health"
        elif e > h:
            domain = "environment"
        else:
            domain = "general"

        return DomainResult(
            domain=domain,
            health_score=h,
            env_score=e,
            corr_score=c,
            health_matches=health_matches,
            env_matches=env_matches,
        )

    def detect_domain(self, query: str) -> str:
        """Raccourci — retourne uniquement la chaîne domaine."""
        return self.detect(query).domain
