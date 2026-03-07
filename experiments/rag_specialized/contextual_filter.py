"""
Filtrage et scoring contextuel multi-domaines.

Filtre et reclasse les documents candidats selon :
- Domaine de la question (santé / pollution / corrélation)
- Sources officielles pertinentes
- Score de diversité (bonus si doc couvre les deux domaines)
"""

from dataclasses import dataclass

from src.core.rag_interface import Document


# Mapping domaine → patterns de sources pertinentes
DOMAIN_SOURCE_PATTERNS: dict[str, list[str]] = {
    "health": [
        "covid", "hospitalisation", "vaccin", "vaccination",
        "ira", "infections-respiratoires", "infections-sexuellement",
        "vih", "arboviroses", "antibiotiques", "gestes-auto",
        "maladies-cardio", "professionnels_sante", "odisse",
        "sursaud", "spf",
    ],
    "environment": [
        "airparif", "openaq", "no2", "pm10", "pm25", "pm2.5",
        "co", "o3", "so2", "qualite_air", "pollution",
    ],
    "correlation": [],   # Toutes sources pertinentes
    "general": [],       # Toutes sources
}

# Mots-clés santé dans le texte du document
HEALTH_TEXT_MARKERS = [
    "hospitalisation", "hospitalisés", "décès", "réanimation",
    "vaccination", "covid", "ira", "urgences", "positivité",
]

# Mots-clés environnement dans le texte du document
ENV_TEXT_MARKERS = [
    "no2", "pm10", "pm2.5", "µg/m", "ozone", "polluant",
    "qualité de l'air", "airparif", "indice atmo",
]


@dataclass
class ScoringResult:
    domain_score: float
    source_match: bool
    has_health_content: bool
    has_env_content: bool


def score_document_for_domain(
    doc: Document,
    domain: str,
    domain_score_weight: float = 0.15,
    diversity_bonus: float = 0.10,
) -> float:
    """
    Calcule un score additionnel basé sur la pertinence domaine du document.

    Args:
        doc: Document candidat
        domain: Domaine détecté ('health', 'environment', 'correlation', 'general')
        domain_score_weight: Poids du boost domaine
        diversity_bonus: Bonus si document couvre santé ET pollution

    Returns:
        Score additionnel [0, 0.3]
    """
    score = 0.0
    text_lower = doc.content.lower()
    source = doc.metadata.get("source", "").lower()

    # Vérifier présence de contenu santé et environnement
    has_health = any(m in text_lower for m in HEALTH_TEXT_MARKERS)
    has_env = any(m in text_lower for m in ENV_TEXT_MARKERS)

    # Source match selon domaine
    patterns = DOMAIN_SOURCE_PATTERNS.get(domain, [])
    source_match = any(p in source for p in patterns) if patterns else True

    if domain == "health":
        if source_match:
            score += domain_score_weight
        if has_health:
            score += 0.03

    elif domain == "environment":
        if source_match:
            score += domain_score_weight
        if has_env:
            score += 0.03

    elif domain == "correlation":
        # Valoriser les docs qui couvrent les deux domaines
        if has_health and has_env:
            score += diversity_bonus
        elif has_health or has_env:
            score += 0.05
        # Toutes sources acceptées, pas de filtre strict

    # Bonus source officielle connue
    official_sources = ["spf", "airparif", "odisse", "sursaud", "openaq"]
    if any(o in source for o in official_sources):
        score += 0.05

    # Bonus récence (données >= 2022)
    date_str = doc.metadata.get("date", "")
    if date_str and len(date_str) >= 4:
        try:
            year = int(date_str[:4])
            if year >= 2022:
                score += 0.03
            elif year >= 2020:
                score += 0.01
        except ValueError:
            pass

    return min(score, 0.35)  # Cap à 0.35 pour ne pas déséquilibrer


def filter_and_score_documents(
    documents: list[Document],
    domain: str,
    domain_score_weight: float = 0.15,
    diversity_bonus: float = 0.10,
    strict: bool = False,
) -> list[Document]:
    """
    Filtre et re-score une liste de documents selon le domaine.

    En mode strict (strict=True), élimine les documents hors-domaine.
    En mode soft (strict=False, défaut), booste les docs pertinents.

    Args:
        documents: Liste de Documents candidats
        domain: Domaine détecté
        domain_score_weight: Poids du boost
        diversity_bonus: Bonus diversité pour corrélations
        strict: Si True, filtre dur ; sinon filtre soft (boost)

    Returns:
        Documents re-scorés et triés (filtrés si strict)
    """
    if domain in ("general", "correlation") or not documents:
        # Mode corrélation/général : scorer sans filtrer
        result = []
        for doc in documents:
            add_score = score_document_for_domain(
                doc, domain, domain_score_weight, diversity_bonus
            )
            reranked_doc = Document(
                content=doc.content,
                metadata=doc.metadata,
                score=(doc.score or 0.0) + add_score,
                doc_id=doc.doc_id,
            )
            result.append(reranked_doc)
        return result

    patterns = DOMAIN_SOURCE_PATTERNS.get(domain, [])
    result = []
    off_domain = []

    for doc in documents:
        source = doc.metadata.get("source", "").lower()
        text_lower = doc.content.lower()

        # Vérification appartenance domaine
        source_match = any(p in source for p in patterns) if patterns else True
        text_has_health = any(m in text_lower for m in HEALTH_TEXT_MARKERS)
        text_has_env = any(m in text_lower for m in ENV_TEXT_MARKERS)

        in_domain = source_match
        if not in_domain:
            # Fallback sur contenu textuel
            if domain == "health" and text_has_health:
                in_domain = True
            elif domain == "environment" and text_has_env:
                in_domain = True

        add_score = score_document_for_domain(
            doc, domain, domain_score_weight, diversity_bonus
        )
        reranked_doc = Document(
            content=doc.content,
            metadata=doc.metadata,
            score=(doc.score or 0.0) + add_score,
            doc_id=doc.doc_id,
        )

        if in_domain:
            result.append(reranked_doc)
        else:
            off_domain.append(reranked_doc)

    if strict:
        return result if result else documents  # Fallback si filtre trop agressif

    # Mode soft : in-domain d'abord, puis out-of-domain en complément
    return result + off_domain
