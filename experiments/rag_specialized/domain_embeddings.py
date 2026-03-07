"""
Expansion terminologique multi-domaines.

Enrichit les requêtes avec des synonymes et termes scientifiques
spécifiques aux domaines santé et pollution, améliorant le recall
du retrieval sémantique.
"""

# Dictionnaire d'expansion : terme → synonymes/définitions
DOMAIN_TERMS: dict[str, str] = {
    # === SANTÉ ===
    "ira": "infection respiratoire aiguë syndrome grippal fièvre toux",
    "covid": "covid-19 coronavirus sars-cov-2 pandémie contagion",
    "covid-19": "coronavirus sars-cov-2 pandémie hospitalisation réanimation",
    "spf": "santé publique france épidémiologie surveillance nationale",
    "réa": "réanimation soins intensifs patients graves",
    "vax": "vaccination immunisation couverture vaccinale dose",
    "vaccination": "vaccin immunisation couverture protection dose rappel",
    "urgences": "passages urgences sos médecins consultations non programmées",
    "odisse": "observatoire données indicateurs santé surveillance épidémiologique",
    "sursaud": "surveillance sanitaire urgences données hospitalières hebdomadaire",
    "désert médical": "pénurie médecins accessibilité soins démographie médicale",
    "antibiotique": "antibiothérapie antibiorésistance consommation ddd",
    "arbovirose": "dengue chikungunya zika maladie vectorielle moustique",
    "vih": "virus immunodéficience humaine sida dépistage séropositif",
    # === ENVIRONNEMENT ===
    "no2": "dioxyde azote pollution atmosphérique trafic automobile qualité air",
    "pm10": "particules fines dix micromètres pollution atmosphérique",
    "pm2.5": "particules très fines pollution atmosphérique inhalation",
    "pm2,5": "particules très fines pollution atmosphérique inhalation",
    "o3": "ozone pollution photochimique été ensoleillement",
    "so2": "dioxyde soufre pollution industrielle acide sulfureux",
    "co": "monoxyde carbone combustion imparfaite pollution intérieure",
    "airparif": "surveillance qualité air île-de-france réseau mesure pollution",
    "openaq": "données ouvertes qualité air international mesures stations",
    "indice atmo": "indice qualité air bon médiocre mauvais très mauvais",
    "qualité de l'air": "pollution atmosphérique polluants concentration seuil",
    "qualité air": "pollution atmosphérique polluants concentration seuil",
    "pic pollution": "épisode pollution concentration élevée alerte seuil dépassement",
    "µg/m³": "microgramme mètre cube concentration polluant mesure",
    "fond urbain": "station fond urbain pollution de fond ambiante",
    "boulevard périphérique": "station trafic trafic routier axe circulation intense no2",
    # === CORRÉLATIONS ===
    "impact pollution santé": "effet respiratoire cardiovasculaire mortalité hospitalisation",
    "pollution respiratoire": "no2 pm10 bronchite asthme ira infection voies respiratoires",
    "confinement pollution": "confinement réduction trafic baisse no2 amélioration qualité air",
}


def expand_query(query: str, domain: str | None = None) -> str:
    """
    Enrichit la requête avec les synonymes du dictionnaire.

    Args:
        query: Requête originale
        domain: Domaine détecté ('health', 'environment', 'correlation', 'general')

    Returns:
        Requête enrichie (originale + termes supplémentaires)
    """
    q_lower = query.lower()
    additions: list[str] = []

    for term, expansion in DOMAIN_TERMS.items():
        if term in q_lower:
            # Ajouter les expansions non déjà présentes
            for word in expansion.split():
                if word not in q_lower and word not in additions:
                    additions.append(word)

    # Ajouter suffixe contextuel selon domaine
    if domain == "health" and "france" not in q_lower:
        additions.append("france épidémiologie")
    elif domain == "environment" and "france" not in q_lower:
        additions.append("france qualité air mesure")
    elif domain == "correlation":
        additions.extend(["impact santé pollution france"])

    if additions:
        return f"{query} {' '.join(additions[:20])}"  # Limiter la longueur
    return query
