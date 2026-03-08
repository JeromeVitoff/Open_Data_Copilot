#!/usr/bin/env python3
"""
Test comparatif : Embeddings génériques (OpenAI) vs Embeddings médicaux (CamemBERT-bio)

Mesure la capacité de discrimination (gap sim_relevant - sim_irrelevant)
sur des paires de documents santé/pollution françaises.

Usage:
    python -m experiments.rag_specialized_v2.medical_embeddings_test
"""

import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_specialized_v2.medical_embeddings import MedicalEmbeddings

# ---------------------------------------------------------------------------
# Paires de test (query, doc pertinent, doc non pertinent)
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "name": "IRA urgences",
        "query": "Infection respiratoire aiguë passages urgences",
        "doc_relevant": "IRA taux passages urgences SOS médecins semaine 2021 région Île-de-France",
        "doc_irrelevant": "Indice qualité air NO2 Paris Airparif mars 2021",
    },
    {
        "name": "COVID réanimation",
        "query": "Hospitalisation COVID-19 réanimation soins intensifs",
        "doc_relevant": "Hospitalisations COVID soins critiques département 75 janvier 2021",
        "doc_irrelevant": "Vaccination grippe saisonnière couverture vaccinale 2020",
    },
    {
        "name": "Vaccination rappel",
        "query": "Vaccination coronavirus doses rappel couverture",
        "doc_relevant": "Couverture vaccinale COVID-19 3ème dose rappel 2022 ODISSE SPF",
        "doc_irrelevant": "Pollution particules fines PM10 Lyon Rhône-Alpes mesures",
    },
    {
        "name": "NO2 qualité air",
        "query": "Pollution dioxyde azote qualité air station mesure",
        "doc_relevant": "Concentrations NO2 µg/m³ station boulevard périphérique Paris Airparif 2021",
        "doc_irrelevant": "Hospitalisations grippe saisonnière Marseille département 13 2020",
    },
    {
        "name": "Corrélation santé-pollution",
        "query": "Corrélation pollution atmosphérique maladies respiratoires santé",
        "doc_relevant": "Impact exposition pollution NO2 PM10 maladies respiratoires IRA urgences",
        "doc_irrelevant": "Vaccination obligatoire enfants calendrier vaccinal 2022",
    },
    {
        "name": "IST dépistage",
        "query": "Infections sexuellement transmissibles dépistage VIH",
        "doc_relevant": "Dépistages VIH IST département résultats positifs ODISSE SPF",
        "doc_irrelevant": "Airparif mesures O3 ozone été canicule Île-de-France",
    },
    {
        "name": "Légionellose",
        "query": "Légionellose déclarations obligatoires cas",
        "doc_relevant": "Légionellose cas déclarés département mois 2021 SPF ODISSE",
        "doc_irrelevant": "PM2.5 particules fines pollution intérieure extérieure",
    },
]

OPENAI_MODEL = "text-embedding-3-small"
MEDICAL_MODEL = "almanach/camembert-bio-base"


def embed_openai(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Calcule les embeddings OpenAI pour une liste de textes."""
    response = client.embeddings.create(input=texts, model=OPENAI_MODEL)
    return [item.embedding for item in response.data]


def run_test(test_case: dict, generic_embs: dict, medical_emb: MedicalEmbeddings) -> dict:
    """Calcule les similarités pour un cas de test."""
    q, rel, irr = test_case["query"], test_case["doc_relevant"], test_case["doc_irrelevant"]

    # Générique
    g_q = np.array(generic_embs[q])
    g_rel = np.array(generic_embs[rel])
    g_irr = np.array(generic_embs[irr])
    sim_g_rel = float(cosine_similarity([g_q], [g_rel])[0][0])
    sim_g_irr = float(cosine_similarity([g_q], [g_irr])[0][0])

    # Médical
    m_q = medical_emb.embed_query(q)
    m_rel = medical_emb.embed_query(rel)
    m_irr = medical_emb.embed_query(irr)
    sim_m_rel = float(cosine_similarity([m_q], [m_rel])[0][0])
    sim_m_irr = float(cosine_similarity([m_q], [m_irr])[0][0])

    return {
        "name": test_case["name"],
        "generic": {"relevant": sim_g_rel, "irrelevant": sim_g_irr, "gap": sim_g_rel - sim_g_irr},
        "medical": {"relevant": sim_m_rel, "irrelevant": sim_m_irr, "gap": sim_m_rel - sim_m_irr},
        "improvement": (sim_m_rel - sim_m_irr) - (sim_g_rel - sim_g_irr),
    }


def main():
    print("=" * 75)
    print("TEST COMPARATIF : Embeddings Generiques vs Medicaux")
    print("=" * 75)
    print(f"Generique : OpenAI {OPENAI_MODEL}")
    print(f"Medical   : {MEDICAL_MODEL}")
    print()

    # 1. Collecter tous les textes à embedder via OpenAI en un seul batch
    all_texts: list[str] = []
    for tc in TEST_CASES:
        all_texts += [tc["query"], tc["doc_relevant"], tc["doc_irrelevant"]]

    print(f"Calcul embeddings OpenAI ({len(all_texts)} textes)...")
    client = OpenAI()
    raw_embs = embed_openai(all_texts, client)
    generic_embs = {t: e for t, e in zip(all_texts, raw_embs)}
    print("Embeddings OpenAI calcules.")

    # 2. Charger CamemBERT-bio
    print()
    medical_emb = MedicalEmbeddings(model_name=MEDICAL_MODEL)
    print()

    # 3. Calculer les résultats
    results = [run_test(tc, generic_embs, medical_emb) for tc in TEST_CASES]

    # 4. Affichage tableau
    print()
    print("=" * 85)
    print("RESULTATS PAR CAS DE TEST")
    print("=" * 85)
    header = f"{'Cas':<24} {'Modele':<10} {'Sim Pertinent':>14} {'Sim Non-Pert':>13} {'Gap':>7}"
    print(header)
    print("-" * 85)

    for r in results:
        name = r["name"][:23]
        g = r["generic"]
        m = r["medical"]
        imp = r["improvement"]
        print(f"{name:<24} {'Generique':<10} {g['relevant']:>14.3f} {g['irrelevant']:>13.3f} {g['gap']:>+7.3f}")
        print(f"{'':<24} {'Medical':<10} {m['relevant']:>14.3f} {m['irrelevant']:>13.3f} {m['gap']:>+7.3f}")
        better = "+" if imp > 0 else ""
        print(f"{'':<24} {'→ Gain':<10} {'':<14} {'':<13} {imp:>+7.3f} {'✓' if imp > 0 else '✗'}")
        print("-" * 85)

    # 5. Stats globales
    avg_gap_g = np.mean([r["generic"]["gap"] for r in results])
    avg_gap_m = np.mean([r["medical"]["gap"] for r in results])
    avg_imp = avg_gap_m - avg_gap_g
    wins = sum(1 for r in results if r["improvement"] > 0)

    print()
    print("=" * 75)
    print("SYNTHESE GLOBALE")
    print("=" * 75)
    print(f"Gap moyen (Generique OpenAI)    : {avg_gap_g:+.4f}")
    print(f"Gap moyen (Medical CamemBERT)   : {avg_gap_m:+.4f}")
    print(f"Amelioration moyenne            : {avg_imp:+.4f}")
    print(f"Cas ou medical > generique      : {wins}/{len(results)}")
    print()

    if avg_imp > 0.05:
        print("CONCLUSION : CamemBERT-bio montre une amelioration SIGNIFICATIVE.")
        print("  → Re-indexation recommandee avec embeddings medicaux.")
        verdict = "go"
    elif avg_imp > 0:
        print("CONCLUSION : CamemBERT-bio montre une legere amelioration.")
        print("  → Re-indexation peut ameliorer les questions medicales.")
        verdict = "marginal"
    else:
        print("CONCLUSION : CamemBERT-bio n'ameliore PAS la discrimination.")
        print("  → OpenAI embeddings generiques restent competitifs.")
        verdict = "no"

    print("=" * 75)
    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict in ("go", "marginal") else 1)
