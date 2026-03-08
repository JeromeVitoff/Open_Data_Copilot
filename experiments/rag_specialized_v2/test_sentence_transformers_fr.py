#!/usr/bin/env python3
"""
Test comparatif sentence-transformers français vs OpenAI.

Évalue la discrimination retrieval (gap sim_pertinent - sim_non_pertinent)
sur 7 paires santé/pollution françaises.

Usage:
    python -m experiments.rag_specialized_v2.test_sentence_transformers_fr
"""

import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Cas de test (identiques au test CamemBERT-bio)
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

# ---------------------------------------------------------------------------
# Modèles à tester
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "OpenAI text-embedding-3-small",
        "key": "openai",
        "type": "api",
        "hf_id": "text-embedding-3-small",
    },
    {
        "name": "MS-MARCO FR (biencoder-camembert)",
        "key": "mmarco_fr",
        "type": "st",
        "hf_id": "antoinelouis/biencoder-camembert-base-mmarco-fr",
    },
    {
        "name": "Sentence-CamemBERT",
        "key": "sentence_camembert",
        "type": "st",
        "hf_id": "dangvantuan/sentence-camembert-base",
    },
    {
        "name": "Solon-embeddings-large",
        "key": "solon",
        "type": "st",
        "hf_id": "OrdalieTech/Solon-embeddings-large-0.1",
    },
]


def embed_openai(texts: list[str], client: OpenAI, model: str) -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def test_model(model_info: dict, client: OpenAI | None = None) -> dict:
    """Calcule les gaps pour un modèle donné."""
    name = model_info["name"]
    print(f"\n  Chargement : {name}")

    try:
        if model_info["type"] == "api":
            all_texts = []
            for tc in TEST_CASES:
                all_texts += [tc["query"], tc["doc_relevant"], tc["doc_irrelevant"]]
            raw = embed_openai(all_texts, client, model_info["hf_id"])
            emb_map = {t: e for t, e in zip(all_texts, raw)}

            gaps = []
            for tc in TEST_CASES:
                q = np.array(emb_map[tc["query"]])
                r = np.array(emb_map[tc["doc_relevant"]])
                ir = np.array(emb_map[tc["doc_irrelevant"]])
                gaps.append(float(cosine_similarity([q], [r])[0][0] - cosine_similarity([q], [ir])[0][0]))

        else:
            st = SentenceTransformer(model_info["hf_id"])
            dim = st.get_sentence_embedding_dimension()
            print(f"  Modele charge (dim={dim})")

            gaps = []
            for tc in TEST_CASES:
                q = st.encode(tc["query"])
                r = st.encode(tc["doc_relevant"])
                ir = st.encode(tc["doc_irrelevant"])
                gaps.append(float(cosine_similarity([q], [r])[0][0] - cosine_similarity([q], [ir])[0][0]))

        avg = float(np.mean(gaps))
        print(f"  Gap moyen : {avg:+.4f}")
        return {"name": name, "gaps": gaps, "avg_gap": avg, "model_info": model_info}

    except Exception as e:
        print(f"  ERREUR : {e}")
        return {"name": name, "gaps": [0.0] * len(TEST_CASES), "avg_gap": 0.0, "error": str(e)}


def main() -> str:
    print("=" * 80)
    print("TEST SENTENCE-TRANSFORMERS FRANÇAIS vs OpenAI")
    print("=" * 80)

    client = OpenAI()
    results: list[dict] = []

    for m in MODELS:
        r = test_model(m, client=client if m["type"] == "api" else None)
        results.append(r)

    # --- Tableau récap ---
    openai_gap = next(r["avg_gap"] for r in results if r["model_info"]["key"] == "openai")

    print("\n" + "=" * 80)
    print("RESULTATS COMPARATIFS")
    print("=" * 80)
    print(f"\n{'Modele':<42} {'Gap moy':>8} {'vs OpenAI':>12}  Verdict")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<42} {'ERREUR':>8} {'—':>12}  FAIL")
            continue
        avg = r["avg_gap"]
        diff = avg - openai_gap
        pct = diff / openai_gap * 100 if openai_gap else 0
        if diff > 0.05:
            verdict = "MEILLEUR"
        elif diff > 0:
            verdict = "Legere hausse"
        elif diff > -0.05:
            verdict = "Equivalent"
        elif diff > -0.10:
            verdict = "Legere baisse"
        else:
            verdict = "PIRE"
        print(f"{r['name']:<42} {avg:>+8.4f} {diff:>+8.4f} ({pct:>+5.1f}%)  {verdict}")

    # --- Détail par cas ---
    print("\n" + "=" * 80)
    print("DETAIL PAR CAS")
    print("=" * 80)
    header = f"{'Cas':<26}" + "".join(f" {r['name'][:14]:>15}" for r in results)
    print(header)
    print("-" * 80)
    for i, tc in enumerate(TEST_CASES):
        row = f"{tc['name']:<26}"
        for r in results:
            if "error" not in r:
                row += f" {r['gaps'][i]:>+15.4f}"
            else:
                row += f" {'ERR':>15}"
        print(row)

    # --- Recommandation ---
    valid = [r for r in results if "error" not in r]
    best = max(valid, key=lambda r: r["avg_gap"])
    best_diff = best["avg_gap"] - openai_gap

    print("\n" + "=" * 80)
    print("RECOMMANDATION FINALE")
    print("=" * 80)
    print(f"\nMeilleur modele : {best['name']}")
    print(f"Gap moyen       : {best['avg_gap']:+.4f}")
    print(f"vs OpenAI       : {best_diff:+.4f} ({best_diff/openai_gap*100:+.1f}%)")
    print()

    if best["model_info"]["key"] == "openai":
        print("VERDICT : OpenAI reste le meilleur. Aucune re-indexation justifiee.")
        print("  Passer directement a BioMistral-7B (Phase 2).")
        verdict = "openai_wins"
    elif best_diff > 0.05:
        print(f"VERDICT : RE-INDEXER avec {best['name']}")
        print(f"  Modele HuggingFace : {best['model_info']['hf_id']}")
        print(f"  Amelioration significative (+{best_diff:.3f}) justifie la re-indexation.")
        verdict = "reindex"
    else:
        print(f"VERDICT : Amelioration marginale ({best_diff:+.3f}). Re-indexation non prioritaire.")
        print("  Passer a BioMistral-7B.")
        verdict = "marginal"

    print("=" * 80)
    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0)
