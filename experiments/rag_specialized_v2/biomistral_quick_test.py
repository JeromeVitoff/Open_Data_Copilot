#!/usr/bin/env python3
"""
Test rapide BioMistral-7B vs GPT-3.5-turbo sur questions santé/pollution.

Compare la qualité de génération sur 7 questions représentatives
avec un contexte simplifié (pas de retrieval complet).

Usage:
    python -m experiments.rag_specialized_v2.biomistral_quick_test
"""

import json
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BIOMISTRAL_MODEL = "BioMistral/BioMistral-7B"
GPT_MODEL = "gpt-3.5-turbo"

# ---------------------------------------------------------------------------
# Questions de test avec contexte simplifié
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    {
        "id": 1,
        "category": "sante",
        "query": "Combien d'hospitalisations COVID-19 en réanimation à Paris en mars 2021 ?",
        "context": "Source SPF / Santé Publique France : Département 75 (Paris), 2021-03-15, hospitalisations COVID-19 : 234, dont réanimation : 48. Retours à domicile cumulés : 1 892.",
    },
    {
        "id": 2,
        "category": "sante",
        "query": "Qu'est-ce qu'une infection respiratoire aiguë (IRA) et comment est-elle surveillée ?",
        "context": "IRA (Infections Respiratoires Aiguës) : syndrome associant fièvre et signes respiratoires. Surveillance via réseau OSCOUR (urgences hospitalières) et SOS Médecins. Indicateurs : taux de passages aux urgences pour IRA, taux d'hospitalisations. Données hebdomadaires ODISSE/SPF par région.",
    },
    {
        "id": 3,
        "category": "sante",
        "query": "Quelle est la couverture vaccinale COVID-19 en France à fin 2022 ?",
        "context": "Couverture vaccinale COVID-19 France, décembre 2022 (ODISSE/SPF) : 1ère dose : 80,3%. Schéma primaire complet : 77,8%. 1ère dose rappel : 54,2%. 2ème dose rappel (personnes éligibles) : 32,1%. Données par département disponibles.",
    },
    {
        "id": 4,
        "category": "pollution",
        "query": "Quelles sont les concentrations moyennes de NO2 à Paris en 2021 selon Airparif ?",
        "context": "Airparif 2021, qualité de l'air Île-de-France. Station PA01H (Paris 1er) : NO2 moyenne annuelle 28,4 µg/m³. Station PA18 (Paris 18e, périphérique) : NO2 47,3 µg/m³. Valeur limite européenne : 40 µg/m³/an. Indice ATMO moyen Paris 2021 : Bon à Moyen.",
    },
    {
        "id": 5,
        "category": "pollution",
        "query": "Qu'est-ce que les particules PM2.5 et quels sont leurs effets sur la santé ?",
        "context": "PM2.5 : particules fines de diamètre aérodynamique inférieur à 2,5 µm. Pénètrent profondément dans les poumons et la circulation sanguine. Effets : aggravation maladies respiratoires et cardiovasculaires, mortalité prématurée. Valeur guide OMS (2021) : 5 µg/m³/an. Concentration moyenne France 2021 : 9,8 µg/m³.",
    },
    {
        "id": 6,
        "category": "correlation",
        "query": "Peut-on observer une corrélation entre les niveaux de NO2 et les hospitalisations pour IRA en Île-de-France ?",
        "context": "Données disponibles : Airparif NO2 (2020-2022) + ODISSE/SPF hospitalisations IRA (2020-2022). Pic pollution NO2 Paris : janvier 2021 (65 µg/m³, épisode 3 jours). Hospitalisations IRA Île-de-France semaine 03/2021 : +23% vs semaine précédente. Corrélation observée mais multifactorielle (période hivernale, COVID).",
    },
    {
        "id": 7,
        "category": "correlation",
        "query": "Quels polluants sont associés à une augmentation des maladies cardiovasculaires ?",
        "context": "Méta-analyses européennes : PM2.5 et NO2 associés à augmentation risque infarctus du myocarde et AVC. Pour 10 µg/m³ de PM2.5 supplémentaires : +5-15% risque cardiovasculaire (exposition long terme). Données PSAS (Programme de Surveillance Air et Santé) disponibles par ville française.",
    },
]

SYSTEM_PROMPT = (
    "Tu es un assistant expert en santé publique et en qualité de l'air en France. "
    "Tu analyses des données officielles (SPF, Airparif, ODISSE). "
    "Réponds en français, de façon factuelle et précise, en citant les chiffres et sources disponibles. "
    "Si les données sont insuffisantes, indique-le clairement."
)


def generate_gpt(query: str, context: str, client: OpenAI) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0.1,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Données disponibles :\n{context}\n\nQuestion : {query}"},
        ],
    )
    return resp.choices[0].message.content or "", time.time() - t0


def generate_biomistral(
    query: str,
    context: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> tuple[str, float]:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Données disponibles :\n{context}\n\n"
        f"Question : {query}\n\n"
        "Réponse :"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), gen_time


def score_response(resp: str) -> float:
    """Score simple : longueur significative + chiffres + mots-clés qualité."""
    import re
    score = 0.0
    if len(resp) > 80:
        score += 0.3
    if re.search(r'\d+[,.]?\d*\s*(µg|%|mg|personnes|cas|hospitalisations)', resp, re.IGNORECASE):
        score += 0.3
    if re.search(r"(source|selon|d'après|airparif|spf|odisse)", resp, re.IGNORECASE):
        score += 0.2
    if re.search(r'(20\d\d|janvier|mars|semaine)', resp, re.IGNORECASE):
        score += 0.2
    return score


def main() -> int:
    print("=" * 75)
    print("TEST RAPIDE : BioMistral-7B vs GPT-3.5-turbo")
    print("=" * 75)

    # --- Charger BioMistral ---
    print(f"\nChargement {BIOMISTRAL_MODEL}...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_MODEL)
    # Restreindre aux A10 (Ampere) — FlashAttention incompatible avec RTX 2080 Ti (Turing)
    max_memory = {0: "20GiB", 1: "20GiB", 2: "0GiB", 3: "0GiB"}
    bio_model = AutoModelForCausalLM.from_pretrained(
        BIOMISTRAL_MODEL,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    print(f"BioMistral charge en {time.time()-t_load:.1f}s — device: {bio_model.device}")

    client = OpenAI()

    results = []

    for q in TEST_QUESTIONS:
        print(f"\n{'='*75}")
        print(f"[{q['id']}/7] {q['category'].upper()} — {q['query'][:65]}...")
        print(f"{'='*75}")

        # GPT-3.5
        gpt_resp, gpt_time = generate_gpt(q["query"], q["context"], client)
        gpt_score = score_response(gpt_resp)
        print(f"\n[GPT-3.5] ({gpt_time:.1f}s | score={gpt_score:.2f})")
        print(f"  {gpt_resp[:200]}...")

        # BioMistral
        bio_resp, bio_time = generate_biomistral(q["query"], q["context"], bio_model, tokenizer)
        bio_score = score_response(bio_resp)
        print(f"\n[BioMistral] ({bio_time:.1f}s | score={bio_score:.2f})")
        print(f"  {bio_resp[:200]}...")

        winner = "BioMistral" if bio_score > gpt_score else "GPT-3.5" if gpt_score > bio_score else "Egalite"
        print(f"\n→ Vainqueur : {winner}")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "gpt35": {"response": gpt_resp, "time": gpt_time, "score": gpt_score},
            "biomistral": {"response": bio_resp, "time": bio_time, "score": bio_score},
            "winner": winner,
        })

    # --- Récap ---
    results_dir = PROJECT_ROOT / "experiments" / "rag_specialized_v2" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "biomistral_quick_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    avg_gpt_score = sum(r["gpt35"]["score"] for r in results) / len(results)
    avg_bio_score = sum(r["biomistral"]["score"] for r in results) / len(results)
    avg_gpt_time = sum(r["gpt35"]["time"] for r in results) / len(results)
    avg_bio_time = sum(r["biomistral"]["time"] for r in results) / len(results)
    bio_wins = sum(1 for r in results if r["winner"] == "BioMistral")

    print("\n" + "=" * 75)
    print("SYNTHESE")
    print("=" * 75)
    print(f"\n{'Modele':<20} {'Score moy':>10} {'Latence moy':>13} {'Victoires':>10}")
    print("-" * 55)
    print(f"{'GPT-3.5-turbo':<20} {avg_gpt_score:>10.3f} {avg_gpt_time:>12.1f}s {7-bio_wins:>10}/7")
    print(f"{'BioMistral-7B':<20} {avg_bio_score:>10.3f} {avg_bio_time:>12.1f}s {bio_wins:>10}/7")

    delta = avg_bio_score - avg_gpt_score
    print(f"\nBioMistral vs GPT-3.5 : {delta:+.3f} ({delta/avg_gpt_score*100:+.1f}%)")

    if delta > 0.05:
        print("\nCONCLUSION : BioMistral-7B est MEILLEUR → Lancer evaluation complete 70 questions")
        verdict = "biomistral_wins"
    elif delta > -0.05:
        print("\nCONCLUSION : Qualites equivalentes → A toi de choisir selon latence/cout")
        verdict = "equivalent"
    else:
        print("\nCONCLUSION : GPT-3.5 reste meilleur → Conserver architecture actuelle")
        verdict = "gpt_wins"

    print(f"\nResultats detailles : {out_path}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
