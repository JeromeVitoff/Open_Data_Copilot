#!/usr/bin/env python3
"""
OpenDataCopilot - Évaluation de la Baseline (Sans RAG)
======================================================

Ce script évalue les performances du LLM SANS contexte documentaire.
Objectif: mesurer le taux d'hallucination et établir une référence.

Usage:
    python -m experiments.baseline.run_baseline
    python experiments/baseline/run_baseline.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baseline.baseline_rag import BaselineRAG
from experiments.baseline.config import BaselineConfig


def detect_hallucination(response: str, has_sources: bool) -> dict:
    """
    Détecte les signes d'hallucination dans une réponse.

    Critères de détection:
    1. Chiffres précis (dates, pourcentages, statistiques) sans sources
    2. Affirmations catégoriques sur des données récentes
    3. Citations de sources inexistantes

    Args:
        response: Texte de la réponse du LLM
        has_sources: Si la réponse a des sources documentaires

    Returns:
        Dict avec is_hallucination (bool), confidence (float), indicators (list)
    """
    indicators = []
    hallucination_score = 0.0

    # Pattern 1: Chiffres précis (années, pourcentages, valeurs)
    precise_numbers = re.findall(
        r'\b(?:'
        r'\d{1,3}(?:\s?\d{3})*(?:,\d+)?%?'  # Nombres avec virgule
        r'|\d+(?:\.\d+)?%'  # Pourcentages
        r'|\d{4}'  # Années
        r')\b',
        response
    )

    # Filtrer les années raisonnables (1900-2100) et petits nombres
    significant_numbers = [
        n for n in precise_numbers
        if not (n.isdigit() and 1 <= int(n) <= 31)  # Pas les jours
        and not (n.isdigit() and 1900 <= int(n) <= 2025)  # Années historiques OK
    ]

    if len(significant_numbers) > 3 and not has_sources:
        indicators.append(f"Nombreux chiffres précis sans source: {significant_numbers[:5]}")
        hallucination_score += 0.4

    # Pattern 2: Statistiques précises
    stats_patterns = [
        r'\d+(?:,\d+)?\s*(?:millions?|milliards?|milliers?)',
        r'\d+(?:,\d+)?\s*(?:µg|mg|kg)/m[³3]',
        r'(?:en|depuis|jusqu\'?à)\s*202[4-9]',  # Références à dates futures/récentes
        r'\d+(?:,\d+)?%\s*(?:de|des|du)',
    ]

    for pattern in stats_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches and not has_sources:
            indicators.append(f"Statistique précise détectée: {matches[0]}")
            hallucination_score += 0.2

    # Pattern 3: Affirmations catégoriques
    categorical_phrases = [
        r'exactement\s+\d+',
        r'précisément\s+\d+',
        r'le\s+chiffre\s+(?:est|était)\s+(?:de\s+)?\d+',
        r'selon\s+les\s+(?:dernières\s+)?données',
        r'les\s+statistiques\s+(?:officielles\s+)?(?:montrent|indiquent)',
    ]

    for pattern in categorical_phrases:
        if re.search(pattern, response, re.IGNORECASE):
            indicators.append(f"Affirmation catégorique: {pattern}")
            hallucination_score += 0.3

    # Pattern 4: Reconnaissance d'incertitude (réduit le score)
    uncertainty_phrases = [
        r"je\s+ne\s+(?:sais|connais)\s+pas",
        r"je\s+n'ai\s+pas\s+(?:accès|de\s+données)",
        r"données\s+(?:non\s+disponibles|manquantes)",
        r"difficile\s+(?:de\s+)?(?:dire|affirmer)",
        r"(?:peut|pourrait)\s+varier",
        r"à\s+ma\s+connaissance",
        r"mes\s+(?:connaissances|informations)\s+(?:s'arrêtent|datent)",
    ]

    uncertainty_count = 0
    for pattern in uncertainty_phrases:
        if re.search(pattern, response, re.IGNORECASE):
            uncertainty_count += 1

    if uncertainty_count > 0:
        hallucination_score -= 0.3 * uncertainty_count
        indicators.append(f"Expressions d'incertitude détectées: {uncertainty_count}")

    # Normaliser le score entre 0 et 1
    hallucination_score = max(0.0, min(1.0, hallucination_score))

    # Déterminer si c'est une hallucination
    is_hallucination = hallucination_score >= 0.4

    return {
        "is_hallucination": is_hallucination,
        "confidence": hallucination_score,
        "indicators": indicators,
        "precise_numbers_found": len(significant_numbers),
    }


def load_questions(filepath: Path) -> list[dict]:
    """Charge les questions d'évaluation depuis le fichier JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("questions", [])


def evaluate_baseline(questions: list[dict], baseline: BaselineRAG) -> list[dict]:
    """
    Évalue la baseline sur toutes les questions.

    Args:
        questions: Liste des questions à évaluer
        baseline: Instance de BaselineRAG

    Returns:
        Liste des résultats d'évaluation
    """
    results = []

    for i, q in enumerate(questions, 1):
        question_id = q.get("id", f"q{i}")
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        is_trap = q.get("is_trap_question", False)
        expected = q.get("expected_behavior", "")
        ground_truth = q.get("ground_truth")

        print(f"\n{'='*60}")
        print(f"[{i}/{len(questions)}] Question {question_id} ({category})")
        print(f"{'='*60}")
        print(f"Q: {question_text[:100]}...")

        # Mesurer le temps de réponse
        start_time = time.time()
        response = baseline.query(question_text)
        elapsed_time = (time.time() - start_time) * 1000  # en ms

        # Analyser la réponse
        answer = response.answer
        has_sources = len(response.sources) > 0

        # Détecter hallucination
        hallucination_analysis = detect_hallucination(answer, has_sources)

        # Afficher un aperçu
        print(f"\nR: {answer[:200]}...")
        print(f"\n📊 Métriques:")
        print(f"   - Temps: {elapsed_time:.0f}ms")
        print(f"   - Tokens: {response.metadata.get('tokens_used', 0)}")
        print(f"   - Coût: ${response.metadata.get('cost_usd', 0):.6f}")
        print(f"   - Hallucination: {'⚠️ OUI' if hallucination_analysis['is_hallucination'] else '✅ NON'}")
        print(f"   - Score hallucination: {hallucination_analysis['confidence']:.2f}")

        if hallucination_analysis["indicators"]:
            print(f"   - Indicateurs: {hallucination_analysis['indicators'][:2]}")

        # Stocker le résultat
        result = {
            "id": question_id,
            "question": question_text,
            "category": category,
            "is_trap_question": is_trap,
            "expected_behavior": expected,
            "ground_truth": ground_truth,
            "response": {
                "answer": answer,
                "confidence": response.confidence,
                "has_sources": has_sources,
            },
            "metrics": {
                "latency_ms": elapsed_time,
                "tokens_used": response.metadata.get("tokens_used", 0),
                "input_tokens": response.metadata.get("input_tokens", 0),
                "output_tokens": response.metadata.get("output_tokens", 0),
                "cost_usd": response.metadata.get("cost_usd", 0),
            },
            "hallucination_analysis": hallucination_analysis,
        }

        results.append(result)

        # Petite pause pour éviter rate limit OpenAI
        time.sleep(0.5)

    return results


def generate_report(results: list[dict], baseline: BaselineRAG) -> dict:
    """
    Génère un rapport complet de l'évaluation.

    Args:
        results: Résultats de l'évaluation
        baseline: Instance pour récupérer les stats globales

    Returns:
        Rapport complet au format dict
    """
    stats = baseline.get_stats()

    # Calculs agrégés
    total_questions = len(results)
    total_latency = sum(r["metrics"]["latency_ms"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    total_tokens = sum(r["metrics"]["tokens_used"] for r in results)

    hallucinations = [r for r in results if r["hallucination_analysis"]["is_hallucination"]]
    hallucination_rate = len(hallucinations) / total_questions if total_questions > 0 else 0

    # Par catégorie
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "hallucinations": 0, "total_cost": 0}
        categories[cat]["count"] += 1
        categories[cat]["total_cost"] += r["metrics"]["cost_usd"]
        if r["hallucination_analysis"]["is_hallucination"]:
            categories[cat]["hallucinations"] += 1

    # Questions pièges
    trap_questions = [r for r in results if r["is_trap_question"]]
    trap_hallucinations = [r for r in trap_questions if r["hallucination_analysis"]["is_hallucination"]]

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": stats.get("model", "gpt-3.5-turbo"),
            "rag_type": "baseline_no_rag",
            "version": "1.0.0",
        },
        "summary": {
            "total_questions": total_questions,
            "avg_latency_ms": total_latency / total_questions if total_questions > 0 else 0,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "hallucination_count": len(hallucinations),
            "hallucination_rate": hallucination_rate,
        },
        "by_category": {
            cat: {
                "count": data["count"],
                "hallucinations": data["hallucinations"],
                "hallucination_rate": data["hallucinations"] / data["count"] if data["count"] > 0 else 0,
                "total_cost_usd": data["total_cost"],
            }
            for cat, data in categories.items()
        },
        "trap_questions": {
            "total": len(trap_questions),
            "hallucinations": len(trap_hallucinations),
            "detection_rate": len(trap_hallucinations) / len(trap_questions) if trap_questions else 0,
        },
        "results": results,
    }

    return report


def print_summary(report: dict) -> None:
    """Affiche un résumé formaté du rapport."""
    summary = report["summary"]
    categories = report["by_category"]
    trap = report["trap_questions"]

    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE L'ÉVALUATION BASELINE (SANS RAG)")
    print("=" * 70)

    print(f"\n🔢 MÉTRIQUES GLOBALES:")
    print(f"   Questions traitées:    {summary['total_questions']}")
    print(f"   Temps moyen:           {summary['avg_latency_ms']:.0f} ms")
    print(f"   Tokens totaux:         {summary['total_tokens']}")
    print(f"   Coût total:            ${summary['total_cost_usd']:.4f}")

    print(f"\n⚠️  HALLUCINATIONS:")
    print(f"   Nombre:                {summary['hallucination_count']}/{summary['total_questions']}")
    print(f"   Taux:                  {summary['hallucination_rate']*100:.1f}%")

    print(f"\n📁 PAR CATÉGORIE:")
    for cat, data in categories.items():
        print(f"   {cat}:")
        print(f"      - Questions: {data['count']}")
        print(f"      - Hallucinations: {data['hallucinations']} ({data['hallucination_rate']*100:.0f}%)")

    print(f"\n🎯 QUESTIONS PIÈGES:")
    print(f"   Total:                 {trap['total']}")
    print(f"   Hallucinations:        {trap['hallucinations']}")
    print(f"   Taux détection:        {trap['detection_rate']*100:.1f}%")

    # Exemples de réponses
    results = report["results"]
    print(f"\n📝 EXEMPLES DE RÉPONSES:")

    # Une bonne réponse (pas d'hallucination)
    good_responses = [r for r in results if not r["hallucination_analysis"]["is_hallucination"]]
    if good_responses:
        ex = good_responses[0]
        print(f"\n   ✅ BONNE RÉPONSE (Q: {ex['id']}):")
        print(f"      Question: {ex['question'][:60]}...")
        print(f"      Réponse: {ex['response']['answer'][:150]}...")

    # Une mauvaise réponse (hallucination)
    bad_responses = [r for r in results if r["hallucination_analysis"]["is_hallucination"]]
    if bad_responses:
        ex = bad_responses[0]
        print(f"\n   ⚠️  HALLUCINATION (Q: {ex['id']}):")
        print(f"      Question: {ex['question'][:60]}...")
        print(f"      Réponse: {ex['response']['answer'][:150]}...")
        print(f"      Indicateurs: {ex['hallucination_analysis']['indicators'][:2]}")

    print("\n" + "=" * 70)


def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("🔬 OpenDataCopilot - Évaluation Baseline (Sans RAG)")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Chemins
    questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees.json"
    results_dir = PROJECT_ROOT / "experiments" / "baseline" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "baseline_report.json"

    # Vérifier que le fichier de questions existe
    if not questions_path.exists():
        print(f"\n❌ Fichier de questions non trouvé: {questions_path}")
        return 1

    # Charger les questions
    print(f"\n📂 Chargement des questions depuis: {questions_path}")
    questions = load_questions(questions_path)
    print(f"   ✅ {len(questions)} questions chargées")

    # Initialiser la baseline
    print("\n🤖 Initialisation de la baseline...")
    config = BaselineConfig()
    baseline = BaselineRAG(config)
    baseline.initialize()
    print(f"   ✅ Modèle: {config.model}")
    print(f"   ✅ Temperature: {config.temperature}")

    # Lancer l'évaluation
    print("\n🚀 Démarrage de l'évaluation...")
    results = evaluate_baseline(questions, baseline)

    # Générer le rapport
    print("\n📊 Génération du rapport...")
    report = generate_report(results, baseline)

    # Sauvegarder
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Rapport sauvegardé: {report_path}")

    # Afficher le résumé
    print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
