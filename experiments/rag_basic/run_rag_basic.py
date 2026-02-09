#!/usr/bin/env python3
"""
OpenDataCopilot - Évaluation du RAG Basique
============================================

Ce script évalue les performances du RAG basique avec FAISS
sur le dataset de 20 questions annotées.

Usage:
    python -m experiments.rag_basic.run_rag_basic
    python experiments/rag_basic/run_rag_basic.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_basic.rag_basic import BasicRAG
from experiments.rag_basic.config import RAGBasicConfig

# Configuration du logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


def detect_hallucination(response: str, sources: list, num_docs: int) -> dict:
    """
    Détecte les signes d'hallucination dans une réponse RAG.

    Critères adaptés au RAG :
    - Chiffres précis sans correspondance dans les sources
    - Affirmations non supportées par le contexte
    - Réponse détaillée avec 0 documents récupérés

    Args:
        response: Texte de la réponse
        sources: Liste des sources citées
        num_docs: Nombre de documents récupérés

    Returns:
        Dict avec analyse d'hallucination
    """
    indicators = []
    hallucination_score = 0.0

    # Pattern 1: Réponse détaillée sans documents
    if num_docs == 0:
        # Vérifier si la réponse contient des chiffres précis
        numbers = re.findall(r'\d+(?:[,\.]\d+)?', response)
        if len(numbers) > 2:
            indicators.append("Chiffres fournis sans documents de contexte")
            hallucination_score += 0.6

    # Pattern 2: Chiffres très précis
    precise_patterns = [
        r'\d+(?:\s?\d{3})+(?:,\d+)?',  # Grands nombres
        r'\d+(?:,\d+)?\s*%',  # Pourcentages
        r'\d+(?:,\d+)?\s*µg/m[³3]',  # Concentrations pollution
    ]

    for pattern in precise_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches and num_docs > 0:
            # Moins grave si on a des sources
            hallucination_score += 0.1
        elif matches:
            indicators.append(f"Statistique précise sans source: {matches[0]}")
            hallucination_score += 0.3

    # Pattern 3: Reconnaissance d'incertitude (positif)
    uncertainty_phrases = [
        r"(?:selon|d'après)\s+les\s+(?:données|documents)",
        r"les\s+données\s+(?:montrent|indiquent)",
        r"(?:je\s+n'ai\s+pas|pas\s+de)\s+(?:données|informations)",
        r"(?:données\s+)?(?:non\s+)?disponibles",
    ]

    for pattern in uncertainty_phrases:
        if re.search(pattern, response, re.IGNORECASE):
            hallucination_score -= 0.2

    # Pattern 4: Citations de sources
    if re.search(r'\[?\d+\]?|source|selon', response, re.IGNORECASE):
        hallucination_score -= 0.1

    # Normaliser
    hallucination_score = max(0.0, min(1.0, hallucination_score))

    return {
        "is_hallucination": hallucination_score >= 0.4,
        "confidence": hallucination_score,
        "indicators": indicators,
        "has_sources": num_docs > 0,
    }


def evaluate_response_quality(response: str, ground_truth: str | None, sources: list) -> dict:
    """
    Évalue la qualité d'une réponse.

    Args:
        response: Réponse générée
        ground_truth: Réponse attendue (si disponible)
        sources: Sources citées

    Returns:
        Dict avec métriques de qualité
    """
    quality = {
        "has_answer": len(response) > 50,
        "cites_sources": bool(sources),
        "mentions_dates": bool(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}/\d{2}/\d{4}', response)),
        "admits_uncertainty": bool(re.search(
            r"ne\s+(?:sais|connais|peux)|pas\s+(?:de\s+)?données|incertain|difficile",
            response, re.IGNORECASE
        )),
    }

    # Score de qualité
    score = 0
    if quality["has_answer"]:
        score += 0.3
    if quality["cites_sources"]:
        score += 0.4
    if quality["mentions_dates"]:
        score += 0.2
    if quality["admits_uncertainty"] and not quality["cites_sources"]:
        score += 0.1  # Bonus si admet ne pas avoir les données

    quality["score"] = score

    return quality


def load_questions(filepath: Path) -> list[dict]:
    """Charge les questions d'évaluation depuis le fichier JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def load_baseline_results(filepath: Path) -> dict | None:
    """Charge les résultats de la baseline pour comparaison."""
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rag(questions: list[dict], rag: BasicRAG) -> list[dict]:
    """
    Évalue le RAG sur toutes les questions.

    Args:
        questions: Liste des questions
        rag: Instance de BasicRAG

    Returns:
        Liste des résultats
    """
    results = []

    for i, q in enumerate(questions, 1):
        question_id = q.get("id", f"q{i}")
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        ground_truth = q.get("ground_truth")
        expected = q.get("expected_behavior", "")

        print(f"\n{'='*60}")
        print(f"[{i}/{len(questions)}] Question {question_id} ({category})")
        print(f"{'='*60}")
        print(f"Q: {question_text[:80]}...")

        # Mesurer le temps total
        start_time = time.time()
        response = rag.query(question_text, top_k=5)
        total_time = (time.time() - start_time) * 1000

        # Analyser la réponse
        answer = response.answer
        sources = response.sources
        documents = response.documents
        num_docs = len(documents)

        # Détecter hallucination
        hallucination = detect_hallucination(answer, sources, num_docs)

        # Évaluer la qualité
        quality = evaluate_response_quality(answer, ground_truth, sources)

        # Afficher un aperçu
        print(f"\nR: {answer[:200]}...")
        print(f"\n📊 Métriques:")
        print(f"   - Temps total: {total_time:.0f}ms")
        print(f"   - Docs récupérés: {num_docs}")
        print(f"   - Score relevance: {response.metadata.get('avg_relevance_score', 0):.2f}")
        print(f"   - Tokens: {response.metadata.get('tokens_used', 0)}")
        print(f"   - Coût: ${response.metadata.get('cost_usd', 0):.6f}")
        print(f"   - Hallucination: {'⚠️ OUI' if hallucination['is_hallucination'] else '✅ NON'}")
        print(f"   - Qualité: {quality['score']:.2f}")

        if sources:
            print(f"   - Sources: {[s.name for s in sources[:3]]}")

        # Stocker le résultat
        result = {
            "id": question_id,
            "question": question_text,
            "category": category,
            "ground_truth": ground_truth,
            "expected_behavior": expected,
            "response": {
                "answer": answer,
                "confidence": response.confidence,
                "num_sources": len(sources),
                "sources": [{"name": s.name, "date": s.date} for s in sources],
            },
            "metrics": {
                "total_time_ms": total_time,
                "retrieval_time_ms": response.metadata.get("retrieval_time_ms", 0),
                "generation_time_ms": response.metadata.get("latency_ms", 0),
                "tokens_used": response.metadata.get("tokens_used", 0),
                "cost_usd": response.metadata.get("cost_usd", 0),
                "num_docs_retrieved": num_docs,
                "avg_relevance_score": response.metadata.get("avg_relevance_score", 0),
            },
            "hallucination_analysis": hallucination,
            "quality_analysis": quality,
        }

        results.append(result)

        # Petite pause
        time.sleep(0.3)

    return results


def generate_report(results: list[dict], rag: BasicRAG, baseline_results: dict | None) -> dict:
    """
    Génère un rapport complet avec comparaison baseline.

    Args:
        results: Résultats de l'évaluation RAG
        rag: Instance pour stats
        baseline_results: Résultats de la baseline (optionnel)

    Returns:
        Rapport complet
    """
    stats = rag.get_stats()

    # Calculs agrégés
    total_questions = len(results)
    total_time = sum(r["metrics"]["total_time_ms"] for r in results)
    total_retrieval = sum(r["metrics"]["retrieval_time_ms"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    total_tokens = sum(r["metrics"]["tokens_used"] for r in results)
    total_docs = sum(r["metrics"]["num_docs_retrieved"] for r in results)

    hallucinations = [r for r in results if r["hallucination_analysis"]["is_hallucination"]]
    with_sources = [r for r in results if r["response"]["num_sources"] > 0]

    avg_quality = sum(r["quality_analysis"]["score"] for r in results) / total_questions
    avg_relevance = sum(r["metrics"]["avg_relevance_score"] for r in results) / total_questions

    # Par catégorie
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "hallucinations": 0, "with_sources": 0, "avg_quality": 0}
        categories[cat]["count"] += 1
        categories[cat]["avg_quality"] += r["quality_analysis"]["score"]
        if r["hallucination_analysis"]["is_hallucination"]:
            categories[cat]["hallucinations"] += 1
        if r["response"]["num_sources"] > 0:
            categories[cat]["with_sources"] += 1

    for cat in categories:
        categories[cat]["avg_quality"] /= categories[cat]["count"]

    # Comparaison avec baseline
    comparison = None
    if baseline_results:
        baseline_summary = baseline_results.get("summary", {})
        comparison = {
            "baseline": {
                "hallucination_rate": baseline_summary.get("hallucination_rate", 0),
                "avg_latency_ms": baseline_summary.get("avg_latency_ms", 0),
                "total_cost_usd": baseline_summary.get("total_cost_usd", 0),
                "responses_with_sources": 0,
                "useful_responses_rate": 0.3,  # Estimation baseline
            },
            "rag_basic": {
                "hallucination_rate": len(hallucinations) / total_questions,
                "avg_latency_ms": total_time / total_questions,
                "total_cost_usd": total_cost,
                "responses_with_sources": len(with_sources) / total_questions,
                "useful_responses_rate": avg_quality,
            },
        }

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": stats.get("model", "gpt-3.5-turbo"),
            "embedding_model": stats.get("embedding_model", "text-embedding-3-small"),
            "rag_type": "basic_faiss",
            "version": "1.0.0",
            "index_size": stats.get("index_size", 0),
        },
        "summary": {
            "total_questions": total_questions,
            "avg_total_time_ms": total_time / total_questions,
            "avg_retrieval_time_ms": total_retrieval / total_questions,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "hallucination_count": len(hallucinations),
            "hallucination_rate": len(hallucinations) / total_questions,
            "responses_with_sources": len(with_sources),
            "sources_rate": len(with_sources) / total_questions,
            "avg_docs_retrieved": total_docs / total_questions,
            "avg_relevance_score": avg_relevance,
            "avg_quality_score": avg_quality,
        },
        "by_category": {
            cat: {
                "count": data["count"],
                "hallucinations": data["hallucinations"],
                "with_sources": data["with_sources"],
                "avg_quality": data["avg_quality"],
            }
            for cat, data in categories.items()
        },
        "comparison_with_baseline": comparison,
        "results": results,
    }

    return report


def print_comparison_table(report: dict) -> None:
    """Affiche un tableau comparatif Baseline vs RAG."""
    comparison = report.get("comparison_with_baseline")
    summary = report["summary"]

    print("\n" + "=" * 70)
    print("📊 COMPARAISON BASELINE vs RAG BASIQUE")
    print("=" * 70)

    if comparison:
        baseline = comparison["baseline"]
        rag = comparison["rag_basic"]

        print(f"\n{'Métrique':<30} {'Baseline':>15} {'RAG Basic':>15}")
        print("-" * 60)
        print(f"{'Taux hallucination':<30} {baseline['hallucination_rate']*100:>14.1f}% {rag['hallucination_rate']*100:>14.1f}%")
        print(f"{'Réponses avec sources':<30} {baseline['responses_with_sources']*100:>14.1f}% {rag['responses_with_sources']*100:>14.1f}%")
        print(f"{'Taux utilité':<30} {baseline['useful_responses_rate']*100:>14.1f}% {rag['useful_responses_rate']*100:>14.1f}%")
        print(f"{'Temps moyen (ms)':<30} {baseline['avg_latency_ms']:>15.0f} {rag['avg_latency_ms']:>15.0f}")
        print(f"{'Coût total ($)':<30} {baseline['total_cost_usd']:>15.4f} {rag['total_cost_usd']:>15.4f}")
    else:
        print("\n⚠️  Résultats baseline non disponibles pour comparaison")
        print(f"\n{'Métrique':<30} {'RAG Basic':>15}")
        print("-" * 45)
        print(f"{'Taux hallucination':<30} {summary['hallucination_rate']*100:>14.1f}%")
        print(f"{'Réponses avec sources':<30} {summary['sources_rate']*100:>14.1f}%")
        print(f"{'Score qualité moyen':<30} {summary['avg_quality_score']:>14.2f}")
        print(f"{'Temps moyen (ms)':<30} {summary['avg_total_time_ms']:>15.0f}")
        print(f"{'Coût total ($)':<30} {summary['total_cost_usd']:>15.4f}")


def print_summary(report: dict) -> None:
    """Affiche un résumé formaté du rapport."""
    summary = report["summary"]
    categories = report["by_category"]

    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE L'ÉVALUATION RAG BASIQUE")
    print("=" * 70)

    print(f"\n🔢 MÉTRIQUES GLOBALES:")
    print(f"   Questions traitées:    {summary['total_questions']}")
    print(f"   Temps moyen:           {summary['avg_total_time_ms']:.0f} ms")
    print(f"   Temps retrieval:       {summary['avg_retrieval_time_ms']:.0f} ms")
    print(f"   Tokens totaux:         {summary['total_tokens']}")
    print(f"   Coût total:            ${summary['total_cost_usd']:.4f}")

    print(f"\n📚 RETRIEVAL:")
    print(f"   Docs récupérés (moy):  {summary['avg_docs_retrieved']:.1f}")
    print(f"   Score relevance (moy): {summary['avg_relevance_score']:.2f}")
    print(f"   Réponses avec sources: {summary['responses_with_sources']}/{summary['total_questions']} ({summary['sources_rate']*100:.0f}%)")

    print(f"\n⚠️  HALLUCINATIONS:")
    print(f"   Nombre:                {summary['hallucination_count']}/{summary['total_questions']}")
    print(f"   Taux:                  {summary['hallucination_rate']*100:.1f}%")

    print(f"\n⭐ QUALITÉ:")
    print(f"   Score moyen:           {summary['avg_quality_score']:.2f}/1.0")

    print(f"\n📁 PAR CATÉGORIE:")
    for cat, data in sorted(categories.items()):
        sources_pct = data['with_sources'] / data['count'] * 100
        print(f"   {cat}:")
        print(f"      - Questions: {data['count']}, Sources: {sources_pct:.0f}%, Qualité: {data['avg_quality']:.2f}")

    # Exemples
    results = report["results"]
    print(f"\n📝 EXEMPLES DE RÉPONSES:")

    # Bonne réponse (avec sources, pas d'hallucination)
    good = [r for r in results if r["response"]["num_sources"] > 0 and not r["hallucination_analysis"]["is_hallucination"]]
    if good:
        ex = good[0]
        print(f"\n   ✅ BONNE RÉPONSE (Q: {ex['id']}):")
        print(f"      Question: {ex['question'][:60]}...")
        print(f"      Sources: {[s['name'] for s in ex['response']['sources'][:2]]}")
        print(f"      Réponse: {ex['response']['answer'][:150]}...")

    # Réponse sans sources
    no_src = [r for r in results if r["response"]["num_sources"] == 0]
    if no_src:
        ex = no_src[0]
        print(f"\n   ⚠️  SANS SOURCES (Q: {ex['id']}):")
        print(f"      Question: {ex['question'][:60]}...")
        print(f"      Réponse: {ex['response']['answer'][:150]}...")

    print("\n" + "=" * 70)


def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("🔬 OpenDataCopilot - Évaluation RAG Basique")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Chemins
    config = RAGBasicConfig()
    questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees.json"
    baseline_path = PROJECT_ROOT / "experiments" / "baseline" / "results" / "baseline_report.json"
    report_path = config.results_dir / "rag_basic_report.json"

    # Vérifier les fichiers
    if not questions_path.exists():
        print(f"\n❌ Fichier de questions non trouvé: {questions_path}")
        return 1

    if not config.index_path.exists():
        print(f"\n❌ Index FAISS non trouvé: {config.index_path}")
        print("   Lancez d'abord: python -m experiments.rag_basic.data_indexer")
        return 1

    # Charger les questions
    print(f"\n📂 Chargement des questions depuis: {questions_path}")
    questions = load_questions(questions_path)
    print(f"   ✅ {len(questions)} questions chargées")

    # Charger les résultats baseline
    baseline_results = load_baseline_results(baseline_path)
    if baseline_results:
        print(f"   ✅ Résultats baseline chargés pour comparaison")

    # Initialiser le RAG
    print("\n🤖 Initialisation du RAG basique...")
    rag = BasicRAG(config)
    rag.initialize()

    # Lancer l'évaluation
    print("\n🚀 Démarrage de l'évaluation...")
    results = evaluate_rag(questions, rag)

    # Générer le rapport
    print("\n📊 Génération du rapport...")
    report = generate_report(results, rag, baseline_results)

    # Sauvegarder
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Rapport sauvegardé: {report_path}")

    # Afficher les résultats
    print_summary(report)
    print_comparison_table(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
