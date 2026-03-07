#!/usr/bin/env python3
"""
OpenDataCopilot - Évaluation du RAG Optimisé
=============================================

Évalue le RAG Optimisé (FAISS + BM25 + CrossEncoder) sur les 70 questions
annotées et compare avec le RAG Basique.

Usage:
    python -m experiments.rag_optimized.run_evaluation
    python experiments/rag_optimized/run_evaluation.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_optimized.rag_optimized import OptimizedRAG
from experiments.rag_optimized.config import RAGOptimizedConfig

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


# ─── Fonctions d'évaluation (identiques au RAG basique) ──────────────────────

def detect_hallucination(response: str, sources: list, num_docs: int) -> dict:
    indicators = []
    hallucination_score = 0.0

    if num_docs == 0:
        numbers = re.findall(r'\d+(?:[,\.]\d+)?', response)
        if len(numbers) > 2:
            indicators.append("Chiffres fournis sans documents de contexte")
            hallucination_score += 0.6

    precise_patterns = [
        r'\d+(?:\s?\d{3})+(?:,\d+)?',
        r'\d+(?:,\d+)?\s*%',
        r'\d+(?:,\d+)?\s*µg/m[³3]',
    ]
    for pattern in precise_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches and num_docs > 0:
            hallucination_score += 0.1
        elif matches:
            indicators.append(f"Statistique précise sans source: {matches[0]}")
            hallucination_score += 0.3

    uncertainty_phrases = [
        r"(?:selon|d'après)\s+les\s+(?:données|documents)",
        r"les\s+données\s+(?:montrent|indiquent)",
        r"(?:je\s+n'ai\s+pas|pas\s+de)\s+(?:données|informations)",
        r"(?:données\s+)?(?:non\s+)?disponibles",
    ]
    for pattern in uncertainty_phrases:
        if re.search(pattern, response, re.IGNORECASE):
            hallucination_score -= 0.2

    if re.search(r'\[?\d+\]?|source|selon', response, re.IGNORECASE):
        hallucination_score -= 0.1

    hallucination_score = max(0.0, min(1.0, hallucination_score))

    return {
        "is_hallucination": hallucination_score >= 0.4,
        "confidence": hallucination_score,
        "indicators": indicators,
        "has_sources": num_docs > 0,
    }


def evaluate_response_quality(response: str, ground_truth: str | None, sources: list) -> dict:
    quality = {
        "has_answer": len(response) > 50,
        "cites_sources": bool(sources),
        "mentions_dates": bool(re.search(
            r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}/\d{2}/\d{4}', response
        )),
        "admits_uncertainty": bool(re.search(
            r"ne\s+(?:sais|connais|peux)|pas\s+(?:de\s+)?données|incertain|difficile",
            response, re.IGNORECASE
        )),
    }

    score = 0
    if quality["has_answer"]:
        score += 0.3
    if quality["cites_sources"]:
        score += 0.4
    if quality["mentions_dates"]:
        score += 0.2
    if quality["admits_uncertainty"] and not quality["cites_sources"]:
        score += 0.1

    quality["score"] = score
    return quality


def load_questions(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def evaluate_optimized_rag(questions: list[dict], rag: OptimizedRAG) -> list[dict]:
    """Évalue le RAG optimisé sur toutes les questions."""
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

        start_time = time.time()
        response = rag.query(question_text, top_k=rag.config.rerank_top_k)
        total_time = (time.time() - start_time) * 1000

        answer = response.answer
        sources = response.sources
        documents = response.documents
        num_docs = len(documents)

        hallucination = detect_hallucination(answer, sources, num_docs)
        quality = evaluate_response_quality(answer, ground_truth, sources)

        print(f"\nR: {answer[:200]}...")
        print(f"\n📊 Métriques:")
        print(f"   - Temps total: {total_time:.0f}ms")
        print(f"   - Docs reranqués: {num_docs}")
        print(f"   - Retrieval: {response.metadata.get('retrieval_time_ms', 0):.0f}ms")
        print(f"   - Reranking: {response.metadata.get('rerank_time_ms', 0):.0f}ms")
        print(f"   - Score confiance: {response.metadata.get('avg_relevance_score', 0):.3f}")
        print(f"   - Tokens: {response.metadata.get('tokens_used', 0)}")
        print(f"   - Coût: ${response.metadata.get('cost_usd', 0):.6f}")
        print(f"   - Hallucination: {'⚠️ OUI' if hallucination['is_hallucination'] else '✅ NON'}")
        print(f"   - Qualité: {quality['score']:.2f}")

        if sources:
            print(f"   - Sources: {[s.name for s in sources[:3]]}")

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
                "rerank_time_ms": response.metadata.get("rerank_time_ms", 0),
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
        time.sleep(0.3)

    return results


def generate_report(
    results: list[dict],
    rag: OptimizedRAG,
    basic_report: dict | None = None,
) -> dict:
    """Génère le rapport complet avec comparaison RAG Basic."""
    stats = rag.get_stats()

    total_questions = len(results)
    total_time = sum(r["metrics"]["total_time_ms"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    total_tokens = sum(r["metrics"]["tokens_used"] for r in results)
    total_docs = sum(r["metrics"]["num_docs_retrieved"] for r in results)

    hallucinations = [r for r in results if r["hallucination_analysis"]["is_hallucination"]]
    with_sources = [r for r in results if r["response"]["num_sources"] > 0]

    avg_quality = sum(r["quality_analysis"]["score"] for r in results) / total_questions
    avg_relevance = sum(r["metrics"]["avg_relevance_score"] for r in results) / total_questions
    avg_rerank_t = sum(r["metrics"]["rerank_time_ms"] for r in results) / total_questions
    avg_retrieval_t = sum(r["metrics"]["retrieval_time_ms"] for r in results) / total_questions

    # Par catégorie
    categories: dict = {}
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

    # Comparaison avec RAG Basic
    comparison = None
    if basic_report:
        basic_summary = basic_report.get("summary", {})
        comparison = {
            "rag_basic": {
                "hallucination_rate": basic_summary.get("hallucination_rate", 0),
                "avg_latency_ms": basic_summary.get("avg_total_time_ms", 0),
                "total_cost_usd": basic_summary.get("total_cost_usd", 0),
                "sources_rate": basic_summary.get("sources_rate", 0),
                "avg_quality_score": basic_summary.get("avg_quality_score", 0),
                "avg_relevance_score": basic_summary.get("avg_relevance_score", 0),
            },
            "rag_optimized": {
                "hallucination_rate": len(hallucinations) / total_questions,
                "avg_latency_ms": total_time / total_questions,
                "total_cost_usd": total_cost,
                "sources_rate": len(with_sources) / total_questions,
                "avg_quality_score": avg_quality,
                "avg_relevance_score": avg_relevance,
            },
            "improvements": {
                "hallucination_delta": (
                    basic_summary.get("hallucination_rate", 0)
                    - len(hallucinations) / total_questions
                ),
                "quality_delta": avg_quality - basic_summary.get("avg_quality_score", 0),
                "relevance_delta": avg_relevance - basic_summary.get("avg_relevance_score", 0),
            },
        }

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": stats.get("model", "gpt-3.5-turbo"),
            "embedding_model": stats.get("embedding_model", "text-embedding-3-small"),
            "reranker_model": stats.get("reranker_model"),
            "rag_type": "optimized_hybrid_rerank",
            "version": "1.0.0",
            "index_size": stats.get("index_size", 0),
            "hybrid_alpha": stats.get("hybrid_alpha", 0.6),
            "retrieval_top_k": stats.get("retrieval_top_k", 20),
            "rerank_top_k": stats.get("rerank_top_k", 5),
        },
        "summary": {
            "total_questions": total_questions,
            "avg_total_time_ms": total_time / total_questions,
            "avg_retrieval_time_ms": avg_retrieval_t,
            "avg_rerank_time_ms": avg_rerank_t,
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
        "comparison_with_basic": comparison,
        "results": results,
    }

    return report


def print_comparison_table(report: dict) -> None:
    """Affiche la comparaison RAG Basic vs RAG Optimisé."""
    comparison = report.get("comparison_with_basic")
    summary = report["summary"]

    print("\n" + "=" * 75)
    print("📊 COMPARAISON RAG BASIQUE vs RAG OPTIMISÉ")
    print("=" * 75)

    if comparison:
        basic = comparison["rag_basic"]
        opt = comparison["rag_optimized"]
        improvements = comparison["improvements"]

        print(f"\n{'Métrique':<32} {'RAG Basic':>15} {'RAG Optimisé':>15} {'Delta':>10}")
        print("-" * 72)

        def delta_str(v: float, positive_is_good: bool = True) -> str:
            sign = "+" if v > 0 else ""
            arrow = "↑" if (v > 0) == positive_is_good else "↓"
            return f"{arrow}{sign}{v:.3f}"

        print(f"{'Taux hallucination':<32} {basic['hallucination_rate']*100:>14.1f}% {opt['hallucination_rate']*100:>14.1f}% {delta_str(-improvements['hallucination_delta'], True):>10}")
        print(f"{'Réponses avec sources':<32} {basic['sources_rate']*100:>14.1f}% {opt['sources_rate']*100:>14.1f}%")
        print(f"{'Score qualité moyen':<32} {basic['avg_quality_score']:>15.3f} {opt['avg_quality_score']:>15.3f} {delta_str(improvements['quality_delta']):>10}")
        print(f"{'Score relevance moyen':<32} {basic['avg_relevance_score']:>15.3f} {opt['avg_relevance_score']:>15.3f} {delta_str(improvements['relevance_delta']):>10}")
        print(f"{'Temps moyen (ms)':<32} {basic['avg_latency_ms']:>15.0f} {opt['avg_latency_ms']:>15.0f}")
        print(f"{'Coût total ($)':<32} {basic['total_cost_usd']:>15.4f} {opt['total_cost_usd']:>15.4f}")

        print(f"\n✨ AMÉLIORATIONS:")
        q_delta = improvements['quality_delta']
        r_delta = improvements['relevance_delta']
        h_delta = improvements['hallucination_delta']
        print(f"   Qualité:     {'+'if q_delta>0 else ''}{q_delta*100:.1f}%")
        print(f"   Relevance:   {'+'if r_delta>0 else ''}{r_delta*100:.1f}%")
        print(f"   Hallucin.:   {'+'if h_delta>0 else ''}{h_delta*100:.1f}% (positif = réduction)")
    else:
        print(f"\n{'Métrique':<32} {'RAG Optimisé':>15}")
        print("-" * 47)
        print(f"{'Taux hallucination':<32} {summary['hallucination_rate']*100:>14.1f}%")
        print(f"{'Réponses avec sources':<32} {summary['sources_rate']*100:>14.1f}%")
        print(f"{'Score qualité moyen':<32} {summary['avg_quality_score']:>15.3f}")
        print(f"{'Temps moyen (ms)':<32} {summary['avg_total_time_ms']:>15.0f}")
        print(f"{'Coût total ($)':<32} {summary['total_cost_usd']:>15.4f}")


def print_summary(report: dict) -> None:
    """Affiche le résumé de l'évaluation."""
    summary = report["summary"]
    meta = report["metadata"]
    categories = report["by_category"]

    print("\n" + "=" * 75)
    print("📊 RÉSUMÉ DE L'ÉVALUATION RAG OPTIMISÉ")
    print("=" * 75)

    print(f"\n🏗️  ARCHITECTURE:")
    print(f"   Retrieval:     FAISS (dense) + BM25 (sparse) → top-{meta['retrieval_top_k']}")
    print(f"   Alpha hybride: {meta['hybrid_alpha']} (FAISS) / {1-meta['hybrid_alpha']:.1f} (BM25)")
    print(f"   Reranker:      {meta['reranker_model']} → top-{meta['rerank_top_k']}")
    print(f"   LLM:           {meta['model']}")
    print(f"   Index:         {meta['index_size']:,} documents")

    print(f"\n🔢 MÉTRIQUES GLOBALES:")
    print(f"   Questions traitées:    {summary['total_questions']}")
    print(f"   Temps moyen total:     {summary['avg_total_time_ms']:.0f} ms")
    print(f"   Temps retrieval:       {summary['avg_retrieval_time_ms']:.0f} ms")
    print(f"   Temps reranking:       {summary['avg_rerank_time_ms']:.0f} ms")
    print(f"   Tokens totaux:         {summary['total_tokens']}")
    print(f"   Coût total:            ${summary['total_cost_usd']:.4f}")

    print(f"\n📚 RETRIEVAL & RERANKING:")
    print(f"   Docs reranqués (moy):  {summary['avg_docs_retrieved']:.1f}")
    print(f"   Score relevance (moy): {summary['avg_relevance_score']:.3f}")
    print(f"   Réponses avec sources: {summary['responses_with_sources']}/{summary['total_questions']} ({summary['sources_rate']*100:.0f}%)")

    print(f"\n⚠️  HALLUCINATIONS:")
    print(f"   Nombre:                {summary['hallucination_count']}/{summary['total_questions']}")
    print(f"   Taux:                  {summary['hallucination_rate']*100:.1f}%")

    print(f"\n⭐ QUALITÉ:")
    print(f"   Score moyen:           {summary['avg_quality_score']:.3f}/1.0")

    print(f"\n📁 PAR CATÉGORIE:")
    for cat, data in sorted(categories.items()):
        sources_pct = data['with_sources'] / data['count'] * 100
        print(f"   {cat}:")
        print(f"      - Questions: {data['count']}, Sources: {sources_pct:.0f}%, Qualité: {data['avg_quality']:.3f}")

    print("\n" + "=" * 75)


def main():
    """Point d'entrée principal."""
    print("=" * 75)
    print("🔬 OpenDataCopilot - Évaluation RAG Optimisé")
    print("   (FAISS + BM25 + CrossEncoder)")
    print("=" * 75)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = RAGOptimizedConfig()

    # Questions enrichies (70 questions)
    questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees_enrichi.json"
    if not questions_path.exists():
        # Fallback sur les 20 questions de base
        questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees.json"

    # Rapport RAG Basic pour comparaison
    basic_report_path = (
        PROJECT_ROOT / "experiments" / "rag_basic" / "results"
        / "rag_basic_1222k_enrichi_report.json"
    )

    # Vérifications
    if not questions_path.exists():
        print(f"\n❌ Fichier de questions non trouvé: {questions_path}")
        return 1
    if not config.index_path.exists():
        print(f"\n❌ Index FAISS non trouvé: {config.index_path}")
        print("   Lancez d'abord: python -m experiments.rag_basic.data_indexer")
        return 1

    # Charger les questions
    print(f"\n📂 Questions: {questions_path}")
    questions = load_questions(questions_path)
    print(f"   ✅ {len(questions)} questions chargées")

    # Charger le rapport Basic pour comparaison
    basic_report = None
    if basic_report_path.exists():
        with open(basic_report_path, "r", encoding="utf-8") as f:
            basic_report = json.load(f)
        print(f"   ✅ Rapport RAG Basic chargé pour comparaison")
    else:
        print(f"   ⚠️  Rapport RAG Basic non trouvé (pas de comparaison)")

    # Initialiser le RAG Optimisé
    print("\n🤖 Initialisation du RAG Optimisé...")
    rag = OptimizedRAG(config)
    rag.initialize()

    # Rapport de nom
    n_docs = rag.retriever.index.ntotal if rag.retriever.index else 0
    label = f"_{n_docs//1000}k" if n_docs > 10_000 else ""
    report_path = config.results_dir / f"rag_optimized{label}_enrichi_report.json"

    # Évaluation
    print(f"\n🚀 Démarrage de l'évaluation sur {len(questions)} questions...")
    print(f"   Pipeline: FAISS+BM25 (top-{config.retrieval_top_k}) → CrossEncoder → top-{config.rerank_top_k}")
    results = evaluate_optimized_rag(questions, rag)

    # Rapport
    print("\n📊 Génération du rapport...")
    report = generate_report(results, rag, basic_report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Rapport sauvegardé: {report_path}")

    # Affichage
    print_summary(report)
    print_comparison_table(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
