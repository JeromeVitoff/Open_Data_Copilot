#!/usr/bin/env python3
"""
OpenDataCopilot - Évaluation RAG Spécialisé Multi-Domaines
===========================================================

Évalue le SpecializedRAG sur 70 questions avec analyse par domaine.

Usage:
    python -m experiments.rag_specialized.run_evaluation
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

from experiments.rag_specialized.rag_specialized import SpecializedRAG
from experiments.rag_specialized.config import RAGSpecializedConfig
from experiments.rag_specialized.domain_detector import DomainDetector

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


DOMAIN_LABELS = {
    "health": "Santé",
    "environment": "Pollution",
    "correlation": "Corrélation",
    "general": "Général",
}


def detect_hallucination(response: str, sources: list, num_docs: int) -> dict:
    indicators = []
    score = 0.0

    if num_docs == 0:
        numbers = re.findall(r'\d+(?:[,\.]\d+)?', response)
        if len(numbers) > 2:
            indicators.append("Chiffres sans documents source")
            score += 0.6

    for pattern in [r'\d+(?:\s?\d{3})+', r'\d+(?:,\d+)?\s*%', r'\d+(?:,\d+)?\s*µg/m']:
        if re.findall(pattern, response, re.IGNORECASE) and num_docs == 0:
            indicators.append(f"Stats précises sans source")
            score += 0.3

    for phrase in [r"selon les données", r"les données (montrent|indiquent)",
                   r"(pas de données|données non disponibles)", r"\[\d+\]"]:
        if re.search(phrase, response, re.IGNORECASE):
            score -= 0.2

    return {
        "is_hallucination": max(0.0, min(1.0, score)) >= 0.4,
        "confidence": max(0.0, min(1.0, score)),
        "indicators": indicators,
        "has_sources": num_docs > 0,
    }


def evaluate_response_quality(response: str, ground_truth: str | None, sources: list) -> dict:
    quality = {
        "has_answer": len(response) > 50,
        "cites_sources": bool(sources),
        "mentions_dates": bool(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}/\d{2}/\d{4}', response)),
        "admits_uncertainty": bool(re.search(
            r"ne\s+(sais|connais|peux)|pas\s+(de\s+)?données|insuffisant|incertain",
            response, re.IGNORECASE
        )),
    }
    score = 0.0
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


def evaluate_specialized_rag(questions: list[dict], rag: SpecializedRAG) -> list[dict]:
    detector = DomainDetector()
    results = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q{i}")
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        ground_truth = q.get("ground_truth")

        # Détecter domaine attendu
        detected = detector.detect(question_text)
        detected_domain = detected.domain

        print(f"\n{'='*65}")
        print(f"[{i}/{len(questions)}] {qid} ({category}) → domaine={detected_domain}")
        print(f"{'='*65}")
        print(f"Q: {question_text[:90]}...")

        start_time = time.time()
        response = rag.query(question_text, top_k=rag.config.rerank_top_k)
        total_time = (time.time() - start_time) * 1000

        answer = response.answer
        sources = response.sources
        num_docs = len(response.documents)

        hallucination = detect_hallucination(answer, sources, num_docs)
        quality = evaluate_response_quality(answer, ground_truth, sources)

        print(f"\nR: {answer[:200]}...")
        print(f"\n📊 domaine={response.metadata.get('detected_domain', '?')} | "
              f"temps={total_time:.0f}ms | docs={num_docs} | "
              f"qualité={quality['score']:.2f} | "
              f"hallucin={'⚠️' if hallucination['is_hallucination'] else '✅'}")

        results.append({
            "id": qid,
            "question": question_text,
            "category": category,
            "detected_domain": detected_domain,
            "ground_truth": ground_truth,
            "response": {
                "answer": answer,
                "confidence": response.confidence,
                "num_sources": len(sources),
                "sources": [{"name": s.name, "date": s.date} for s in sources],
                "detected_domain": response.metadata.get("detected_domain", "general"),
            },
            "metrics": {
                "total_time_ms": total_time,
                "retrieval_time_ms": response.metadata.get("retrieval_time_ms", 0),
                "filter_time_ms": response.metadata.get("filter_time_ms", 0),
                "rerank_time_ms": response.metadata.get("rerank_time_ms", 0),
                "generation_time_ms": response.metadata.get("latency_ms", 0),
                "tokens_used": response.metadata.get("tokens_used", 0),
                "cost_usd": response.metadata.get("cost_usd", 0),
                "num_docs_retrieved": num_docs,
                "avg_relevance_score": response.metadata.get("avg_relevance_score", 0),
            },
            "hallucination_analysis": hallucination,
            "quality_analysis": quality,
        })
        time.sleep(0.3)

    return results


def generate_report(
    results: list[dict],
    rag: SpecializedRAG,
    basic_report: dict | None = None,
    optimized_report: dict | None = None,
) -> dict:
    stats = rag.get_stats()
    n = len(results)

    total_time = sum(r["metrics"]["total_time_ms"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    total_tokens = sum(r["metrics"]["tokens_used"] for r in results)
    hallucinations = [r for r in results if r["hallucination_analysis"]["is_hallucination"]]
    with_sources = [r for r in results if r["response"]["num_sources"] > 0]
    avg_quality = sum(r["quality_analysis"]["score"] for r in results) / n
    avg_relevance = sum(r["metrics"]["avg_relevance_score"] for r in results) / n

    # Par domaine détecté
    by_domain: dict = {}
    for r in results:
        domain = r.get("detected_domain", "general")
        if domain not in by_domain:
            by_domain[domain] = {
                "count": 0, "hallucinations": 0,
                "with_sources": 0, "avg_quality": 0.0,
                "avg_time_ms": 0.0, "avg_relevance": 0.0,
            }
        by_domain[domain]["count"] += 1
        by_domain[domain]["avg_quality"] += r["quality_analysis"]["score"]
        by_domain[domain]["avg_time_ms"] += r["metrics"]["total_time_ms"]
        by_domain[domain]["avg_relevance"] += r["metrics"]["avg_relevance_score"]
        if r["hallucination_analysis"]["is_hallucination"]:
            by_domain[domain]["hallucinations"] += 1
        if r["response"]["num_sources"] > 0:
            by_domain[domain]["with_sources"] += 1

    for domain in by_domain:
        c = by_domain[domain]["count"]
        by_domain[domain]["avg_quality"] /= c
        by_domain[domain]["avg_time_ms"] /= c
        by_domain[domain]["avg_relevance"] /= c

    # Par catégorie
    by_category: dict = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"count": 0, "avg_quality": 0.0, "with_sources": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["avg_quality"] += r["quality_analysis"]["score"]
        if r["response"]["num_sources"] > 0:
            by_category[cat]["with_sources"] += 1
    for cat in by_category:
        by_category[cat]["avg_quality"] /= by_category[cat]["count"]

    # Comparaison avec architectures précédentes
    comparison = None
    if basic_report or optimized_report:
        comparison = {
            "rag_specialized": {
                "hallucination_rate": len(hallucinations) / n,
                "avg_latency_ms": total_time / n,
                "total_cost_usd": total_cost,
                "sources_rate": len(with_sources) / n,
                "avg_quality_score": avg_quality,
                "avg_relevance_score": avg_relevance,
            }
        }
        if basic_report:
            bs = basic_report.get("summary", {})
            comparison["rag_basic"] = {
                "hallucination_rate": bs.get("hallucination_rate", 0),
                "avg_latency_ms": bs.get("avg_total_time_ms", 0),
                "total_cost_usd": bs.get("total_cost_usd", 0),
                "sources_rate": bs.get("sources_rate", 0),
                "avg_quality_score": bs.get("avg_quality_score", 0),
                "avg_relevance_score": bs.get("avg_relevance_score", 0),
            }
        if optimized_report:
            os_ = optimized_report.get("summary", {})
            comparison["rag_optimized"] = {
                "hallucination_rate": os_.get("hallucination_rate", 0),
                "avg_latency_ms": os_.get("avg_total_time_ms", 0),
                "total_cost_usd": os_.get("total_cost_usd", 0),
                "sources_rate": os_.get("sources_rate", 0),
                "avg_quality_score": os_.get("avg_quality_score", 0),
                "avg_relevance_score": os_.get("avg_relevance_score", 0),
            }
        if comparison.get("rag_basic"):
            comparison["improvements_vs_basic"] = {
                "quality_delta": avg_quality - comparison["rag_basic"]["avg_quality_score"],
                "hallucination_delta": (
                    comparison["rag_basic"]["hallucination_rate"] - len(hallucinations) / n
                ),
            }
        if comparison.get("rag_optimized"):
            comparison["improvements_vs_optimized"] = {
                "quality_delta": avg_quality - comparison["rag_optimized"]["avg_quality_score"],
                "hallucination_delta": (
                    comparison["rag_optimized"]["hallucination_rate"] - len(hallucinations) / n
                ),
            }

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": stats.get("model"),
            "reranker_model": stats.get("reranker_model"),
            "rag_type": "specialized_multi_domain",
            "version": "1.0.0",
            "index_size": stats.get("index_size", 0),
            "domain_score_weight": stats.get("domain_score_weight"),
            "hybrid_alpha": stats.get("hybrid_alpha"),
            "retrieval_top_k": stats.get("retrieval_top_k"),
            "rerank_top_k": stats.get("rerank_top_k"),
        },
        "summary": {
            "total_questions": n,
            "avg_total_time_ms": total_time / n,
            "avg_retrieval_time_ms": stats.get("avg_retrieval_time_ms", 0),
            "avg_filter_time_ms": stats.get("avg_filter_time_ms", 0),
            "avg_rerank_time_ms": stats.get("avg_rerank_time_ms", 0),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "hallucination_count": len(hallucinations),
            "hallucination_rate": len(hallucinations) / n,
            "responses_with_sources": len(with_sources),
            "sources_rate": len(with_sources) / n,
            "avg_relevance_score": avg_relevance,
            "avg_quality_score": avg_quality,
        },
        "by_domain": by_domain,
        "by_category": by_category,
        "domain_distribution": stats.get("domain_distribution", {}),
        "comparison": comparison,
        "results": results,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]
    meta = report["metadata"]
    by_domain = report.get("by_domain", {})
    comparison = report.get("comparison", {})

    print("\n" + "=" * 75)
    print("📊 RÉSUMÉ RAG SPÉCIALISÉ MULTI-DOMAINES")
    print("=" * 75)
    print(f"\n🏗️  ARCHITECTURE:")
    print(f"   Détection domaine: santé / pollution / corrélation / général")
    print(f"   Expansion terminologique: DOMAIN_TERMS ({meta.get('retrieval_top_k')} candidats)")
    print(f"   Scoring domaine: weight={meta.get('domain_score_weight')}")
    print(f"   Reranker: {meta.get('reranker_model')}")
    print(f"   LLM: {meta.get('model')} + prompts spécialisés")
    print(f"   Index: {meta.get('index_size', 0):,} documents")

    print(f"\n🔢 MÉTRIQUES GLOBALES:")
    print(f"   Questions traitées:   {summary['total_questions']}")
    print(f"   Temps moyen total:    {summary['avg_total_time_ms']:.0f} ms")
    print(f"   Tokens totaux:        {summary['total_tokens']}")
    print(f"   Coût total:           ${summary['total_cost_usd']:.4f}")
    print(f"   Hallucinations:       {summary['hallucination_count']}/{summary['total_questions']} ({summary['hallucination_rate']*100:.1f}%)")
    print(f"   Réponses avec source: {summary['responses_with_sources']}/{summary['total_questions']} ({summary['sources_rate']*100:.0f}%)")
    print(f"   Score qualité moyen:  {summary['avg_quality_score']:.3f}/1.0")

    print(f"\n📁 PAR DOMAINE DÉTECTÉ:")
    for domain, data in sorted(by_domain.items()):
        label = DOMAIN_LABELS.get(domain, domain)
        print(f"   {label} ({data['count']} questions):")
        print(f"      Qualité: {data['avg_quality']:.3f} | "
              f"Temps: {data['avg_time_ms']:.0f}ms | "
              f"Hallucin: {data['hallucinations']}/{data['count']}")

    if comparison:
        print(f"\n{'='*75}")
        print("📊 COMPARAISON TOUTES ARCHITECTURES")
        print(f"{'='*75}")
        print(f"\n{'Architecture':<28} {'Qualité':>8} {'Hallucin':>10} {'Sources':>8} {'Coût':>9} {'Temps':>8}")
        print("-" * 73)

        def row(name: str, d: dict) -> str:
            return (
                f"{name:<28} {d.get('avg_quality_score', 0):>8.3f} "
                f"{d.get('hallucination_rate', 0)*100:>9.1f}% "
                f"{d.get('sources_rate', 0)*100:>7.0f}% "
                f"${d.get('total_cost_usd', 0):>8.3f} "
                f"{d.get('avg_latency_ms', 0):>7.0f}ms"
            )

        if comparison.get("rag_basic"):
            print(row("RAG Basic (1.2M docs)", comparison["rag_basic"]))
        if comparison.get("rag_optimized"):
            print(row("RAG Optimisé v2", comparison["rag_optimized"]))
        print(row("RAG Spécialisé", comparison["rag_specialized"]))

        if comparison.get("improvements_vs_optimized"):
            imp = comparison["improvements_vs_optimized"]
            q = imp["quality_delta"]
            print(f"\n✨ vs RAG Optimisé:")
            print(f"   Qualité: {'+'if q>0 else ''}{q*100:.1f}%")

    print("\n" + "=" * 75)


def main():
    print("=" * 75)
    print("🔬 OpenDataCopilot - Évaluation RAG Spécialisé Multi-Domaines")
    print("=" * 75)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = RAGSpecializedConfig()

    questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees_enrichi.json"
    basic_report_path = (
        PROJECT_ROOT / "experiments" / "rag_basic" / "results"
        / "rag_basic_1222k_enrichi_report.json"
    )
    optimized_report_path = (
        PROJECT_ROOT / "experiments" / "rag_optimized" / "results"
        / "rag_optimized_1222k_enrichi_report.json"
    )

    if not questions_path.exists():
        print(f"❌ Questions non trouvées: {questions_path}")
        return 1
    if not config.index_path.exists():
        print(f"❌ Index FAISS non trouvé: {config.index_path}")
        return 1

    print(f"\n📂 Questions: {questions_path}")
    questions = load_questions(questions_path)
    print(f"   ✅ {len(questions)} questions chargées")

    basic_report = None
    if basic_report_path.exists():
        with open(basic_report_path) as f:
            basic_report = json.load(f)
        print("   ✅ Rapport RAG Basic chargé")

    optimized_report = None
    if optimized_report_path.exists():
        with open(optimized_report_path) as f:
            optimized_report = json.load(f)
        print("   ✅ Rapport RAG Optimisé chargé")

    print("\n🤖 Initialisation du RAG Spécialisé...")
    rag = SpecializedRAG(config)
    rag.initialize()

    n_docs = rag.retriever.index.ntotal if rag.retriever.index else 0
    report_path = config.results_dir / f"rag_specialized_{n_docs//1000}k_enrichi_report.json"

    print(f"\n🚀 Évaluation sur {len(questions)} questions...")
    print(f"   Pipeline: DomainDetect → TermExpand → FAISS+BM25(20) → DomainScore → CrossEncoder(5)")
    results = evaluate_specialized_rag(questions, rag)

    print("\n📊 Génération du rapport...")
    report = generate_report(results, rag, basic_report, optimized_report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"   ✅ Rapport sauvegardé: {report_path}")

    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
