#!/usr/bin/env python3
"""
OpenDataCopilot - Évaluation RAG Spécialisé Médical v2
=======================================================

Évalue SpecializedMedicalRAG (CamemBERT-bio + CrossEncoder + GPT-3.5)
sur 70 questions et compare avec toutes les architectures précédentes.

Usage:
    python -m experiments.rag_specialized_v2.run_evaluation
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

from experiments.rag_specialized_v2.rag_specialized_medical import SpecializedMedicalRAG
from experiments.rag_specialized_v2.config import RAGSpecializedV2Config

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


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
            indicators.append("Stats précises sans source")
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
        "mentions_dates": bool(re.search(
            r'\d{4}[-/]\d{2}[-/]\d{2}|\d{2}/\d{2}/\d{4}'
            r'|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre'
            r'|\d{4}|\bsemaine\b',
            response, re.IGNORECASE,
        )),
        "admits_uncertainty": bool(re.search(
            r"ne\s+(sais|connais|peux)|pas\s+(de\s+)?données|insuffisant|incertain",
            response, re.IGNORECASE,
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
        return json.load(f).get("questions", [])


def evaluate_rag(questions: list[dict], rag: SpecializedMedicalRAG) -> list[dict]:
    results = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q{i}")
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        ground_truth = q.get("ground_truth")

        print(f"\n{'='*65}")
        print(f"[{i}/{len(questions)}] {qid} ({category})")
        print(f"{'='*65}")
        print(f"Q: {question_text[:90]}...")

        t0 = time.time()
        response = rag.query(question_text, top_k=rag.config.rerank_top_k)
        total_time = (time.time() - t0) * 1000

        answer = response.answer
        sources = response.sources
        num_docs = len(response.documents)

        hallucination = detect_hallucination(answer, sources, num_docs)
        quality = evaluate_response_quality(answer, ground_truth, sources)

        print(f"\nR: {answer[:200]}...")
        print(
            f"\ndocs={num_docs} | temps={total_time:.0f}ms | "
            f"qualite={quality['score']:.2f} | "
            f"hallucin={'WARN' if hallucination['is_hallucination'] else 'OK'}"
        )

        results.append({
            "id": qid,
            "question": question_text,
            "category": category,
            "ground_truth": ground_truth,
            "response": {
                "answer": answer,
                "confidence": response.confidence,
                "num_sources": len(sources),
                "sources": [{"name": s.name, "date": s.date} for s in sources],
                "embedding_model": response.metadata.get("embedding_model", ""),
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
        })
        time.sleep(0.3)

    return results


def generate_report(
    results: list[dict],
    rag: SpecializedMedicalRAG,
    prev_reports: dict | None = None,
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

    summary = {
        "total_questions": n,
        "avg_total_time_ms": total_time / n,
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "hallucination_count": len(hallucinations),
        "hallucination_rate": len(hallucinations) / n,
        "responses_with_sources": len(with_sources),
        "sources_rate": len(with_sources) / n,
        "avg_relevance_score": avg_relevance,
        "avg_quality_score": avg_quality,
    }

    comparison = {
        "rag_specialized_v2": {
            "avg_quality_score": avg_quality,
            "hallucination_rate": len(hallucinations) / n,
            "sources_rate": len(with_sources) / n,
            "total_cost_usd": total_cost,
            "avg_latency_ms": total_time / n,
            "avg_relevance_score": avg_relevance,
        }
    }

    if prev_reports:
        arch_keys = [
            ("rag_basic", "rag_basic"),
            ("rag_optimized", "rag_optimized"),
            ("rag_specialized", "rag_specialized_v1"),
        ]
        for file_key, comp_key in arch_keys:
            rep = prev_reports.get(file_key)
            if rep:
                s = rep.get("summary", {})
                comparison[comp_key] = {
                    "avg_quality_score": s.get("avg_quality_score", 0),
                    "hallucination_rate": s.get("hallucination_rate", 0),
                    "sources_rate": s.get("sources_rate", 0),
                    "total_cost_usd": s.get("total_cost_usd", 0),
                    "avg_latency_ms": s.get("avg_total_time_ms", 0),
                }

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": stats.get("model"),
            "embedding_model": stats.get("embedding_model"),
            "reranker_model": stats.get("reranker_model"),
            "rag_type": "specialized_medical_v2",
            "version": "2.0.0",
            "index_size": stats.get("index_size", 0),
            "hybrid_alpha": stats.get("hybrid_alpha"),
            "retrieval_top_k": stats.get("retrieval_top_k"),
            "rerank_top_k": stats.get("rerank_top_k"),
        },
        "summary": summary,
        "by_category": by_category,
        "comparison": comparison,
        "results": results,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]
    meta = report["metadata"]
    comparison = report.get("comparison", {})

    print("\n" + "=" * 75)
    print("RESUME RAG SPECIALISE MEDICAL v2")
    print("=" * 75)
    print(f"\nARCHITECTURE:")
    print(f"   Embeddings      : {meta.get('embedding_model')}")
    print(f"   Reranker        : {meta.get('reranker_model')}")
    print(f"   LLM             : {meta.get('model')}")
    print(f"   Index           : {meta.get('index_size', 0):,} documents")
    print(f"   alpha hybride   : {meta.get('hybrid_alpha')}")

    print(f"\nMETRIQUES GLOBALES ({summary['total_questions']} questions) :")
    print(f"   Temps moyen     : {summary['avg_total_time_ms']:.0f} ms")
    print(f"   Coût total      : ${summary['total_cost_usd']:.4f}")
    print(f"   Hallucinations  : {summary['hallucination_count']}/{summary['total_questions']} ({summary['hallucination_rate']*100:.1f}%)")
    print(f"   Avec sources    : {summary['responses_with_sources']}/{summary['total_questions']} ({summary['sources_rate']*100:.0f}%)")
    print(f"   Score qualité   : {summary['avg_quality_score']:.3f}/1.0")

    if len(comparison) > 1:
        print(f"\n{'='*75}")
        print("COMPARAISON TOUTES ARCHITECTURES")
        print(f"{'='*75}")
        print(f"\n{'Architecture':<32} {'Qualite':>8} {'Hallucin':>10} {'Sources':>8} {'Cout':>9} {'Temps':>8}")
        print("-" * 75)

        order = [
            ("rag_basic",          "RAG Basic (1.2M docs)"),
            ("rag_optimized",      "RAG Optimise v2"),
            ("rag_specialized_v1", "RAG Specialise v1"),
            ("rag_specialized_v2", "RAG Medical v2 (CamemBERT)"),
        ]
        for key, label in order:
            d = comparison.get(key)
            if not d:
                continue
            print(
                f"{label:<32} {d.get('avg_quality_score', 0):>8.3f} "
                f"{d.get('hallucination_rate', 0)*100:>9.1f}% "
                f"{d.get('sources_rate', 0)*100:>7.0f}% "
                f"${d.get('total_cost_usd', 0):>8.3f} "
                f"{d.get('avg_latency_ms', 0):>7.0f}ms"
            )

        v2 = comparison.get("rag_specialized_v2", {})
        opt = comparison.get("rag_optimized", {})
        if opt and v2:
            delta_q = v2.get("avg_quality_score", 0) - opt.get("avg_quality_score", 0)
            print(f"\nvs RAG Optimise v2 : qualite {'+' if delta_q >= 0 else ''}{delta_q*100:.1f}%")

    print("\n" + "=" * 75)


def main() -> int:
    print("=" * 75)
    print("OpenDataCopilot - Evaluation RAG Specialise Medical v2")
    print("=" * 75)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = RAGSpecializedV2Config()

    questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees_enrichi.json"
    if not questions_path.exists():
        print(f"Questions non trouvees: {questions_path}")
        return 1
    if not config.medical_index_path.exists():
        print(f"Index FAISS medical absent: {config.medical_index_path}")
        print("Lancez d'abord: python -m experiments.rag_specialized_v2.data_indexer_medical")
        return 1

    questions = load_questions(questions_path)
    print(f"\n{len(questions)} questions chargees")

    # Charger les rapports précédents pour comparaison
    prev_reports: dict = {}
    report_paths = {
        "rag_basic": PROJECT_ROOT / "experiments" / "rag_basic" / "results" / "rag_basic_1222k_enrichi_report.json",
        "rag_optimized": PROJECT_ROOT / "experiments" / "rag_optimized" / "results" / "rag_optimized_1222k_enrichi_report.json",
        "rag_specialized": PROJECT_ROOT / "experiments" / "rag_specialized" / "results" / "rag_specialized_1222k_enrichi_report.json",
    }
    for key, path in report_paths.items():
        if path.exists():
            with open(path) as f:
                prev_reports[key] = json.load(f)
            print(f"Rapport {key} charge")

    print("\nInitialisation du RAG Medical v2...")
    rag = SpecializedMedicalRAG(config)
    rag.initialize()

    n_docs = rag.retriever.index.ntotal if rag.retriever.index else 0
    report_path = config.results_dir / f"rag_specialized_v2_{n_docs // 1000}k_enrichi_report.json"

    print(f"\nEvaluation sur {len(questions)} questions...")
    print("Pipeline: CamemBERT-bio embed → FAISS_medical + BM25 → CrossEncoder(5)")
    results = evaluate_rag(questions, rag)

    print("\nGeneration du rapport...")
    report = generate_report(results, rag, prev_reports)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Rapport sauvegarde: {report_path}")

    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
