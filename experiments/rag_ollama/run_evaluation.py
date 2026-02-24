#!/usr/bin/env python3
"""
OpenDataCopilot - Évaluation RAG Ollama
=========================================

Évalue Mistral 7B et Llama3 8B sur les 50 questions enrichies,
puis compare avec GPT-3.5 (RAG Basic).

Même méthodologie que run_rag_basic.py pour équité de comparaison :
- Mêmes questions (questions_annotees_enrichi.json)
- Même retrieval FAISS (572K docs)
- Mêmes métriques (utilité, relevance, hallucinations, sources)

Usage:
    python -m experiments.rag_ollama.run_evaluation
    python -m experiments.rag_ollama.run_evaluation --model mistral:7b
    python -m experiments.rag_ollama.run_evaluation --model llama3:8b
    python -m experiments.rag_ollama.run_evaluation --all
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rag_ollama.rag_ollama import OllamaRAG
from experiments.rag_ollama.config import OllamaConfig, AVAILABLE_MODELS

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


# ─────────────────────────────────────────────────────────────
# Métriques (copiées de run_rag_basic.py pour équité)
# ─────────────────────────────────────────────────────────────

def detect_hallucination(response: str, sources: list, num_docs: int) -> dict:
    """Identique à run_rag_basic.py."""
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
    """Identique à run_rag_basic.py."""
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


# ─────────────────────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────────────────────

def load_questions(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def load_gpt35_results() -> dict | None:
    """Charge les résultats GPT-3.5 pour comparaison."""
    ref_path = PROJECT_ROOT / "experiments" / "rag_basic" / "results" / "rag_basic_572k_enrichi_report.json"
    if ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback sur le rapport 20 questions
    fallback = PROJECT_ROOT / "experiments" / "rag_basic" / "results" / "rag_basic_572k_report.json"
    if fallback.exists():
        with open(fallback, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────
# Boucle d'évaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(
    questions: list[dict],
    rag: OllamaRAG,
    model_name: str,
) -> list[dict]:
    """Évalue un modèle Ollama sur toutes les questions."""
    results = []
    process = psutil.Process()

    print(f"\n{'='*70}")
    print(f"🤖 Évaluation : {model_name}")
    print(f"{'='*70}")

    for i, q in enumerate(questions, 1):
        question_id = q.get("id", f"q{i}")
        question_text = q.get("question", "")
        category = q.get("category", "unknown")
        ground_truth = q.get("ground_truth")

        ram_gb = process.memory_info().rss / (1024 ** 3)

        print(f"\n[{i:02d}/{len(questions)}] {question_id} ({category})")
        print(f"  Q: {question_text[:75]}...")
        print(f"  💾 RAM: {ram_gb:.1f} GB", end="", flush=True)

        start_time = time.time()
        try:
            response = rag.query(question_text, top_k=5)
            total_time = (time.time() - start_time) * 1000
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"Erreur sur {question_id}: {e}")
            results.append({
                "id": question_id,
                "question": question_text,
                "category": category,
                "ground_truth": ground_truth,
                "model": model_name,
                "response": {"answer": f"ERREUR: {e}", "confidence": 0, "num_sources": 0, "sources": []},
                "metrics": {"total_time_ms": total_time, "retrieval_time_ms": 0,
                            "generation_time_ms": 0, "tokens_used": 0, "output_tokens": 0,
                            "cost_usd": 0, "num_docs_retrieved": 0, "avg_relevance_score": 0,
                            "tokens_per_second": 0},
                "hallucination_analysis": {"is_hallucination": False, "confidence": 0, "indicators": [], "has_sources": False},
                "quality_analysis": {"has_answer": False, "cites_sources": False, "mentions_dates": False, "admits_uncertainty": False, "score": 0},
            })
            continue

        answer = response.answer
        sources = response.sources
        documents = response.documents
        num_docs = len(documents)

        hallucination = detect_hallucination(answer, sources, num_docs)
        quality = evaluate_response_quality(answer, ground_truth, sources)

        relevance = response.metadata.get("avg_relevance_score", 0)
        tokens = response.metadata.get("tokens_used", 0)
        output_tokens = response.metadata.get("output_tokens", 0)
        cost = response.metadata.get("cost_usd", 0)
        tps = response.metadata.get("ollama_tokens_per_second", 0)

        print(f"\n  ⏱️  {total_time:.0f}ms | 🎯 Relevance: {relevance:.2f} | "
              f"📊 Qualité: {quality['score']:.2f} | "
              f"⚡ {tps:.0f} tok/s | "
              f"{'⚠️  HALLUC' if hallucination['is_hallucination'] else '✅ OK'}")
        print(f"  R: {answer[:120]}...")

        result = {
            "id": question_id,
            "question": question_text,
            "category": category,
            "ground_truth": ground_truth,
            "model": model_name,
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
                "tokens_used": tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "num_docs_retrieved": num_docs,
                "avg_relevance_score": relevance,
                "tokens_per_second": tps,
            },
            "hallucination_analysis": hallucination,
            "quality_analysis": quality,
        }
        results.append(result)

        # Pause courte pour ne pas saturer Ollama
        time.sleep(0.5)

    return results


# ─────────────────────────────────────────────────────────────
# Rapport
# ─────────────────────────────────────────────────────────────

def generate_model_report(results: list[dict], rag: OllamaRAG, model_name: str) -> dict:
    """Génère le rapport pour un modèle."""
    stats = rag.get_stats()
    total = len(results)

    valid = [r for r in results if "ERREUR" not in r["response"]["answer"]]
    errors = total - len(valid)

    total_time = sum(r["metrics"]["total_time_ms"] for r in valid)
    total_retrieval = sum(r["metrics"]["retrieval_time_ms"] for r in valid)
    total_cost = sum(r["metrics"]["cost_usd"] for r in valid)
    total_tokens = sum(r["metrics"]["tokens_used"] for r in valid)
    total_output = sum(r["metrics"]["output_tokens"] for r in valid)
    total_tps = sum(r["metrics"]["tokens_per_second"] for r in valid)

    hallucinations = [r for r in valid if r["hallucination_analysis"]["is_hallucination"]]
    with_sources = [r for r in valid if r["response"]["num_sources"] > 0]

    n = len(valid) or 1
    avg_quality = sum(r["quality_analysis"]["score"] for r in valid) / n
    avg_relevance = sum(r["metrics"]["avg_relevance_score"] for r in valid) / n
    avg_tps = total_tps / n

    # Par catégorie
    categories: dict[str, dict] = {}
    for r in valid:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "hallucinations": 0, "with_sources": 0,
                               "avg_quality": 0.0, "avg_relevance": 0.0}
        categories[cat]["count"] += 1
        categories[cat]["avg_quality"] += r["quality_analysis"]["score"]
        categories[cat]["avg_relevance"] += r["metrics"]["avg_relevance_score"]
        if r["hallucination_analysis"]["is_hallucination"]:
            categories[cat]["hallucinations"] += 1
        if r["response"]["num_sources"] > 0:
            categories[cat]["with_sources"] += 1

    for cat in categories:
        cnt = categories[cat]["count"]
        categories[cat]["avg_quality"] /= cnt
        categories[cat]["avg_relevance"] /= cnt

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "embedding_model": stats.get("embedding_model"),
            "rag_type": "ollama_faiss",
            "index_size": stats.get("index_size", 0),
            "ollama_url": rag.config.ollama_base_url,
        },
        "summary": {
            "total_questions": total,
            "valid_responses": len(valid),
            "errors": errors,
            "avg_total_time_ms": total_time / n,
            "avg_retrieval_time_ms": total_retrieval / n,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "total_output_tokens": total_output,
            "avg_tokens_per_second": avg_tps,
            "hallucination_count": len(hallucinations),
            "hallucination_rate": len(hallucinations) / n,
            "responses_with_sources": len(with_sources),
            "sources_rate": len(with_sources) / n,
            "avg_docs_retrieved": sum(r["metrics"]["num_docs_retrieved"] for r in valid) / n,
            "avg_relevance_score": avg_relevance,
            "avg_quality_score": avg_quality,
        },
        "by_category": {
            cat: {
                "count": d["count"],
                "hallucinations": d["hallucinations"],
                "with_sources": d["with_sources"],
                "avg_quality": d["avg_quality"],
                "avg_relevance": d["avg_relevance"],
            }
            for cat, d in sorted(categories.items())
        },
        "results": results,
    }


# ─────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────

def print_model_summary(report: dict) -> None:
    """Affiche un résumé d'un modèle."""
    s = report["summary"]
    m = report["metadata"]["model"]
    cats = report.get("by_category", {})

    print(f"\n{'='*70}")
    print(f"📊 RÉSUMÉ — {m}")
    print(f"{'='*70}")
    print(f"  Questions traitées : {s['total_questions']} ({s['errors']} erreurs)")
    print(f"  Temps moyen       : {s['avg_total_time_ms']:.0f} ms")
    print(f"  Retrieval moyen   : {s['avg_retrieval_time_ms']:.0f} ms")
    print(f"  Génération moy    : {s['avg_total_time_ms'] - s['avg_retrieval_time_ms']:.0f} ms")
    print(f"  Tokens/sec        : {s['avg_tokens_per_second']:.0f}")
    print(f"  Coût total        : ${s['total_cost_usd']:.5f} (embeddings uniquement)")
    print(f"  Score relevance   : {s['avg_relevance_score']:.3f}")
    print(f"  Score qualité     : {s['avg_quality_score']:.2f}/1.0 ({s['avg_quality_score']*100:.0f}%)")
    print(f"  Sources citées    : {s['responses_with_sources']}/{s['total_questions']} ({s['sources_rate']*100:.0f}%)")
    print(f"  Hallucinations    : {s['hallucination_count']}/{s['total_questions']} ({s['hallucination_rate']*100:.1f}%)")

    # Par type de question enrichi
    enriched_cats = ["temporelle_precise", "geographique", "multi_criteres"]
    enriched_data = {c: cats[c] for c in enriched_cats if c in cats}
    if enriched_data:
        print(f"\n  📊 Questions enrichies :")
        for cat, d in enriched_data.items():
            print(f"    {cat:<25}: qualité {d['avg_quality']:.2f} | relevance {d['avg_relevance']:.3f}")


def print_comparison_table(all_reports: list[dict], gpt35_report: dict | None) -> None:
    """Affiche le tableau comparatif final."""
    print(f"\n{'='*80}")
    print("📊 TABLEAU COMPARATIF FINAL")
    print(f"{'='*80}")

    rows = []

    # GPT-3.5
    if gpt35_report:
        s = gpt35_report["summary"]
        rows.append({
            "label": "GPT-3.5 (RAG Basic)",
            "quality": s.get("avg_quality_score", 0) * 100,
            "relevance": s.get("avg_relevance_score", 0),
            "sources": s.get("sources_rate", 0) * 100,
            "hallucination": s.get("hallucination_rate", 0) * 100,
            "latency": s.get("avg_total_time_ms", s.get("avg_total_time_ms", 0)),
            "cost_q": s.get("total_cost_usd", 0) / max(s.get("total_questions", 1), 1),
            "tps": "N/A",
        })

    # Modèles Ollama
    for report in all_reports:
        s = report["summary"]
        rows.append({
            "label": report["metadata"]["model"],
            "quality": s["avg_quality_score"] * 100,
            "relevance": s["avg_relevance_score"],
            "sources": s["sources_rate"] * 100,
            "hallucination": s["hallucination_rate"] * 100,
            "latency": s["avg_total_time_ms"],
            "cost_q": s["total_cost_usd"] / max(s["total_questions"], 1),
            "tps": f"{s['avg_tokens_per_second']:.0f}",
        })

    # Affichage
    hdr = f"{'Métrique':<28}"
    for r in rows:
        hdr += f" {r['label']:>17}"
    print(hdr)
    print("-" * (28 + 18 * len(rows)))

    metrics = [
        ("Utilité (%)", "quality", "{:.1f}"),
        ("Relevance", "relevance", "{:.3f}"),
        ("Sources (%)", "sources", "{:.0f}"),
        ("Hallucination (%)", "hallucination", "{:.1f}"),
        ("Latence (ms)", "latency", "{:.0f}"),
        ("Coût/question ($)", "cost_q", "${:.5f}"),
        ("Tokens/sec", "tps", "{}"),
    ]

    for label, key, fmt in metrics:
        line = f"{label:<28}"
        for r in rows:
            val = r[key]
            if val == "N/A":
                line += f" {'N/A':>17}"
            else:
                line += f" {fmt.format(val):>17}"
        print(line)

    print(f"\n{'='*80}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    """Point d'entrée principal."""
    print("=" * 70)
    print("🚀 OpenDataCopilot — Évaluation RAG Ollama")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Sélection des modèles
    run_all = "--all" in sys.argv
    model_arg = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model_arg = arg.split("=", 1)[1]
        elif arg not in ("--all",) and not arg.startswith("--"):
            model_arg = arg

    if run_all:
        models_to_eval = list(AVAILABLE_MODELS.values())
    elif model_arg:
        models_to_eval = [model_arg]
    else:
        # Par défaut : les deux modèles
        models_to_eval = list(AVAILABLE_MODELS.values())

    print(f"\n🤖 Modèles à évaluer : {models_to_eval}")

    # Chemins
    questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees_enrichi.json"
    if not questions_path.exists():
        # Fallback sur le dataset original
        questions_path = PROJECT_ROOT / "evaluation" / "datasets" / "questions_annotees.json"
        print(f"⚠️  Dataset enrichi non trouvé, utilisation de: {questions_path.name}")

    # Charger les questions
    print(f"\n📂 Chargement: {questions_path.name}")
    questions = load_questions(questions_path)
    print(f"   ✅ {len(questions)} questions chargées")

    # Référence GPT-3.5
    gpt35_report = load_gpt35_results()
    if gpt35_report:
        n_q = gpt35_report["summary"]["total_questions"]
        quality = gpt35_report["summary"].get("avg_quality_score", 0)
        print(f"   ✅ Référence GPT-3.5 chargée ({n_q} questions, utilité: {quality*100:.0f}%)")

    # Info système
    vm = psutil.virtual_memory()
    print(f"\n💻 VM : {psutil.cpu_count()} CPU | "
          f"RAM libre: {vm.available/(1024**3):.0f} GB / {vm.total/(1024**3):.0f} GB")

    # Évaluation par modèle
    all_reports = []
    run_start = time.time()

    for model_name in models_to_eval:
        print(f"\n{'='*70}")
        print(f"⚙️  Initialisation de {model_name}...")

        config = OllamaConfig(model_name=model_name)
        rag = OllamaRAG(config)

        try:
            rag.initialize()
        except Exception as e:
            print(f"❌ Impossible d'initialiser {model_name}: {e}")
            continue

        # Évaluer
        results = evaluate_model(questions, rag, model_name)

        # Générer le rapport
        report = generate_model_report(results, rag, model_name)

        # Sauvegarder
        safe_name = model_name.replace(":", "_").replace("/", "_")
        report_path = config.results_dir / f"rag_ollama_{safe_name}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n   ✅ Rapport sauvegardé: {report_path.name}")

        # Résumé
        print_model_summary(report)
        all_reports.append(report)

    # Rapport comparatif global
    if all_reports:
        combined_path = (
            PROJECT_ROOT / "experiments" / "rag_ollama" / "results"
            / "ollama_comparison_report.json"
        )
        combined = {
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": [r["metadata"]["model"] for r in all_reports],
            "total_runtime_minutes": (time.time() - run_start) / 60,
            "reports": {r["metadata"]["model"]: r["summary"] for r in all_reports},
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"\n   ✅ Rapport comparatif: {combined_path.name}")

        # Tableau final
        print_comparison_table(all_reports, gpt35_report)

        # Générer les visualisations
        try:
            _generate_html_report(all_reports, gpt35_report)
        except Exception as e:
            print(f"\n⚠️  Visualisations non générées: {e}")

    total_time = (time.time() - run_start) / 60
    print(f"\n✅ Évaluation terminée en {total_time:.1f} minutes")
    return 0


def _generate_html_report(ollama_reports: list[dict], gpt35_report: dict | None) -> None:
    """Génère les visualisations Plotly dans un fichier HTML."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("⚠️  Plotly non installé — visualisations HTML ignorées")
        return

    # Données
    labels = []
    qualities, relevances, latencies, costs, tps_vals = [], [], [], [], []
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]

    if gpt35_report:
        s = gpt35_report["summary"]
        labels.append("GPT-3.5")
        qualities.append(s.get("avg_quality_score", 0) * 100)
        relevances.append(s.get("avg_relevance_score", 0))
        latencies.append(s.get("avg_total_time_ms", 0))
        n = max(s.get("total_questions", 1), 1)
        costs.append(s.get("total_cost_usd", 0) / n * 1000)
        tps_vals.append(None)

    for report in ollama_reports:
        s = report["summary"]
        labels.append(report["metadata"]["model"])
        qualities.append(s["avg_quality_score"] * 100)
        relevances.append(s["avg_relevance_score"])
        latencies.append(s["avg_total_time_ms"])
        n = max(s["total_questions"], 1)
        costs.append(s["total_cost_usd"] / n * 1000)
        tps_vals.append(s["avg_tokens_per_second"])

    # Graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Score d'Utilité (%)",
            "Latence Moyenne (ms)",
            "Score de Relevance",
            "Coût par Question (millièmes $)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for i, (label, color) in enumerate(zip(labels, colors)):
        # Utilité
        fig.add_trace(go.Bar(
            name=label, x=[label], y=[qualities[i]],
            marker_color=color,
            text=[f"{qualities[i]:.0f}%"], textposition="outside",
            showlegend=(i == 0),
        ), row=1, col=1)

        # Latence
        fig.add_trace(go.Bar(
            name=label, x=[label], y=[latencies[i]],
            marker_color=color,
            text=[f"{latencies[i]:.0f}ms"], textposition="outside",
            showlegend=False,
        ), row=1, col=2)

        # Relevance
        fig.add_trace(go.Bar(
            name=label, x=[label], y=[relevances[i]],
            marker_color=color,
            text=[f"{relevances[i]:.3f}"], textposition="outside",
            showlegend=False,
        ), row=2, col=1)

        # Coût
        fig.add_trace(go.Bar(
            name=label, x=[label], y=[costs[i]],
            marker_color=color,
            text=[f"${costs[i]:.4f}m"], textposition="outside",
            showlegend=False,
        ), row=2, col=2)

    fig.update_yaxes(range=[0, 110], row=1, col=1)
    fig.update_layout(
        title="OpenDataCopilot — GPT-3.5 vs Ollama (Mistral 7B / Llama3 8B)",
        height=700,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", y=1.05),
    )

    # Graphique par type de question enrichi
    enriched_cats = ["temporelle_precise", "geographique", "multi_criteres"]
    cat_labels = ["Temporelle\nPrécise", "Géographique", "Multi-critères"]

    fig2 = go.Figure()
    for i, (label, color) in enumerate(zip(labels, colors)):
        y_vals = []
        for cat in enriched_cats:
            if i == 0 and gpt35_report:
                by_cat = gpt35_report.get("by_category", {})
                val = by_cat.get(cat, {}).get("avg_quality", 0) * 100
            elif i > 0 or not gpt35_report:
                idx = i if not gpt35_report else i - 1
                if idx < len(ollama_reports):
                    by_cat = ollama_reports[idx].get("by_category", {})
                    val = by_cat.get(cat, {}).get("avg_quality", 0) * 100
                else:
                    val = 0
            else:
                val = 0
            y_vals.append(val)

        fig2.add_trace(go.Bar(
            name=label, x=cat_labels, y=y_vals,
            marker_color=color,
            text=[f"{v:.0f}%" for v in y_vals],
            textposition="outside",
        ))

    fig2.update_layout(
        title="Performance par Type de Question Enrichie",
        barmode="group",
        height=450,
        template="plotly_white",
        yaxis=dict(range=[0, 110], title="Score d'utilité (%)"),
    )

    # Scatter : Qualité vs Latence (ROI)
    fig3 = go.Figure()
    for i, (label, color) in enumerate(zip(labels, colors)):
        tps_text = f"<br>Tokens/sec: {tps_vals[i]:.0f}" if tps_vals[i] else ""
        fig3.add_trace(go.Scatter(
            x=[latencies[i]], y=[qualities[i]],
            mode="markers+text",
            name=label,
            text=[label],
            textposition="top center",
            marker=dict(size=20, color=color, line=dict(width=2, color="white")),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Latence: {latencies[i]:.0f}ms<br>"
                f"Utilité: {qualities[i]:.0f}%<br>"
                f"Coût/q: ${costs[i]/1000:.6f}"
                f"{tps_text}"
                "<extra></extra>"
            ),
        ))

    fig3.update_layout(
        title="ROI : Qualité vs Latence (taille = coût)",
        xaxis_title="Latence moyenne (ms)",
        yaxis_title="Score d'utilité (%)",
        height=450,
        template="plotly_white",
    )

    # Générer le HTML
    output_path = PROJECT_ROOT / "evaluation" / "results" / "comparison_with_ollama.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write('<meta charset="utf-8">\n')
        f.write("<title>OpenDataCopilot - GPT-3.5 vs Ollama</title>\n")
        f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n')
        f.write("<style>\n")
        f.write("body{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;padding:20px;background:#f5f5f5}\n")
        f.write("h1{color:#333;text-align:center}h2{color:#4472C4;margin-top:30px}\n")
        f.write(".chart{background:white;padding:20px;margin:20px 0;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1)}\n")
        f.write(".summary{background:#e8f4fd;padding:20px;border-radius:8px;margin:20px 0;font-size:14px}\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write("<h1>OpenDataCopilot — GPT-3.5 vs Ollama Local</h1>\n")
        f.write(f"<p style='text-align:center;color:#666'>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>\n")

        # Résumé
        f.write("<div class='summary'><h2>Résumé</h2><ul>\n")
        f.write("<li>Retrieval identique pour tous (FAISS 572K docs + OpenAI embeddings)</li>\n")
        f.write("<li>Seul le LLM de génération change</li>\n")
        f.write("<li>50 questions enrichies avec ground truths vérifiés</li>\n")
        f.write("</ul></div>\n")

        for idx, (title, fig_obj) in enumerate([
            ("Métriques Globales", fig),
            ("Performance par Type de Question", fig2),
            ("ROI : Qualité vs Latence", fig3),
        ]):
            did = f"chart_{idx}"
            f.write(f"<div class='chart'><h2>{title}</h2>\n")
            f.write(f"<div id='{did}'></div>\n<script>\n")
            fig_json = fig_obj.to_json()
            f.write(f"var d=JSON.parse('{fig_json.replace(chr(39), chr(92)+chr(39))}');\n")
            f.write(f"Plotly.newPlot('{did}',d.data,d.layout);\n")
            f.write("</script></div>\n")

        f.write("</body>\n</html>")

    print(f"   ✅ Visualisations: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
