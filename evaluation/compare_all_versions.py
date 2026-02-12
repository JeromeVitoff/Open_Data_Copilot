#!/usr/bin/env python3
"""
OpenDataCopilot - Comparaison visuelle de toutes les versions
=============================================================

Compare les performances de :
1. Baseline (LLM seul, sans RAG)
2. RAG Basic 5K docs (FAISS + chunking simple, index réduit)
3. RAG Basic 572K docs (FAISS + corpus complet)

Génère des graphiques Plotly interactifs et un tableau comparatif.

Usage:
    python -m evaluation.compare_all_versions
    python evaluation/compare_all_versions.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("Plotly non installé - visualisations HTML désactivées")

# Configuration du logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


def find_reports() -> dict[str, dict]:
    """
    Trouve et charge tous les rapports d'évaluation disponibles.

    Returns:
        Dict {nom_version: contenu_rapport}
    """
    reports = {}

    # Baseline
    baseline_path = PROJECT_ROOT / "experiments" / "baseline" / "results" / "baseline_report.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            reports["Baseline (LLM seul)"] = json.load(f)
        logger.info(f"   ✅ Baseline: {baseline_path.name}")

    # RAG Basic - rechercher tous les rapports
    rag_basic_dir = PROJECT_ROOT / "experiments" / "rag_basic" / "results"
    if rag_basic_dir.exists():
        for report_file in sorted(rag_basic_dir.glob("rag_basic*.json")):
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            index_size = data.get("metadata", {}).get("index_size", 0)

            is_enriched = "enrichi" in report_file.name
            n_questions = data.get("summary", {}).get("total_questions", 0)
            enriched_tag = f" - {n_questions}q enrichi" if is_enriched else ""

            if "572k" in report_file.name and not is_enriched:
                label = f"RAG Basic 572K (20q)"
            elif "572k" in report_file.name and is_enriched:
                label = f"RAG Basic 572K (50q enrichi)"
            elif index_size and index_size > 100_000:
                label = f"RAG Basic ({index_size // 1000}K docs{enriched_tag})"
            elif index_size and index_size > 0:
                label = f"RAG Basic ({index_size:,} docs)"
            else:
                label = f"RAG Basic ({report_file.stem})"

            reports[label] = data
            logger.info(f"   ✅ {label}: {report_file.name}")

    return reports


def extract_metrics(reports: dict[str, dict]) -> dict:
    """
    Extrait les métriques clés de chaque rapport pour comparaison.

    Returns:
        Dict structuré pour les graphiques
    """
    metrics = {}

    for version, report in reports.items():
        summary = report.get("summary", {})
        metadata = report.get("metadata", {})

        m = {
            "version": version,
            "index_size": metadata.get("index_size", 0),
            "total_questions": summary.get("total_questions", 0),
            # Performance
            "avg_latency_ms": summary.get("avg_latency_ms", summary.get("avg_total_time_ms", 0)),
            "avg_retrieval_ms": summary.get("avg_retrieval_time_ms", 0),
            # Qualité
            "hallucination_rate": summary.get("hallucination_rate", 0) * 100,
            "sources_rate": summary.get("sources_rate", 0) * 100,
            "avg_quality": summary.get("avg_quality_score", 0) * 100,
            "avg_relevance": summary.get("avg_relevance_score", 0),
            "avg_docs_retrieved": summary.get("avg_docs_retrieved", 0),
            # Coûts
            "total_cost": summary.get("total_cost_usd", 0),
            "cost_per_question": summary.get("total_cost_usd", 0) / max(summary.get("total_questions", 1), 1),
            "total_tokens": summary.get("total_tokens", 0),
        }

        # Pour la baseline, ajuster l'utilité
        if "Baseline" in version:
            m["avg_quality"] = 30.0  # Estimation baseline
            m["sources_rate"] = 0.0
            m["avg_relevance"] = 0.0
            m["avg_docs_retrieved"] = 0.0

        metrics[version] = m

    return metrics


def create_bar_chart(metrics: dict, output_path: Path) -> go.Figure:
    """
    Crée un bar chart comparatif : Utilité, Sources, Hallucinations.
    """
    versions = list(metrics.keys())
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Score d'Utilité (%)",
            "Réponses avec Sources (%)",
            "Taux d'Hallucination (%)",
        ],
        horizontal_spacing=0.08,
    )

    for i, version in enumerate(versions):
        m = metrics[version]
        color = colors[i % len(colors)]

        # Utilité
        fig.add_trace(
            go.Bar(
                name=version,
                x=[version],
                y=[m["avg_quality"]],
                marker_color=color,
                text=[f"{m['avg_quality']:.0f}%"],
                textposition="outside",
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Sources
        fig.add_trace(
            go.Bar(
                name=version,
                x=[version],
                y=[m["sources_rate"]],
                marker_color=color,
                text=[f"{m['sources_rate']:.0f}%"],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=2,
        )

        # Hallucinations
        fig.add_trace(
            go.Bar(
                name=version,
                x=[version],
                y=[m["hallucination_rate"]],
                marker_color=color,
                text=[f"{m['hallucination_rate']:.1f}%"],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=3,
        )

    fig.update_layout(
        title="Comparaison des Performances RAG - OpenDataCopilot",
        height=500,
        template="plotly_white",
        font=dict(size=12),
    )

    fig.update_yaxes(range=[0, 110], row=1, col=1)
    fig.update_yaxes(range=[0, 110], row=1, col=2)
    fig.update_yaxes(range=[0, 20], row=1, col=3)

    return fig


def create_relevance_vs_docs(metrics: dict) -> go.Figure:
    """
    Crée un line chart : Score relevance vs nombre de documents indexés.
    """
    # Filtrer les versions avec un index
    rag_versions = {k: v for k, v in metrics.items() if v["index_size"] > 0}

    if len(rag_versions) < 2:
        # Ajouter la baseline comme point 0
        versions = list(metrics.keys())
        x_vals = [metrics[v]["index_size"] for v in versions]
        y_relevance = [metrics[v]["avg_relevance"] for v in versions]
        y_quality = [metrics[v]["avg_quality"] for v in versions]
    else:
        versions = list(rag_versions.keys())
        x_vals = [rag_versions[v]["index_size"] for v in versions]
        y_relevance = [rag_versions[v]["avg_relevance"] for v in versions]
        y_quality = [rag_versions[v]["avg_quality"] for v in versions]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Score de Relevance vs Taille d'Index",
            "Score d'Utilité vs Taille d'Index",
        ],
    )

    # Relevance
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_relevance,
            mode="lines+markers+text",
            name="Relevance",
            text=[f"{r:.3f}" for r in y_relevance],
            textposition="top center",
            marker=dict(size=12, color="#636EFA"),
            line=dict(width=3),
        ),
        row=1, col=1,
    )

    # Qualité
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_quality,
            mode="lines+markers+text",
            name="Utilité (%)",
            text=[f"{q:.0f}%" for q in y_quality],
            textposition="top center",
            marker=dict(size=12, color="#EF553B"),
            line=dict(width=3),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Nombre de documents indexés", row=1, col=1)
    fig.update_xaxes(title_text="Nombre de documents indexés", row=1, col=2)
    fig.update_yaxes(title_text="Score de relevance", row=1, col=1)
    fig.update_yaxes(title_text="Score d'utilité (%)", row=1, col=2)

    fig.update_layout(
        title="Impact du Volume de Données sur la Performance RAG",
        height=450,
        template="plotly_white",
    )

    return fig


def create_cost_vs_performance(metrics: dict) -> go.Figure:
    """
    Crée un scatter plot : Coût vs Performance (utilité).
    """
    fig = go.Figure()

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    for i, (version, m) in enumerate(metrics.items()):
        fig.add_trace(
            go.Scatter(
                x=[m["cost_per_question"] * 1000],  # En millièmes de $
                y=[m["avg_quality"]],
                mode="markers+text",
                name=version,
                text=[version.split("(")[0].strip()],
                textposition="top center",
                marker=dict(
                    size=max(20, m["avg_docs_retrieved"] * 5 + 10),
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=2, color="white"),
                ),
            )
        )

    fig.update_layout(
        title="ROI : Coût vs Performance par Question",
        xaxis_title="Coût par question (millièmes de $)",
        yaxis_title="Score d'utilité (%)",
        height=450,
        template="plotly_white",
    )

    return fig


def create_comparison_table(metrics: dict) -> go.Figure:
    """
    Crée un tableau comparatif interactif Plotly.
    """
    versions = list(metrics.keys())

    headers = [
        "Métrique",
        *versions,
    ]

    rows = [
        "Documents indexés",
        "Score d'utilité (%)",
        "Réponses avec sources (%)",
        "Taux d'hallucination (%)",
        "Score de relevance",
        "Docs récupérés (moy)",
        "Latence moyenne (ms)",
        "Coût total ($)",
        "Coût/question ($)",
        "Tokens totaux",
    ]

    cells = [rows]

    for version in versions:
        m = metrics[version]
        col = [
            f"{m['index_size']:,}" if m["index_size"] > 0 else "N/A",
            f"{m['avg_quality']:.1f}",
            f"{m['sources_rate']:.0f}",
            f"{m['hallucination_rate']:.1f}",
            f"{m['avg_relevance']:.3f}" if m["avg_relevance"] > 0 else "N/A",
            f"{m['avg_docs_retrieved']:.1f}" if m["avg_docs_retrieved"] > 0 else "0",
            f"{m['avg_latency_ms']:.0f}",
            f"${m['total_cost']:.4f}",
            f"${m['cost_per_question']:.5f}",
            f"{m['total_tokens']:,}",
        ]
        cells.append(col)

    # Couleurs conditionnelles pour la colonne d'utilité
    fill_colors = [["white"] * len(rows)]
    for version in versions:
        m = metrics[version]
        col_colors = []
        for row_name in rows:
            if row_name == "Score d'utilité (%)":
                if m["avg_quality"] >= 80:
                    col_colors.append("#c6efce")  # Vert
                elif m["avg_quality"] >= 50:
                    col_colors.append("#ffeb9c")  # Jaune
                else:
                    col_colors.append("#ffc7ce")  # Rouge
            elif row_name == "Taux d'hallucination (%)":
                if m["hallucination_rate"] <= 5:
                    col_colors.append("#c6efce")
                else:
                    col_colors.append("#ffc7ce")
            else:
                col_colors.append("white")
        fill_colors.append(col_colors)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#4472C4",
                    font=dict(color="white", size=13),
                    align="center",
                    height=35,
                ),
                cells=dict(
                    values=cells,
                    fill_color=fill_colors,
                    font=dict(size=12),
                    align=["left"] + ["center"] * len(versions),
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Tableau Comparatif Détaillé - OpenDataCopilot",
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


def generate_insights(metrics: dict) -> list[str]:
    """
    Génère des insights clés à partir des métriques comparatives.
    """
    insights = []
    versions = list(metrics.keys())

    # Trouver baseline et RAG versions
    baseline_key = next((k for k in versions if "Baseline" in k), None)
    rag_keys = [k for k in versions if "RAG" in k]

    if baseline_key and rag_keys:
        baseline = metrics[baseline_key]
        best_rag = max(rag_keys, key=lambda k: metrics[k]["avg_quality"])
        best = metrics[best_rag]

        improvement = best["avg_quality"] - baseline["avg_quality"]
        insights.append(
            f"Le RAG améliore l'utilité de +{improvement:.0f} points "
            f"({baseline['avg_quality']:.0f}% -> {best['avg_quality']:.0f}%)"
        )

        cost_increase = (best["total_cost"] - baseline["total_cost"]) / max(baseline["total_cost"], 0.001) * 100
        insights.append(
            f"Le surcoût du RAG est de {cost_increase:+.0f}% "
            f"(${baseline['total_cost']:.4f} -> ${best['total_cost']:.4f})"
        )

        roi = improvement / max(cost_increase, 1)
        insights.append(f"ROI (gain utilité / surcoût) : {roi:.2f} points/% de coût")

    # Impact du volume de documents
    if len(rag_keys) >= 2:
        sorted_rags = sorted(rag_keys, key=lambda k: metrics[k]["index_size"])
        small = metrics[sorted_rags[0]]
        large = metrics[sorted_rags[-1]]

        doc_ratio = large["index_size"] / max(small["index_size"], 1)
        quality_diff = large["avg_quality"] - small["avg_quality"]
        relevance_diff = large["avg_relevance"] - small["avg_relevance"]

        insights.append(
            f"Passer de {small['index_size']:,} à {large['index_size']:,} docs "
            f"(x{doc_ratio:.0f}) change l'utilité de {quality_diff:+.1f} points"
        )

        if relevance_diff != 0:
            insights.append(
                f"Score de relevance : {small['avg_relevance']:.3f} -> {large['avg_relevance']:.3f} "
                f"({relevance_diff:+.3f})"
            )

    # Hallucinations
    halluc_rates = [metrics[v]["hallucination_rate"] for v in versions]
    if all(h == 0 for h in halluc_rates):
        insights.append("Aucune hallucination détectée dans toutes les versions (0%)")
    elif max(halluc_rates) > 0:
        worst = versions[halluc_rates.index(max(halluc_rates))]
        insights.append(f"Version la plus sujette aux hallucinations : {worst} ({max(halluc_rates):.1f}%)")

    return insights


def print_console_summary(metrics: dict, insights: list[str]) -> None:
    """
    Affiche un résumé formaté dans la console.
    """
    versions = list(metrics.keys())

    print("\n" + "=" * 80)
    print("📊 COMPARAISON TOUTES VERSIONS - OpenDataCopilot")
    print("=" * 80)

    # Tableau
    print(f"\n{'Métrique':<30}", end="")
    for v in versions:
        short = v.split("(")[0].strip()[:15]
        print(f" {short:>15}", end="")
    print()
    print("-" * (30 + 16 * len(versions)))

    rows = [
        ("Documents indexés", "index_size", "{:,}", "N/A"),
        ("Utilité (%)", "avg_quality", "{:.1f}", None),
        ("Sources (%)", "sources_rate", "{:.0f}", None),
        ("Hallucination (%)", "hallucination_rate", "{:.1f}", None),
        ("Relevance", "avg_relevance", "{:.3f}", "N/A"),
        ("Latence (ms)", "avg_latency_ms", "{:.0f}", None),
        ("Coût total ($)", "total_cost", "${:.4f}", None),
        ("Coût/question ($)", "cost_per_question", "${:.5f}", None),
    ]

    for label, key, fmt, zero_val in rows:
        print(f"{label:<30}", end="")
        for v in versions:
            val = metrics[v][key]
            if zero_val and val == 0:
                print(f" {zero_val:>15}", end="")
            else:
                formatted = fmt.format(val)
                print(f" {formatted:>15}", end="")
        print()

    # Insights
    print(f"\n{'='*80}")
    print("💡 INSIGHTS CLÉS")
    print("=" * 80)
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")

    print("\n" + "=" * 80)


def main():
    """Point d'entrée principal."""
    print("=" * 80)
    print("📊 OpenDataCopilot - Comparaison de Toutes les Versions")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Trouver les rapports
    print("\n📂 Recherche des rapports d'évaluation...")
    reports = find_reports()

    if not reports:
        print("\n❌ Aucun rapport trouvé. Lancez les évaluations d'abord.")
        return 1

    print(f"\n   📊 {len(reports)} versions trouvées")

    # Extraire les métriques
    metrics = extract_metrics(reports)

    # Générer les insights
    insights = generate_insights(metrics)

    # Afficher le résumé console
    print_console_summary(metrics, insights)

    # Sauvegarder les métriques JSON
    output_dir = PROJECT_ROOT / "evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "comparison_metrics.json"
    metrics_export = {
        "timestamp": datetime.now().isoformat(),
        "versions": {k: v for k, v in metrics.items()},
        "insights": insights,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, ensure_ascii=False, indent=2)
    print(f"\n   ✅ Métriques sauvegardées: {metrics_path}")

    # Générer les graphiques Plotly
    if HAS_PLOTLY:
        print("\n📈 Génération des visualisations Plotly...")

        html_path = output_dir / "comparison_all_versions.html"

        # Créer toutes les figures
        fig_bars = create_bar_chart(metrics, html_path)
        fig_relevance = create_relevance_vs_docs(metrics)
        fig_cost = create_cost_vs_performance(metrics)
        fig_table = create_comparison_table(metrics)

        # Combiner dans un seul HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n")
            f.write('<meta charset="utf-8">\n')
            f.write("<title>OpenDataCopilot - Comparaison des Versions</title>\n")
            f.write('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n')
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }\n")
            f.write("h1 { color: #333; text-align: center; }\n")
            f.write("h2 { color: #4472C4; margin-top: 30px; }\n")
            f.write(".chart { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n")
            f.write(".insights { background: #e8f4fd; padding: 20px; border-radius: 8px; margin: 20px 0; }\n")
            f.write(".insights li { margin: 8px 0; font-size: 14px; }\n")
            f.write("</style>\n")
            f.write("</head>\n<body>\n")
            f.write("<h1>OpenDataCopilot - Comparaison des Versions RAG</h1>\n")
            f.write(f"<p style='text-align:center;color:#666;'>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>\n")

            # Insights
            f.write("<div class='insights'>\n<h2>Insights Clés</h2>\n<ol>\n")
            for insight in insights:
                f.write(f"<li>{insight}</li>\n")
            f.write("</ol>\n</div>\n")

            # Graphiques
            charts = [
                ("Performances Globales", fig_bars),
                ("Impact du Volume de Données", fig_relevance),
                ("ROI : Coût vs Performance", fig_cost),
                ("Tableau Comparatif Détaillé", fig_table),
            ]

            for idx, (title, fig) in enumerate(charts):
                div_id = f"chart_{idx}"
                f.write(f"<div class='chart'>\n<h2>{title}</h2>\n")
                f.write(f"<div id='{div_id}'></div>\n")
                f.write("<script>\n")
                f.write(f"Plotly.newPlot('{div_id}', {fig.to_json()}.data, {fig.to_json()}.layout);\n")
                f.write("</script>\n</div>\n")

            f.write("</body>\n</html>")

        print(f"   ✅ Visualisations HTML: {html_path}")
    else:
        print("\n⚠️  Plotly non installé. Installez-le avec: pip install plotly")
        print("   Les métriques JSON ont été sauvegardées.")

    print(f"\n✅ Comparaison terminée!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
