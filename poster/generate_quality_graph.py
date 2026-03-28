#!/usr/bin/env python3
"""
Génère le graphique de qualité des réponses (avg_quality_score)
depuis les résultats réels des expériences.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Scores réels extraits des JSON d'expériences
labels = ['Baseline\n(sans RAG)', 'RAG\nBasic', 'RAG\nOptimisé', 'RAG\nSpécialisé', 'Mistral\n7B', 'Llama3\n8B']
scores = [0.300, 0.710, 0.754, 0.706, 0.757, 0.760]

colors = [
    '#c0392b',   # rouge foncé  — Baseline
    '#2980b9',   # bleu moyen   — RAG Basic
    '#1a5276',   # bleu foncé   — RAG Optimisé
    '#d35400',   # orange foncé — RAG Spécialisé
    '#27ae60',   # vert moyen   — Mistral
    '#1e8449',   # vert foncé   — Llama3
]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(labels))
bars = ax.bar(x, scores, color=colors, width=0.6, edgecolor='white', linewidth=1.2)

# Valeurs au-dessus des barres
for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f'{score:.3f}',
        ha='center', va='bottom',
        fontsize=9, fontweight='bold', color='#2c3e50'
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Score qualité moyen', fontsize=10)
ax.set_ylim(0, 0.95)
ax.set_title('Qualité des réponses par architecture', fontsize=11, fontweight='bold', pad=10)
ax.axhline(y=0.7, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(5.55, 0.705, 'seuil 0.7', fontsize=7.5, color='gray', va='bottom')
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = 'poster/assets/graph_quality_bars.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'✅  Graphique sauvegardé : {out}')
