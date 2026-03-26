#!/usr/bin/env python3
"""
Génération graphiques PNG pour poster scientifique OpenDataCopilot.
Exécuter depuis la racine du projet : python poster/generate_graphs.py
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Config matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 13
plt.rcParams['figure.dpi'] = 150

# Couleurs identiques au poster LaTeX
C = {
    'primary':   '#2C3E50',
    'secondary': '#3498DB',
    'accent':    '#27AE60',
    'alert':     '#E74C3C',
    'orange':    '#E67E22',
    'gray':      '#95A5A6',
}

os.makedirs('poster/assets', exist_ok=True)


# ── 1. Impact diversité corpus ────────────────────────────────────────────
def graph_diversite():
    # Données réelles issues des JSONs de résultats
    labels  = ['5K docs\n(Baseline)', '572K docs\n(Enrichi v1)', '1,222K docs\n(Enrichi v2)']
    volumes = [5_000, 572_000, 1_222_802]
    relevance = [0.524, 0.607, 0.623]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(3), relevance, '-o',
            color=C['secondary'], linewidth=2.5, markersize=10,
            markerfacecolor=C['accent'], markeredgecolor='white', markeredgewidth=1.5,
            zorder=3)

    # Zone de remplissage
    ax.fill_between(range(3), relevance, alpha=0.12, color=C['secondary'])

    # Annotations valeurs
    for i, (r, lbl) in enumerate(zip(relevance, labels)):
        offset = (+0.05, +0.008) if i < 2 else (-0.05, +0.008)
        ax.annotate(f'{r:.3f}',
                    xy=(i, r), xytext=(i + offset[0], r + offset[1] + 0.015),
                    fontsize=12, fontweight='bold', color=C['primary'],
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=C['secondary'], alpha=0.9))

    # Annotation delta
    ax.annotate('', xy=(2, 0.623), xytext=(0, 0.524),
                arrowprops=dict(arrowstyle='->', color=C['accent'],
                                lw=1.5, connectionstyle='arc3,rad=-0.2'))
    ax.text(1.5, 0.545, '+18.9%\nrelevance', ha='center', fontsize=10,
            color=C['accent'], fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score de pertinence (cosine)', fontsize=12, fontweight='bold')
    ax.set_title('Impact de la Diversité du Corpus', fontsize=14, fontweight='bold',
                 color=C['primary'], pad=10)
    ax.set_ylim(0.45, 0.70)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('poster/assets/graph_diversite.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✅ graph_diversite.png')


# ── 2. Complexité vs Performance ──────────────────────────────────────────
def graph_complexite():
    architectures = ['Baseline', 'RAG Basic', 'RAG Optimisé', 'RAG Spécialisé']
    complexite    = [1, 2, 3, 4]
    qualite       = [0.300, 0.743, 0.754, 0.706]
    latences      = [1.8, 2.9, 5.0, 9.4]
    colors_pts    = [C['alert'], C['secondary'], C['accent'], C['orange']]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Points (taille proportionnelle à la latence)
    for x, y, lbl, col, lat in zip(complexite, qualite, architectures, colors_pts, latences):
        size = lat * 40
        ax.scatter(x, y, s=size, c=col, alpha=0.85,
                   edgecolors='white', linewidth=1.5, zorder=4)
        dy = 0.025 if y < 0.73 else -0.035
        ax.annotate(lbl, xy=(x, y), xytext=(x, y + dy),
                    ha='center', fontsize=10, fontweight='bold', color=C['primary'],
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.85))

    # Ligne tendance (quadratique sur les 3 premiers points)
    x_fit = np.array([1, 2, 3])
    y_fit = np.array([0.300, 0.743, 0.754])
    z = np.polyfit(x_fit, y_fit, 2)
    p = np.poly1d(z)
    xs = np.linspace(1, 4.2, 100)
    ax.plot(xs, p(xs), '--', color=C['gray'], alpha=0.6, linewidth=1.5,
            label='Tendance quadratique')

    # Zone "rendements décroissants"
    ax.axvspan(3, 4, alpha=0.05, color=C['alert'])
    ax.text(3.5, 0.35, 'Rendements\ndécroissants', ha='center', fontsize=9,
            color=C['alert'], style='italic')

    ax.set_xlabel('Complexité architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score qualité (utilité)', fontsize=12, fontweight='bold')
    ax.set_title('Complexité vs Performance', fontsize=14, fontweight='bold',
                 color=C['primary'], pad=10)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['1\nBaseline', '2\nBasic', '3\nOptimisé', '4\nSpécialisé'],
                       fontsize=10)
    ax.set_ylim(0.20, 0.85)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Légende taille des points
    for lat, lbl in [(1.8, '1.8s'), (5.0, '5.0s'), (9.4, '9.4s')]:
        ax.scatter([], [], s=lat * 40, c=C['gray'], alpha=0.6, label=f'Latence {lbl}')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig('poster/assets/graph_complexite.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✅ graph_complexite.png')


# ── 3. Tableau comparaison embeddings ────────────────────────────────────
def tableau_embeddings():
    data = {
        'Modèle': ['OpenAI text-emb-3-small', 'Sentence-CamemBERT', 'Solon', 'CamemBERT-bio'],
        'Gap moyen': ['0.622', '0.527', '0.467', '0.137'],
        'vs OpenAI': ['référence', '−4.6 %', '−24.9 %', '−78.0 %'],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('tight')
    ax.axis('off')

    col_widths = [0.48, 0.26, 0.26]
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)

    # En-tête
    for j in range(len(df.columns)):
        cell = table[(0, j)]
        cell.set_facecolor(C['primary'])
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_edgecolor('white')

    # Lignes de données
    row_colors = ['#FFFFFF', '#F0F4F8', '#FFFFFF', '#FFEBEE']
    text_colors = ['black', 'black', 'black', C['alert']]
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_facecolor(row_colors[i - 1])
            cell.set_text_props(color=text_colors[i - 1], fontsize=12)
            cell.set_edgecolor('#E0E0E0')
            if i == 1:  # OpenAI = meilleur
                cell.set_text_props(weight='bold', color=C['accent'], fontsize=12)

    ax.set_title('Comparaison Modèles d\'Embeddings', fontsize=13,
                 fontweight='bold', color=C['primary'], pad=10)

    plt.tight_layout()
    plt.savefig('poster/assets/tableau_embeddings.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('✅ tableau_embeddings.png')


if __name__ == '__main__':
    print('Génération graphiques PNG pour poster...')
    graph_diversite()
    graph_complexite()
    tableau_embeddings()
    print('\n✅ Tous les graphiques générés dans poster/assets/')
