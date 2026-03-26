# Poster Scientifique OpenDataCopilot

Poster A0 portrait (84 × 119 cm) pour la soutenance du Master 2 Data Science MIASHS.

## Compilation rapide

```bash
# Depuis la racine du projet
bash poster/compile.sh
# → génère poster_opendatacopilot.pdf
```

## Compilation manuelle

```bash
# Étape 1 : générer les graphiques Python
python poster/generate_graphs.py

# Étape 2 : compiler LaTeX
cd poster
pdflatex poster.tex
pdflatex poster.tex   # 2e passe pour stabiliser la mise en page
```

## Structure

```
poster/
├── poster.tex            # Source LaTeX principal (tikzposter A0)
├── compile.sh            # Script de compilation tout-en-un
├── generate_graphs.py    # Génère les graphiques PNG (matplotlib)
├── README.md             # Ce fichier
└── assets/
    ├── graph_diversite.png       # Impact diversité corpus (généré)
    ├── graph_complexite.png      # Complexité vs performance (généré)
    ├── tableau_embeddings.png    # Comparaison embeddings (généré)
    ├── 03_temps_reel_air.png     # Capture Streamlit (copié)
    └── 05_pollution_actuelle.png # Capture Streamlit (copié)
```

## Dépendances LaTeX (TexLive)

Packages requis (inclus dans `texlive-full`) :
- `tikzposter` — classe de poster A0
- `tikz` + `pgfplots` — schémas et graphiques
- `booktabs` — tableaux professionnels
- `graphicx` — inclusion d'images
- `xcolor` — couleurs personnalisées
- `lmodern` — polices vectorielles
- `siunitx` — formatage nombres (optionnel)

Installation rapide (Debian/Ubuntu) :
```bash
sudo apt install texlive-full
```

## Dépendances Python

```bash
pip install matplotlib numpy pandas
```

## Contenu du poster (4 colonnes)

| Col. | % | Contenu |
|------|---|---------|
| 1 | 22% | Contexte, enjeux, état de l'art |
| 2 | 26% | Méthodologie (4 architectures), corpus (1,2M docs) |
| 3 | 26% | Résultats comparaisons, 6 découvertes scientifiques |
| 4 | 26% | Prototype temps réel, évaluation critique, perspectives |

## Données sources

Résultats extraits depuis :
- `experiments/baseline/results/baseline_report.json`
- `experiments/rag_basic/results/rag_basic_1222k_enrichi_report.json`
- `experiments/rag_optimized/results/rag_optimized_1222k_enrichi_report.json`
- `experiments/rag_specialized/results/rag_specialized_1222k_enrichi_report.json`
- `experiments/rag_ollama/results/rag_ollama_mistral_7b_report.json`
