#!/bin/bash
# Compilation du poster scientifique OpenDataCopilot
# Exécuter depuis la racine du projet : bash poster/compile.sh
set -e

POSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$POSTER_DIR")"
OUTPUT="$ROOT_DIR/poster_opendatacopilot.pdf"

echo "═══════════════════════════════════════════"
echo " Compilation poster LaTeX A0"
echo "═══════════════════════════════════════════"

# Générer les graphiques Python si pas présents
if [ ! -f "$POSTER_DIR/assets/graph_diversite.png" ]; then
    echo "→ Génération graphiques Python..."
    cd "$ROOT_DIR"
    python poster/generate_graphs.py
fi

# Vérifier les assets
echo "→ Vérification assets..."
for img in graph_diversite.png graph_complexite.png tableau_embeddings.png \
           03_temps_reel_air.png 05_pollution_actuelle.png; do
    if [ ! -f "$POSTER_DIR/assets/$img" ]; then
        echo "  ❌ Manquant : assets/$img"
        exit 1
    fi
    echo "  ✅ assets/$img"
done

# Compilation LaTeX (2 passes)
cd "$POSTER_DIR"
echo "→ pdflatex pass 1..."
pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /tmp/pdflatex.log 2>&1 || {
    echo "❌ Erreur LaTeX (pass 1) :"
    tail -30 /tmp/pdflatex.log
    exit 1
}

echo "→ pdflatex pass 2..."
pdflatex -interaction=nonstopmode poster.tex >> /tmp/pdflatex.log 2>&1 || {
    echo "❌ Erreur LaTeX (pass 2) :"
    tail -20 /tmp/pdflatex.log
    exit 1
}

# Déplacer le PDF à la racine
mv poster.pdf "$OUTPUT"
echo ""
echo "═══════════════════════════════════════════"
echo "✅ Poster généré : poster_opendatacopilot.pdf"
echo "   Format : A0 portrait (84 × 119 cm)"
echo "   $(du -sh "$OUTPUT" | cut -f1)"
echo "═══════════════════════════════════════════"
