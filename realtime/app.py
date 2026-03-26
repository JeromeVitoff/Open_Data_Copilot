"""
Interface Streamlit pour OpenDataCopilot - Version sobre
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Charger .env automatiquement (priorité au fichier .env à la racine du projet)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import streamlit as st

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OpenDataCopilot",
    page_icon="📊",
    layout="wide",
)

# ── CSS sobre ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main { background-color: #ffffff; }
    h1 { color: #2c3e50; font-weight: 400; font-size: 2.5rem; margin-bottom: 1rem; }
    h2 { color: #34495e; font-weight: 400; font-size: 1.5rem; margin-top: 2rem; }
    .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.6;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        border-radius: 4px;
    }
    .stButton button:hover { background-color: #2980b9; }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #dee2e6;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache RAG ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    """Charge HybridRAG une seule fois."""
    from realtime.hybrid_rag import HybridRAG
    return HybridRAG()


# ── Titre ─────────────────────────────────────────────────────────────────────
st.title("OpenDataCopilot")
st.markdown(
    "Assistant de recherche sur les données de santé publique "
    "et pollution atmosphérique"
)

# ── À propos ──────────────────────────────────────────────────────────────────
with st.expander("À propos"):
    st.markdown("""
Ce système permet d'interroger :
- **Données historiques** : 1,2 million de documents (2019-2023)
- **Données temps réel** : APIs Santé Publique France, Airparif, OpenAQ

Sources : SPF, ODISSE, Airparif, OpenAQ
    """)

# ── Exemples ──────────────────────────────────────────────────────────────────
with st.expander("Exemples de questions"):
    st.markdown("""
**Questions historiques :**
- Hospitalisations COVID à Paris en mars 2021
- Taux de vaccination en France en 2022

**Questions temps réel :**
- Qualité de l'air à Paris aujourd'hui
- Pollution actuelle en Île-de-France

**Questions mixtes :**
- Évolution de la pollution NO2 à Paris ces dernières années
    """)

# ── Zone de saisie ────────────────────────────────────────────────────────────
st.markdown("---")
question = st.text_input(
    "Votre question :",
    placeholder="Ex: Qualité de l'air à Paris aujourd'hui",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("Rechercher", use_container_width=True)

# ── Traitement ────────────────────────────────────────────────────────────────
if search_button and question.strip():
    with st.spinner("Recherche en cours..."):
        try:
            rag = load_rag()
            result = rag.query(question)

            # Réponse
            st.markdown("### Réponse")
            st.markdown(result["answer"])

            # Métadonnées sources
            st.markdown("---")
            st.markdown("### Sources consultées")

            temp_type = result["temporal_analysis"]["type"]
            type_labels = {
                "historical":  "Données historiques",
                "realtime":    "Données historiques + temps réel",
                "mixed":       "Données mixtes",
                "unspecified": "Données historiques",
            }
            st.caption(f"Type de recherche : {type_labels.get(temp_type, 'Non spécifié')}")
            st.caption(
                f"{result['num_historical']} sources historiques, "
                f"{result['num_realtime']} sources temps réel"
            )

            # Détail sources
            with st.expander("Voir le détail des sources", expanded=False):
                for i, source in enumerate(result["sources"][:10], 1):
                    stype  = source["metadata"].get("source_type", "temps réel")
                    sname  = source["metadata"].get("source", "Inconnu")
                    date   = source["metadata"].get("date", "N/A")
                    text   = source["text"][:200]
                    st.markdown(
                        f'<div class="source-box">'
                        f"<strong>Source {i}</strong> ({stype})<br>"
                        f"{sname} — {date}<br>"
                        f"<small>{text}…</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        except Exception as exc:
            st.error(f"Erreur lors de la recherche : {exc}")
            st.error(
                "Vérifiez que toutes les clés API sont configurées "
                "dans le fichier .env"
            )

elif search_button:
    st.warning("Veuillez saisir une question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "OpenDataCopilot — Projet Master 2 Data Science — "
    "Université Paul Valéry Montpellier"
)
