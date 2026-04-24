# OpenDataCopilot

**Chatbot RAG multi-domaines sur données ouvertes françaises — Santé publique & Pollution atmosphérique**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Projet Master 2 Data Science MIASHS — Université Paul Valéry Montpellier 3  
> Jérôme ADJIMON — jerome.vitoff@etu.umontpellier.fr

---

## Description

OpenDataCopilot est un chatbot RAG (Retrieval-Augmented Generation) qui permet d'interroger en langage naturel **1 222 802 documents** issus des données ouvertes françaises sur la santé publique et la qualité de l'air. Le projet adopte une démarche comparative rigoureuse entre 4 architectures RAG, 3 LLMs et 4 modèles d'embeddings.

### Corpus de données

| Source | Domaine | Volume |
|--------|---------|--------|
| Santé Publique France (SPF) — COVID | Santé | 452 K documents |
| Airparif | Qualité de l'air Île-de-France | 720 K documents |
| ODISSE | Épidémiologie | 50 K documents |
| **Total** | **2019–2023** | **1 222 802 documents** |

---

## Résultats

### Comparaison des architectures (70 questions annotées)

| Architecture | Qualité | Latence | Coût total | Hallucinations |
|-------------|---------|---------|------------|----------------|
| Baseline (LLM seul) | 0.300 | 1,8 s | $0,011 | 0 % |
| RAG Basic (FAISS dense) | 0.743 | 2,9 s | $0,091 | 0 % |
| RAG Optimisé (BM25+FAISS+Rerank) | **0.754** | 5,0 s | $0,103 | 0 % |
| RAG Spécialisé (multi-domaine) | 0.706 | 9,4 s | $0,117 | 0 % |
| RAG + Mistral 7B (local) | **0.757** | 4,1 s | ~$0 | 0 % |
| RAG + Llama3 8B (local) | 0.760 | 4,5 s | ~$0 | 0 % |

**Système recommandé : RAG Optimisé + Mistral 7B** — meilleur rapport qualité/coût, déploiement local RGPD-compliant.

### Découvertes scientifiques

1. **Diversité > Volume** : +114 % de documents n'apporte que +18,9 % de pertinence
2. **Open-source compétitif** : Mistral 7B ≈ GPT-3.5-turbo, ×500 moins cher
3. **Complexité ≠ Performance** : la sur-ingénierie (RAG Spécialisé) réduit la qualité de 6 %
4. **Tâche > Domaine** : OpenAI généraliste surpasse CamemBERT-bio (−78 % de pertinence)
5. **RAG = 0 % hallucination** : validé sur toutes les architectures et 70 questions
6. **20 → 70 questions** : l'augmentation du dataset révèle les vraies performances

---

## Architecture du projet

```
OpenDataCopilot/
├── experiments/                  # 4 architectures RAG comparées
│   ├── baseline/                 # LLM seul (GPT-3.5-turbo), sans contexte
│   ├── rag_basic/                # FAISS dense + GPT-3.5 + text-embedding-3-small
│   ├── rag_optimized/            # BM25 + FAISS hybride + reranking cross-encoder
│   ├── rag_specialized/          # Détection de domaine + filtrage contextuel
│   └── rag_specialized_v2/       # Embeddings médicaux (CamemBERT-bio, BioMistral)
├── realtime/                     # Prototype hybride temps réel
│   ├── app.py                    # Interface Streamlit
│   ├── temporal_detector.py      # Détection questions temps réel vs historique
│   ├── hybrid_rag.py             # Fusion FAISS historique + APIs temps réel
│   └── cache_manager.py          # Cache (×4 performances)
├── evaluation/                   # Comparaison des versions
│   └── compare_all_versions.py
├── notebooks/                    # Exploration des données
├── data/                         # Données et rapports de téléchargement
├── docs/                         # Analyse baseline
├── poster/                       # Poster scientifique A0 (LaTeX)
└── soutenance/                   # Supports de soutenance
```

---

## Expériences

### 1. Baseline — LLM seul

- **Modèle** : GPT-3.5-turbo, sans contexte RAG
- **Résultat** : score de qualité 0.30 — le LLM renvoie l'utilisateur vers des sources externes, sans données précises

### 2. RAG Basic

- **Retrieval** : FAISS dense (index 1 222 802 docs)
- **Embeddings** : `text-embedding-3-small` (OpenAI)
- **LLM** : GPT-3.5-turbo
- **Résultat** : qualité 0.743, sources citées 100 %, 0 % hallucination

### 3. RAG Optimisé

- **Retrieval** : recherche hybride BM25 + FAISS (`hybrid_alpha=0.6`), top-20 puis reranking
- **Reranker** : `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-5 final
- **Embeddings** : `text-embedding-3-small`
- **LLM** : GPT-3.5-turbo
- **Résultat** : qualité 0.754, meilleure architecture GPT-3.5

### 4. RAG Spécialisé

- **Retrieval** : détecteur de domaine (santé / environnement / corrélation), filtrage contextuel, prompts spécialisés par domaine
- **LLM** : GPT-3.5-turbo
- **Résultat** : qualité 0.706 — la sur-spécialisation dégrade les performances (−6 % vs RAG Optimisé)

### 5. RAG Spécialisé v2 — Embeddings médicaux

- Test des embeddings Sentence-CamemBERT, Solon, CamemBERT-bio
- Test de BioMistral 7B (LLM médical)
- **Résultat** : OpenAI généraliste surpasse CamemBERT-bio (−78 % de pertinence)

### 6. RAG Ollama — LLMs locaux

- **Modèles** : Mistral 7B et Llama3 8B via Ollama local
- **Résultat** : qualité 0.757 (Mistral) et 0.760 (Llama3), coût quasi nul (~$0.002 total), 0 % hallucination

---

## Prototype hybride temps réel

Le module `realtime/` fusionne données historiques et APIs temps réel :

```
Question
   │
   ▼
Détecteur temporel (19/19 tests OK)
   ├── Question historique → FAISS (1,2M docs)
   └── Question temps réel → 3 APIs (SPF · Airparif · OpenAQ)
                    │
                    ▼
               Fusion + Cache (×4 perf.)
                    │
                    ▼
            LLM (Mistral / GPT) → Réponse sourcée
```

**Interface** : Streamlit (`realtime/app.py`)

---

## Dataset d'évaluation

70 questions annotées réparties en 3 domaines :

| Catégorie | Questions |
|-----------|-----------|
| Santé (COVID, vaccination, épidémiologie) | 30 |
| Qualité de l'air (NO2, PM10, O3) | 20 |
| Corrélations santé–pollution | 20 |
| **Total** | **70** |

**Métriques** : score de qualité (utilité jury), pertinence FAISS, taux d'hallucination, latence, coût/question.

---

## Installation

### Prérequis

- Python 3.10+
- 8 GB RAM minimum
- Clé API OpenAI
- [Ollama](https://ollama.ai) pour les LLMs locaux (optionnel)

### Mise en place

```bash
# 1. Cloner le repository
git clone https://github.com/JeromeVitoff/Open_Data_Copilot.git
cd Open_Data_Copilot

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

### Configuration `.env`

```env
OPENAI_API_KEY=sk-...
AIRPARIF_API_KEY=...      # optionnel, pour le temps réel
OPENAQ_API_KEY=...        # optionnel, pour le temps réel
```

---

## Utilisation

### Lancer l'interface Streamlit

```bash
streamlit run realtime/app.py
```

### Exécuter une expérience

```bash
# Baseline
python experiments/baseline/run_baseline.py

# RAG Basic
python experiments/rag_basic/run_rag_basic.py

# RAG Optimisé
python experiments/rag_optimized/run_evaluation.py

# RAG Spécialisé
python experiments/rag_specialized/run_evaluation.py

# RAG Ollama (nécessite Ollama en local)
python experiments/rag_ollama/run_evaluation.py
```

### Comparer les résultats

```bash
python evaluation/compare_all_versions.py
```

---

## Sources de données

| Source | Description | Fréquence |
|--------|-------------|-----------|
| [Santé Publique France](https://www.santepubliquefrance.fr) | Données COVID-19, épidémiologie | Hebdomadaire |
| [ODISSE (SPF)](https://odisse.santepubliquefrance.fr) | Surveillance épidémiologique | Hebdomadaire |
| [Airparif](https://data-airparif-asso.opendata.arcgis.com) | Qualité air Île-de-France | Horaire |
| [OpenAQ](https://openaq.org) | Pollution mondiale | Temps réel |

---

## Licence

MIT License

---

## Auteur

**Jérôme ADJIMON** — Master 2 Data Science MIASHS, Université Paul Valéry Montpellier 3

---

## Remerciements

- Santé Publique France pour les données épidémiologiques
- Airparif pour les données de qualité de l'air
- OpenAQ pour l'API pollution mondiale
