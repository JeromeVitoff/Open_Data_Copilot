# OpenDataCopilot

**Chatbot RAG intelligent sur données ouvertes françaises - Santé publique & Environnement**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Description

OpenDataCopilot est un projet de recherche Master 2 Data Science qui combine un chatbot intelligent, une architecture RAG (Retrieval-Augmented Generation), et des données ouvertes françaises sur deux domaines :

- **Santé publique** : données hospitalières, épidémiologie, démographie médicale
- **Environnement** : pollution atmosphérique (NO2, PM2.5, O3), qualité de l'air

### Objectif scientifique

Ce projet adopte une **démarche comparative rigoureuse** entre 4 architectures RAG :

| Architecture | Description | Vector Store | Embeddings |
|-------------|-------------|--------------|------------|
| **Baseline** | LLM seul sans RAG | - | - |
| **RAG Basic** | Chunking simple + recherche vectorielle | FAISS | OpenAI ada-002 |
| **RAG Optimized** | Chunking intelligent + hybrid search + reranking | ChromaDB | OpenAI + BM25 |
| **RAG Specialized** | Fine-tuning embeddings domaine santé/climat | ChromaDB | SBERT médical |

### Métriques d'évaluation

- Précision et Recall sur questions annotées
- Taux d'hallucination
- Citation accuracy (vérification des sources)
- Latence de réponse
- Coût API

---

## Fonctionnalités

- Chatbot conversationnel multidomaine (santé + pollution)
- Croisement automatique des données (ex: pollution → hospitalisations)
- Citations systématiques des sources officielles
- Interface utilisateur intuitive (Streamlit)
- API REST documentée (FastAPI)
- Pipeline de données automatisé avec cache
- Dashboard de visualisation temps réel

### Exemples de questions

```
"Quel est le taux de pollution NO2 à Paris aujourd'hui ?"
"Évolution de la grippe cette saison en Île-de-France ?"
"Lien entre pics de pollution et hospitalisations respiratoires à Montpellier ?"
"Combien de médecins pour 1000 habitants dans l'Hérault ?"
```

---

## Architecture du projet

```
OpenDataCopilot/
├── src/                          # Code source principal
│   ├── api/                      # FastAPI backend
│   ├── ui/                       # Streamlit frontend
│   ├── core/                     # Interfaces RAG communes
│   ├── config/                   # Configuration Pydantic
│   └── utils/                    # Utilitaires partagés
├── experiments/                  # 4 implémentations RAG
│   ├── baseline/                 # LLM seul
│   ├── rag_basic/               # RAG simple FAISS
│   ├── rag_optimized/           # RAG avancé ChromaDB
│   └── rag_specialized/         # RAG fine-tuné
├── data/                         # Données
│   ├── raw/                     # Données brutes
│   ├── processed/               # Données nettoyées
│   ├── pipelines/               # Scripts ETL
│   └── vectorstore/             # Bases vectorielles
├── evaluation/                   # Benchmarks
│   ├── datasets/                # Questions annotées
│   ├── metrics/                 # Scripts métriques
│   └── results/                 # Résultats
├── notebooks/                    # Analyses Jupyter
├── docs/                         # Documentation
├── tests/                        # Tests pytest
└── scripts/                      # CLI utilitaires
```

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- macOS / Linux / Windows
- 8 GB RAM minimum
- Clés API : OpenAI, Airparif (optionnel), OpenAQ (optionnel)

### Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/OpenDataCopilot.git
cd OpenDataCopilot

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 5. Télécharger les données initiales
python -m data.pipelines.download_sante_publique

# 6. Lancer l'application
python -m scripts.run_app
```

### Configuration

Créer un fichier `.env` à la racine :

```env
# LLM
OPENAI_API_KEY=sk-...

# APIs données
AIRPARIF_API_KEY=...
OPENAQ_API_KEY=...

# Optionnel
COHERE_API_KEY=...  # Pour reranking
```

---

## Utilisation

### Lancer le chatbot (Streamlit)

```bash
streamlit run src/ui/app.py
```

### Lancer l'API (FastAPI)

```bash
uvicorn src.api.main:app --reload --port 8000
```

Documentation API : http://localhost:8000/docs

### Exécuter les benchmarks

```bash
# Comparer les 4 architectures
python -m scripts.run_benchmarks --all

# Benchmark spécifique
python -m scripts.run_benchmarks --architecture rag_optimized
```

### Notebooks d'exploration

```bash
jupyter lab notebooks/
```

---

## Sources de données

### Santé publique

| Source | Description | Fréquence |
|--------|-------------|-----------|
| [data.gouv.fr](https://www.data.gouv.fr) | Données hospitalières COVID-19 | Quotidien |
| [Santé Publique France](https://odisse.santepubliquefrance.fr) | Épidémiologie, surveillance | Hebdomadaire |
| [DREES](https://data.drees.solidarites-sante.gouv.fr) | Démographie médicale | Annuel |

### Environnement / Pollution

| Source | Description | Fréquence |
|--------|-------------|-----------|
| [Airparif](https://data-airparif-asso.opendata.arcgis.com) | Qualité air Île-de-France | Horaire |
| [OpenAQ](https://openaq.org) | Pollution mondiale | Temps réel |
| [Atmo France](https://www.atmo-france.org) | Indices ATMO nationaux | Quotidien |

---

## Développement

### Tests

```bash
# Tous les tests
pytest

# Avec couverture
pytest --cov=src --cov-report=html

# Tests unitaires uniquement
pytest tests/unit/
```

### Linting & Formatting

```bash
# Vérification
ruff check .

# Correction automatique
ruff check --fix .

# Formatage
ruff format .
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## Évaluation scientifique

### Dataset d'évaluation

Le fichier `evaluation/datasets/questions_annotees.json` contient 50-100 questions annotées :

```json
{
  "id": "Q001",
  "question": "Quel est le taux de NO2 à Paris aujourd'hui ?",
  "domain": "pollution",
  "difficulty": "easy",
  "expected_sources": ["airparif"],
  "ground_truth": "...",
  "requires_realtime": true
}
```

### Métriques implémentées

- **Retrieval** : Precision@k, Recall@k, MRR, NDCG
- **Generation** : ROUGE-L, BERTScore, factual consistency
- **Hallucination** : % réponses sans source valide
- **Performance** : latence p50/p95/p99, tokens/seconde

---

## Roadmap

- [x] Phase 1 : Structure projet et pipeline données
- [ ] Phase 2 : Baseline sans RAG
- [ ] Phase 3 : RAG Basic (FAISS)
- [ ] Phase 4 : RAG Optimized (ChromaDB + reranking)
- [ ] Phase 5 : RAG Specialized (fine-tuning)
- [ ] Phase 6 : Benchmarks comparatifs
- [ ] Phase 7 : Interface finale et documentation

---

## Contribution

Ce projet est développé dans le cadre d'un Master 2 Data Science. Les contributions sont bienvenues :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit (`git commit -m 'Add: nouvelle fonctionnalité'`)
4. Push (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

---

## Licence

MIT License - voir [LICENSE](LICENSE)

---

## Auteur

**Jérôme** - Master 2 Data Science

---

## Remerciements

- Santé Publique France pour les données épidémiologiques
- Airparif pour les données de qualité de l'air
- OpenAQ pour l'API pollution mondiale
- LangChain pour le framework RAG
