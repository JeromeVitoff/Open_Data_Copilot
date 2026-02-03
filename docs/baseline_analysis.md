# Analyse de la Baseline Sans RAG

## OpenDataCopilot - Évaluation des Performances LLM

**Date d'évaluation :** 3 février 2026
**Version :** 1.0.0
**Auteur :** Projet Master 2 Data Science

---

## 1. Contexte du Projet

### 1.1 Objectif

**OpenDataCopilot** est un assistant conversationnel intelligent spécialisé dans les **données ouvertes françaises** sur deux domaines critiques :

| Domaine | Description | Sources |
|---------|-------------|---------|
| **Santé Publique** | Hospitalisations COVID, urgences, démographie médicale | Santé Publique France |
| **Qualité de l'Air** | Pollution NO₂, PM2.5, PM10, O₃ en Île-de-France | Airparif |

### 1.2 Problématique

Les LLMs actuels présentent des **limites importantes** pour les questions factuelles :

```
┌─────────────────────────────────────────────────────────────┐
│                    PROBLÈMES DES LLMs                       │
├─────────────────────────────────────────────────────────────┤
│  ❌ Données figées (cutoff de formation)                    │
│  ❌ Pas d'accès aux données en temps réel                   │
│  ❌ Risque d'hallucination sur les chiffres                 │
│  ❌ Impossibilité de citer des sources vérifiables          │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Hypothèse

> **H₀** : Un LLM sans contexte documentaire (RAG) ne peut pas répondre utilement aux questions factuelles sur des données récentes.

---

## 2. Méthodologie

### 2.1 Configuration du Modèle

| Paramètre | Valeur |
|-----------|--------|
| **Modèle** | GPT-3.5-Turbo |
| **Temperature** | 0.0 (déterministe) |
| **Max Tokens** | 500 |
| **Coût Input** | $0.0015 / 1K tokens |
| **Coût Output** | $0.002 / 1K tokens |

### 2.2 Prompt Système

Le prompt système a été optimisé pour **minimiser les hallucinations** :

```text
Tu es un assistant qui répond aux questions sur la santé
publique et la pollution en France.

IMPORTANT :
- Si tu ne connais pas une information précise, dis-le clairement
- Ne fournis JAMAIS de chiffres ou statistiques sans être
  absolument certain
- Pour les données récentes (2024-2025), admets les limites
  de tes connaissances
- Préfère dire "je ne sais pas" plutôt que d'inventer des données

Rappel : Tes connaissances s'arrêtent à ta date de formation.
Pour des données en temps réel, indique que tu ne peux pas y accéder.
```

### 2.3 Dataset de Test

**20 questions annotées** couvrant différentes catégories :

```
                    RÉPARTITION DES QUESTIONS

    COVID-19          ████████░░░░░░░░░░░░  10% (2)
    Épidémiologie     ████████████░░░░░░░░  15% (3)
    Démographie méd.  ████████░░░░░░░░░░░░  10% (2)
    Qualité de l'air  ████████░░░░░░░░░░░░  10% (2)
    Réglementation    ████████░░░░░░░░░░░░  10% (2)
    Historique        ████████░░░░░░░░░░░░  10% (2)
    Corrélations      ████████░░░░░░░░░░░░  10% (2)
    Autres            ██████████████░░░░░░  25% (5)
```

### 2.4 Métriques d'Évaluation

| Métrique | Description |
|----------|-------------|
| **Latence** | Temps de réponse (ms) |
| **Tokens** | Consommation input + output |
| **Coût** | Estimation en USD |
| **Hallucination** | Détection automatique basée sur patterns |

#### Algorithme de Détection d'Hallucination

```python
def detect_hallucination(response, has_sources):
    score = 0.0

    # +0.4 si nombreux chiffres précis sans source
    # +0.2 par statistique précise (µg/m³, %, millions)
    # +0.3 par affirmation catégorique
    # -0.3 par expression d'incertitude

    return score >= 0.4  # Seuil d'hallucination
```

---

## 3. Résultats Quantitatifs

### 3.1 Métriques Globales

```
╔══════════════════════════════════════════════════════════════╗
║                 RÉSULTATS BASELINE SANS RAG                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Questions traitées     ████████████████████  20            ║
║                                                              ║
║   Temps moyen            1 772 ms                            ║
║   Tokens totaux          6 965                               ║
║   Coût total             $0.0114                             ║
║                                                              ║
║   ┌──────────────────────────────────────────────────────┐   ║
║   │  TAUX D'HALLUCINATION                                │   ║
║   │                                                      │   ║
║   │           ░░░░░░░░░░░░░░░░░░░░  0%                   │   ║
║   │                                                      │   ║
║   │  0 hallucinations détectées sur 20 questions         │   ║
║   └──────────────────────────────────────────────────────┘   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 3.2 Tableau Détaillé

| Métrique | Valeur | Écart-type |
|----------|--------|------------|
| Latence moyenne | 1 772 ms | ±523 ms |
| Tokens par question | 348 | ±32 |
| Coût par question | $0.00057 | - |
| Score hallucination moyen | 0.02 | ±0.06 |

### 3.3 Résultats par Catégorie

| Catégorie | Questions | Halluc. | Taux | Comportement |
|-----------|-----------|---------|------|--------------|
| COVID-19 | 2 | 0 | 0% | ⚠️ Refuse de répondre |
| Épidémiologie | 3 | 0 | 0% | ✅ Répond aux générales |
| Démographie médicale | 2 | 0 | 0% | ⚠️ Refuse de répondre |
| Qualité de l'air | 2 | 0 | 0% | ⚠️ Refuse de répondre |
| Réglementation | 2 | 0 | 0% | ✅ Donne des infos |
| Historique | 2 | 0 | 0% | ⚠️ Refuse de répondre |
| Corrélations | 2 | 0 | 0% | ✅ Explique les mécanismes |

### 3.4 Distribution des Temps de Réponse

```
    Latence (ms)

    3000+ │                          █
    2500  │ █                   █
    2000  │ █    █         █    █  █
    1500  │ █ █  █  █   █  █ █  █  █ █  █
    1000  │ █ █  █  █ █ █  █ █  █  █ █  █ █
     500  │ █ █  █  █ █ █  █ █  █  █ █  █ █
       0  └──────────────────────────────────
          Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 ...  Q20
```

---

## 4. Exemples de Réponses

### 4.1 ✅ Réponse Prudente (Type Majoritaire)

**Question :** *Combien de personnes sont actuellement hospitalisées pour COVID-19 en France ?*

> **Réponse LLM :**
> Je n'ai pas accès aux données en temps réel. Pour obtenir le nombre actuel de personnes hospitalisées pour COVID-19 en France, je vous recommande de consulter les sources officielles telles que le site de Santé publique France ou data.gouv.fr.

**Analyse :**
- ✅ Admet ne pas avoir l'information
- ✅ Redirige vers des sources fiables
- ❌ **N'apporte aucune valeur ajoutée**

---

### 4.2 ✅ Réponse avec Connaissances Générales

**Question :** *Les enfants sont-ils plus vulnérables à la pollution que les adultes ?*

> **Réponse LLM :**
> En général, les enfants sont en effet plus vulnérables à la pollution que les adultes. Leurs organes en développement, leur système immunitaire encore immature et leur tendance à respirer plus rapidement les exposent davantage aux effets néfastes des polluants atmosphériques.

**Analyse :**
- ✅ Répond avec des connaissances établies
- ✅ Pas de chiffres inventés
- ⚠️ Réponse générique sans données locales

---

### 4.3 ⚠️ Réponse Partiellement Utile

**Question :** *Quels sont les seuils d'alerte pollution pour les PM2.5 en France ?*

> **Réponse LLM :**
> Les seuils d'alerte pollution pour les PM2.5 en France sont les suivants :
> - 25 µg/m³ en moyenne journalière
> - 50 µg/m³ en moyenne annuelle
>
> Cependant, veuillez noter que ces seuils peuvent varier...

**Analyse :**
- ⚠️ Donne des chiffres (potentiellement obsolètes)
- ⚠️ Score hallucination : 0.20 (sous le seuil)
- ❌ Les seuils réglementaires évoluent

---

## 5. Analyse Critique

### 5.1 Pourquoi 0% d'Hallucination N'est PAS Suffisant

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    LE PARADOXE DE LA PRUDENCE EXCESSIVE                        │
│                                                                 │
│    ┌─────────────┐         ┌─────────────┐                     │
│    │ Hallucine   │         │ Dit "je ne  │                     │
│    │ des données │   VS    │ sais pas"   │                     │
│    └─────────────┘         └─────────────┘                     │
│         ❌ Faux            ❌ Inutile                          │
│                                                                 │
│    Les deux cas sont des ÉCHECS pour l'utilisateur             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Taux de réponses "Je ne sais pas" : ~70%**

Le modèle refuse de répondre à la majorité des questions factuelles, ce qui le rend **inutile** pour son cas d'usage prévu.

### 5.2 Limites de l'Approche Sans RAG

| Limite | Impact | Exemple |
|--------|--------|---------|
| **Pas de données récentes** | Impossible de répondre sur 2024-2026 | COVID, pollution actuelle |
| **Pas de sources citables** | Réponses non vérifiables | Aucune référence |
| **Connaissances génériques** | Pas de spécificité locale | Pas de données Île-de-France |
| **Cutoff temporel** | Information potentiellement obsolète | Seuils réglementaires |

### 5.3 Matrice Utilité vs Fiabilité

```
                    UTILITÉ DE LA RÉPONSE

            Faible            Élevée
           ┌─────────────────┬─────────────────┐
    Haute  │   BASELINE      │    OBJECTIF     │
 F         │   ACTUELLE      │    RAG          │
 I         │                 │                 │
 A         │  "Je ne sais    │  Données        │
 B         │   pas"          │  + Sources      │
 I         │                 │                 │
 L         ├─────────────────┼─────────────────┤
 I         │   ÉCHEC         │   DANGER        │
 T         │   TOTAL         │                 │
 É         │                 │                 │
           │  Erreurs +      │  Hallucinations │
    Basse  │  Pas d'info     │  crédibles      │
           └─────────────────┴─────────────────┘
```

### 5.4 Justification du Besoin de RAG

```
╔═══════════════════════════════════════════════════════════════╗
║                 POURQUOI LE RAG EST ESSENTIEL                 ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. FRAÎCHEUR DES DONNÉES                                     ║
║     └─► Données actualisées quotidiennement                   ║
║                                                               ║
║  2. TRAÇABILITÉ                                               ║
║     └─► Sources citées et vérifiables                         ║
║                                                               ║
║  3. PRÉCISION LOCALE                                          ║
║     └─► Données spécifiques Île-de-France                     ║
║                                                               ║
║  4. RÉDUCTION DES HALLUCINATIONS                              ║
║     └─► Le LLM s'appuie sur des documents réels               ║
║                                                               ║
║  5. CONFIANCE UTILISATEUR                                     ║
║     └─► Réponses avec preuves documentaires                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 6. Prochaines Étapes

### 6.1 Roadmap d'Implémentation

```
    PHASE 1              PHASE 2              PHASE 3
    ────────             ────────             ────────

    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │BASELINE │   ──►   │  RAG    │   ──►   │  RAG    │
    │Sans RAG │         │ Basique │         │Optimisé │
    └─────────┘         └─────────┘         └─────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐         ┌─────────┐
    │0% util. │         │~60% ut. │         │~85% ut. │
    │0% hall. │         │~15% hal.│         │~5% hall.│
    └─────────┘         └─────────┘         └─────────┘

    ✅ FAIT              À FAIRE              À FAIRE
```

### 6.2 Comparaison des Approches Prévues

| Caractéristique | Baseline | RAG Basique | RAG Optimisé |
|-----------------|----------|-------------|--------------|
| **Retrieval** | ❌ Aucun | ✅ Dense (embeddings) | ✅ Hybrid (BM25 + Dense) |
| **Reranking** | ❌ Non | ❌ Non | ✅ Cohere Rerank |
| **Chunking** | N/A | Fixe (512 tokens) | Sémantique |
| **Sources** | ❌ Non | ✅ Top-k docs | ✅ Filtrées + scorées |
| **Coût estimé** | $0.0006/q | $0.002/q | $0.005/q |

### 6.3 Métriques Cibles

| Métrique | Baseline | Cible RAG |
|----------|----------|-----------|
| Taux d'hallucination | 0%* | < 10% |
| Taux de réponses utiles | ~30% | > 80% |
| Précision factuelle | N/A | > 85% |
| Sources citées | 0% | 100% |

*\* 0% car refuse de répondre*

---

## 7. Conclusion

### Synthèse des Résultats

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   La baseline sans RAG démontre que :                          │
│                                                                 │
│   ✅ Un prompt bien conçu PEUT éviter les hallucinations       │
│                                                                 │
│   ❌ MAIS au prix d'une utilité quasi-nulle                    │
│                                                                 │
│   ══════════════════════════════════════════════════════════   │
│                                                                 │
│   Le RAG est INDISPENSABLE pour un assistant factuel           │
│   sur les données ouvertes françaises.                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Recommandation

> **Implémenter un système RAG** avec retrieval sur les données Santé Publique France et Airparif pour transformer cet assistant "prudent mais inutile" en un **outil réellement utile** pour les utilisateurs.

---

## Annexes

### A. Fichiers Générés

| Fichier | Description |
|---------|-------------|
| `experiments/baseline/baseline_rag.py` | Implémentation baseline |
| `experiments/baseline/config.py` | Configuration modèle |
| `experiments/baseline/run_baseline.py` | Script d'évaluation |
| `experiments/baseline/results/baseline_report.json` | Résultats complets |

### B. Données de Test

- **Source** : `evaluation/datasets/questions_annotees.json`
- **Nombre** : 20 questions annotées
- **Catégories** : 7 (COVID, pollution, épidémiologie, etc.)

### C. Reproductibilité

```bash
# Exécuter l'évaluation baseline
cd OpenDataCopilot
python -m experiments.baseline.run_baseline

# Résultats dans :
# experiments/baseline/results/baseline_report.json
```

---

**Document généré le 3 février 2026**
**OpenDataCopilot v1.0.0**
