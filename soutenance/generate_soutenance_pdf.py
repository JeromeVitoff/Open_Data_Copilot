#!/usr/bin/env python3
"""
Génère le PDF de préparation à la soutenance OpenDataCopilot
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── Couleurs ──────────────────────────────────────────────────────────────────
BLUE_DARK  = colors.HexColor('#1a3c5e')
BLUE_MED   = colors.HexColor('#2980b9')
BLUE_LIGHT = colors.HexColor('#d6eaf8')
GREEN_DARK = colors.HexColor('#1e6b3e')
GREEN_LIGHT= colors.HexColor('#d5f5e3')
ORANGE     = colors.HexColor('#d35400')
RED        = colors.HexColor('#c0392b')
GRAY_LIGHT = colors.HexColor('#f2f3f4')
GRAY_MED   = colors.HexColor('#bdc3c7')
WHITE      = colors.white
BLACK      = colors.black

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle('CoverTitle',
        fontName='Helvetica-Bold', fontSize=26, textColor=WHITE,
        spaceAfter=8, alignment=TA_CENTER, leading=32))
    styles.add(ParagraphStyle('CoverSubtitle',
        fontName='Helvetica', fontSize=14, textColor=colors.HexColor('#d6eaf8'),
        spaceAfter=6, alignment=TA_CENTER, leading=18))
    styles.add(ParagraphStyle('CoverMeta',
        fontName='Helvetica', fontSize=11, textColor=colors.HexColor('#aed6f1'),
        spaceAfter=4, alignment=TA_CENTER))

    styles.add(ParagraphStyle('SectionTitle',
        fontName='Helvetica-Bold', fontSize=14, textColor=WHITE,
        spaceAfter=2, spaceBefore=14, leading=18))
    styles.add(ParagraphStyle('SubsectionTitle',
        fontName='Helvetica-Bold', fontSize=11, textColor=BLUE_DARK,
        spaceAfter=4, spaceBefore=10, leading=14))
    styles.add(ParagraphStyle('QuestionTitle',
        fontName='Helvetica-Bold', fontSize=10.5, textColor=BLUE_DARK,
        spaceAfter=3, spaceBefore=8, leading=14))
    styles.add(ParagraphStyle('Body',
        fontName='Helvetica', fontSize=9.5, textColor=BLACK,
        spaceAfter=4, leading=14, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('BodyBold',
        fontName='Helvetica-Bold', fontSize=9.5, textColor=BLACK,
        spaceAfter=4, leading=14))
    styles.add(ParagraphStyle('BulletItem',
        fontName='Helvetica', fontSize=9.5, textColor=BLACK,
        spaceAfter=3, leading=13, leftIndent=14, bulletIndent=0))
    styles.add(ParagraphStyle('CodeBlock',
        fontName='Courier', fontSize=8.5, textColor=colors.HexColor('#2c3e50'),
        spaceAfter=3, leading=12, leftIndent=10,
        backColor=colors.HexColor('#f4f6f7')))
    styles.add(ParagraphStyle('Note',
        fontName='Helvetica-Oblique', fontSize=9, textColor=colors.HexColor('#666666'),
        spaceAfter=4, leading=12, leftIndent=10))
    styles.add(ParagraphStyle('Verdict',
        fontName='Helvetica-Bold', fontSize=10, textColor=GREEN_DARK,
        spaceAfter=6, leading=14))
    styles.add(ParagraphStyle('VerdictBad',
        fontName='Helvetica-Bold', fontSize=10, textColor=RED,
        spaceAfter=6, leading=14))
    styles.add(ParagraphStyle('TableHeader',
        fontName='Helvetica-Bold', fontSize=9, textColor=WHITE,
        alignment=TA_CENTER))
    styles.add(ParagraphStyle('TableCell',
        fontName='Helvetica', fontSize=9, textColor=BLACK,
        alignment=TA_LEFT))
    styles.add(ParagraphStyle('TableCellCenter',
        fontName='Helvetica', fontSize=9, textColor=BLACK,
        alignment=TA_CENTER))
    return styles

# ── Helpers ───────────────────────────────────────────────────────────────────
def section_header(title, styles, color=BLUE_DARK):
    """Bande colorée avec titre blanc"""
    tbl = Table([[Paragraph(title, styles['SectionTitle'])]], colWidths=[17*cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', [4]),
    ]))
    return tbl

def question_block(qnum, question, answer_paragraphs, styles):
    """Bloc question + réponse encadrée"""
    items = []
    # En-tête question
    q_tbl = Table([[Paragraph(f"Q{qnum} — {question}", styles['QuestionTitle'])]],
                  colWidths=[17*cm])
    q_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BLUE_LIGHT),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LINEBELOW', (0,0), (-1,-1), 1.5, BLUE_MED),
    ]))
    items.append(q_tbl)
    # Réponse
    for p in answer_paragraphs:
        items.append(p)
    items.append(Spacer(1, 6))
    return KeepTogether(items)

def colored_table(headers, rows, styles, col_widths=None):
    """Table avec en-tête bleue"""
    h = [Paragraph(h, styles['TableHeader']) for h in headers]
    data = [h]
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), styles['TableCell']) for c in row])
    if col_widths is None:
        col_widths = [17*cm / len(headers)] * len(headers)
    tbl = Table(data, colWidths=col_widths)
    style = [
        ('BACKGROUND', (0,0), (-1,0), BLUE_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_MED),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]
    for i in range(1, len(data), 2):
        style.append(('BACKGROUND', (0,i), (-1,i), GRAY_LIGHT))
    tbl.setStyle(TableStyle(style))
    return tbl

# ── PAGE DE GARDE ─────────────────────────────────────────────────────────────
def build_cover(styles):
    elements = []

    # Fond bleu simulé avec un tableau pleine page
    cover_data = [[
        Paragraph("PRÉPARATION À LA SOUTENANCE", styles['CoverTitle']),
    ]]
    cover_tbl = Table(cover_data, colWidths=[17*cm])
    cover_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BLUE_DARK),
        ('TOPPADDING', (0,0), (-1,-1), 40),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ]))
    elements.append(cover_tbl)
    elements.append(Spacer(1, 0.5*cm))

    subtitle_data = [[Paragraph("OpenDataCopilot — LLM + Open Data", styles['CoverSubtitle'])]]
    subtitle_tbl = Table(subtitle_data, colWidths=[17*cm])
    subtitle_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BLUE_MED),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 20),
        ('RIGHTPADDING', (0,0), (-1,-1), 20),
    ]))
    elements.append(subtitle_tbl)
    elements.append(Spacer(1, 1*cm))

    # Bloc description
    desc = [
        "Chatbot RAG multi-domaines (Santé publique + Environnement)",
        "Comparaison de 4 architectures · 3 LLMs · 4 embeddings",
        "70 questions d'évaluation · Données publiques officielles françaises",
        "Master 2 Data Science — Mars 2026",
    ]
    for d in desc:
        elements.append(Paragraph(f"• {d}", styles['CoverMeta']))
    elements.append(Spacer(1, 1.5*cm))

    # Avertissement jury
    jury_data = [[
        Paragraph(
            '"Nous serons particulièrement regardants à la comparaison de votre proposition '
            'avec d\'autres solutions... il ne suffira pas d\'appliquer les algos tout faits."',
            ParagraphStyle('JuryQuote', fontName='Helvetica-Oblique', fontSize=10,
                           textColor=ORANGE, leading=15, alignment=TA_CENTER)),
    ]]
    jury_tbl = Table(jury_data, colWidths=[17*cm])
    jury_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fef9e7')),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
        ('BOX', (0,0), (-1,-1), 1.5, ORANGE),
    ]))
    elements.append(jury_tbl)
    elements.append(Paragraph("— Jérôme Pasquet, directeur du jury",
                               ParagraphStyle('JuryName', fontName='Helvetica-Oblique',
                                              fontSize=9, textColor=GRAY_MED,
                                              alignment=TA_CENTER, spaceAfter=6)))
    elements.append(PageBreak())
    return elements

# ── SECTION 1 : Attentes du jury ──────────────────────────────────────────────
def build_section1(styles):
    elements = []
    elements.append(section_header("1. As-tu répondu aux attentes du jury ?", styles))
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph(
        "Le jury a explicitement demandé une <b>comparaison rigoureuse</b> avec d'autres solutions "
        "(sans RAG, différents algorithmes, plateformes non spécialisées). "
        "Voici l'analyse point par point de ce qui a été réalisé dans le projet.",
        styles['Body']))
    elements.append(Spacer(1, 0.3*cm))

    headers = ["Attente du jury", "Réalisé ?", "Preuve dans le code"]
    rows = [
        ["Méthode sans RAG (baseline)", "✅ OUI", "experiments/baseline/ — LLM seul, 0 retrieval"],
        ["RAG non spécialisé", "✅ OUI", "experiments/rag_basic/ — FAISS + OpenAI naïf"],
        ["Différents algorithmes de retrieval", "✅ OUI", "BM25 + FAISS hybrid, CrossEncoder reranker"],
        ["Comparaison quantitative", "✅ OUI", "70 questions, métriques qualité/latence/coût/hallucination"],
        ["Spécialisation domaine", "✅ OUI", "rag_specialized/ : détection domaine, prompts, embeddings médicaux"],
        ["Plusieurs LLMs testés", "✅ OUI", "GPT-3.5-turbo, Mistral 7B, Llama3 8B (Ollama local)"],
        ["Plusieurs embeddings comparés", "✅ OUI", "OpenAI text-embedding-3-small, CamemBERT-bio, Solon-large"],
        ["Analyse des trade-offs", "✅ OUI", "Sur-spécialisation identifiée : Spécialisé < Optimisé (0.706 vs 0.754)"],
    ]
    elements.append(colored_table(headers, rows, styles, col_widths=[6*cm, 2.5*cm, 8.5*cm]))
    elements.append(Spacer(1, 0.4*cm))

    verdict_tbl = Table([[
        Paragraph(
            "✅ VERDICT : Tu as fait exactement ce qui était demandé. "
            "4 architectures comparées, 3 LLMs, 4 modèles d'embedding, "
            "évaluation quantitative sur 70 questions. "
            "Le jury ne peut pas dire que tu as « juste appliqué des algos tout faits ».",
            styles['Verdict'])
    ]], colWidths=[17*cm])
    verdict_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), GREEN_LIGHT),
        ('BOX', (0,0), (-1,-1), 1.5, GREEN_DARK),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(verdict_tbl)
    elements.append(Spacer(1, 0.5*cm))

    # Tableau des scores réels
    elements.append(Paragraph("Scores réels extraits des JSON d'expériences", styles['SubsectionTitle']))
    score_headers = ["Architecture", "Qualité", "Latence moy.", "Coût ($/query)", "Hallucinations", "Docs"]
    score_rows = [
        ["Baseline (sans RAG)", "~0.300", "1.77s", "$0.00057", "Élevée", "0"],
        ["RAG Basic", "0.710", "1.89s", "$0.00093", "0%", "5"],
        ["RAG Optimisé", "0.754", "4.98s", "$0.00147", "0%", "5"],
        ["RAG Spécialisé", "0.706", "9.39s", "$0.00167", "0%", "5"],
        ["Mistral 7B (Ollama)", "0.757", "4.12s", "~$0", "0%", "5"],
        ["Llama3 8B (Ollama)", "0.760", "4.52s", "~$0", "0%", "5"],
    ]
    elements.append(colored_table(score_headers, score_rows, styles,
                                  col_widths=[4.2*cm, 2*cm, 2.3*cm, 2.5*cm, 3*cm, 3*cm]))
    elements.append(PageBreak())
    return elements

# ── SECTION 2 : Architecture technique ───────────────────────────────────────
def build_section2(styles):
    elements = []
    elements.append(section_header("2. Architecture technique — Vue d'ensemble", styles))
    elements.append(Spacer(1, 0.4*cm))

    arch_headers = ["Composant", "Choix technique", "Justification"]
    arch_rows = [
        ["Vector Store", "FAISS (IndexFlatIP)", "Rapide, local, open-source, 572K vecteurs"],
        ["Dense Embeddings", "OpenAI text-embedding-3-small (1536-dim)", "Meilleur rapport qualité/coût ($0.00002/1K)"],
        ["Sparse Retrieval", "BM25 (k1=1.5, b=0.75, pickle cache)", "Capture termes exacts, dates, codes"],
        ["Hybrid Fusion", "0.6·FAISS + 0.4·BM25", "Combine sémantique + lexical"],
        ["Reranker", "CrossEncoder ms-marco-MiniLM-L-6-v2", "Scores pertinence (query,doc) joints, +précision"],
        ["LLM principal", "GPT-3.5-turbo (temp=0.0)", "Déterministe, $1.5/1M tokens, standard académique"],
        ["LLMs alternatifs", "Mistral 7B + Llama3 8B via Ollama", "0$ coût génération, local, contrôle LLM isolé"],
        ["Détection domaine", "Pattern matching lexical (~0ms)", "Transparent, rapide, sans appel LLM"],
        ["Expansion requête", "Dictionnaire synonymes (~20 termes)", "Améliore recall sans overhead"],
        ["Embeddings médicaux", "CamemBERT-bio (almanach/camembert-bio-base)", "Fine-tuné corpus biomédical français"],
        ["Interface", "Streamlit", "Démo rapide, focus sur le backend RAG"],
    ]
    elements.append(colored_table(arch_headers, arch_rows, styles,
                                  col_widths=[3.8*cm, 5.5*cm, 7.7*cm]))
    elements.append(PageBreak())
    return elements

# ── SECTION 3 : Questions Bloc 1 (Positionnement) ────────────────────────────
def build_section3(styles):
    elements = []
    elements.append(section_header("3. Questions — Positionnement & Valeur ajoutée", styles, BLUE_MED))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (1, "Pourquoi le RAG plutôt que du fine-tuning d'un LLM ?",
         [Paragraph(
             "Le fine-tuning encode les connaissances dans les <b>poids du modèle</b> — figés à la date "
             "d'entraînement, coûteux à réentraîner ($10K–$100K), et sans traçabilité des sources. "
             "Le RAG <b>externalise la connaissance</b> dans une base documentaire mise à jour sans "
             "retoucher le modèle.",
             styles['Body']),
          Paragraph(
              "Pour santé et pollution, les données changent quotidiennement (données hospitalières SPF, "
              "mesures OpenAQ hourly) — le fine-tuning est structurellement inadapté. "
              "De plus, le RAG permet de citer les sources, essentiel pour un usage scientifique.",
              styles['Body'])]),

        (2, "Quelle est la différence avec ChatGPT ?",
         [Paragraph(
             "ChatGPT répond depuis ses paramètres, sans accès aux données actuelles et sans citer de "
             "source vérifiable. Il peut halluciner sur des chiffres précis (taux NO2 à Paris aujourd'hui, "
             "hospitalisations COVID semaine 12).",
             styles['Body']),
          Paragraph(
              "Mon système : <b>corpus fermé et contrôlé</b> (sources officielles françaises), réponse "
              "synthétisée avec citations ligne par ligne <b>[Doc 1] Source: Airparif, 2025-03-27</b>, "
              "0% hallucination mesurée. Pour un usage scientifique ou citoyen, la traçabilité est "
              "non-négociable.",
              styles['Body'])]),

        (3, "Pourquoi deux domaines ensemble (santé + environnement) ?",
         [Paragraph(
             "La vraie valeur est dans les <b>requêtes de corrélation</b> : "
             "<i>« Quel est le lien entre les pics de NO2 et les hospitalisations respiratoires ? »</i>. "
             "Un système mono-domaine ne peut pas répondre à cette question en fusionnant deux sources "
             "hétérogènes.",
             styles['Body']),
          Paragraph(
              "La difficulté technique (détection domaine, scoring diversité, prompts multi-domaine) est "
              "précisément ce qui justifie le travail de recherche. C'est le cœur de la valeur ajoutée.",
              styles['Body'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 4 : Questions techniques ─────────────────────────────────────────
def build_section4(styles):
    elements = []
    elements.append(section_header("4. Questions — Architecture technique", styles))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (4, "Explique la différence entre tes 4 architectures.",
         [Paragraph("<b>Baseline :</b> LLM pur, aucun retrieval. Référence. Résultat : qualité ~0.30, hallucinations élevées.", styles['Body']),
          Paragraph("<b>RAG Basic :</b> FAISS + embeddings OpenAI. Recherche sémantique dense uniquement. Qualité : 0.71, 0% hallucination.", styles['Body']),
          Paragraph("<b>RAG Optimisé :</b> Hybrid retrieval (FAISS 60% + BM25 40%) + CrossEncoder reranker. 20 candidats → 5 finaux. Qualité : 0.754.", styles['Body']),
          Paragraph("<b>RAG Spécialisé :</b> + détection domaine, expansion requêtes, filtre contextuel, prompts domaine-aware. Qualité : 0.706 (sur-spécialisation possible).", styles['Body'])]),

        (5, "Pourquoi BM25 + FAISS ? Pourquoi pas FAISS seul ?",
         [Paragraph(
             "FAISS (bi-encoder) encode requête et documents <b>indépendamment</b>. Il capture bien la "
             "sémantique mais rate les correspondances exactes : si la requête contient "
             "<i>« NO2 Paris 15 mars 2024 »</i>, FAISS peut retourner des documents généraux sans "
             "matcher la date exacte.",
             styles['Body']),
          Paragraph(
              "BM25 (TF-IDF probabiliste, k1=1.5, b=0.75) est parfait pour ce cas. "
              "La fusion pondérée <b>score = 0.6·FAISS + 0.4·BM25</b> combine sémantique + lexical. "
              "C'est l'approche standard en IR moderne (RRF, BEIR benchmark).",
              styles['Body'])]),

        (6, "Comment fonctionne le CrossEncoder ? Pourquoi pas un bi-encoder pour le reranking ?",
         [Paragraph(
             "Un <b>bi-encoder</b> encode query et document séparément → produit scalaire. "
             "Rapide mais moins précis car les deux représentations sont indépendantes.",
             styles['Body']),
          Paragraph(
              "Un <b>CrossEncoder</b> lit la paire (query, document) ensemble via BERT → un seul score "
              "de pertinence direct. Il modélise les interactions fines entre mots des deux textes. "
              "Modèle : <i>cross-encoder/ms-marco-MiniLM-L-6-v2</i>, entraîné sur MS MARCO. "
              "Latence : +100–500ms pour 20 paires, gain de précision justifié.",
              styles['Body'])]),

        (7, "Pourquoi FAISS et pas ChromaDB ou Pinecone ?",
         [Paragraph(
             "FAISS est plus rapide en inférence pure (IndexFlatIP), open-source, sans dépendance réseau. "
             "Pour 572K vecteurs locaux, FAISS est suffisant. "
             "ChromaDB apporte la gestion native des métadonnées — utile en production mais overhead "
             "inutile pour le benchmark. Pinecone est SaaS, coût non-négligeable, hors scope académique.",
             styles['Body'])]),

        (8, "Comment ton système détecte-t-il le domaine d'une requête ?",
         [Paragraph(
             "Approche <b>lexicale</b> (pattern matching sur mots-clés) intentionnellement choisie pour "
             "la vitesse (~0ms) et la transparence.",
             styles['Body']),
          Paragraph(
              "Mots-clés santé : covid, vaccination, hospitalisation, IRA, mortalité… "
              "Mots-clés environnement : NO2, PM10, PM2.5, Airparif, µg/m³… "
              "Corrélation : « impact », « lien », « effet sur », « respiratoire + pollution ».",
              styles['Body']),
          Paragraph(
              "Alternative testée et non retenue : classifier LLM — trop lent (+800ms) et coûteux "
              "pour chaque requête.",
              styles['Note'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 5 : Évaluation & Métriques ───────────────────────────────────────
def build_section5(styles):
    elements = []
    elements.append(section_header("5. Questions — Évaluation & Métriques", styles))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (9, "Ton score de qualité 0.75, c'est quoi exactement ? Comment le calcules-tu ?",
         [Paragraph(
             "Score composite sur 70 questions annotées. Il agrège : "
             "(1) annotation humaine binaire (bonne réponse : 0.7–0.9, mauvaise : 0.3–0.5), "
             "(2) taux de présence de sources citées, "
             "(3) absence d'hallucinations détectées par pattern matching.",
             styles['Body']),
          Paragraph(
              "Ce n'est pas un score BLEU/ROUGE classique — les ground truths sont partiels car "
              "les bonnes réponses sont dynamiques (données temps réel). "
              "<b>C'est une limitation connue et assumée.</b>",
              styles['Body'])]),

        (10, "Pourquoi le RAG Spécialisé (0.706) est moins bon que le RAG Optimisé (0.754) ?",
         [Paragraph(
             "C'est une <b>découverte clé du projet</b>. L'expansion de requêtes ajoute ~4 synonymes "
             "par terme. Sur les questions générales, cela bruite le retrieval. "
             "Le filtre domaine peut aussi éliminer des documents pertinents inter-domaines.",
             styles['Body']),
          Paragraph(
              "<b>Conclusion :</b> la spécialisation apporte +5% sur les requêtes de corrélation, "
              "mais −5% sur les questions mono-domaine génériques. "
              "C'est le trade-off spécialisation/généralisation classique en NLP.",
              styles['Body'])]),

        (11, "Comment tu détectes les hallucinations automatiquement ?",
         [Paragraph(
             "Heuristique pattern-matching en 5 points :",
             styles['Body']),
          Paragraph("• Chiffres précis sans citation source → +0.4", styles['BulletItem']),
          Paragraph("• Patterns statistiques sans support → +0.2 par match", styles['BulletItem']),
          Paragraph("• Assertions catégoriques absolues → +0.3 par match", styles['BulletItem']),
          Paragraph("• Expressions d'incertitude présentes → −0.2", styles['BulletItem']),
          Paragraph("• Citations [Doc X] présentes → −0.1", styles['BulletItem']),
          Paragraph(
              "Score > 0.4 = hallucination suspectée. "
              "<b>Questions-pièges</b> : « Y a-t-il une épidémie de choléra en France ? » — "
              "le baseline invente des détails, le RAG répond « non, aucune donnée dans mes sources ». "
              "Résultat : 0% hallucination sur toutes variantes RAG.",
              styles['Body'])]),

        (12, "Pourquoi 70 questions ? Comment as-tu construit le dataset ?",
         [Paragraph(
             "20 questions initiales annotées manuellement + 50 questions enrichies par catégories "
             "(COVID, démographie médicale, épidémiologie, qualité air, corrélation, pièges). "
             "Catégories équilibrées, difficultés variées (easy/medium/hard).",
             styles['Body']),
          Paragraph(
              "Limite principale : pas de golden answers automatisables pour les données temps réel. "
              "Idéalement, un dataset de référence public (BioASQ, BEIR) renforcerait la comparabilité "
              "externe — à mentionner comme perspective.",
              styles['Note'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 6 : Données & Pipeline ───────────────────────────────────────────
def build_section6(styles):
    elements = []
    elements.append(section_header("6. Questions — Données & Pipeline", styles, BLUE_MED))
    elements.append(Spacer(1, 0.3*cm))

    # Tableau sources
    elements.append(Paragraph("Sources de données utilisées", styles['SubsectionTitle']))
    src_headers = ["Source", "Domaine", "Fréquence", "Format"]
    src_rows = [
        ["data.gouv.fr / SPF", "Santé (COVID, hospitalisations)", "Quotidienne", "CSV/JSON API"],
        ["ODISSE (SPF)", "Épidémiologie, surveillance", "Hebdomadaire", "CSV"],
        ["DREES", "Démographie médicale", "Annuelle", "CSV"],
        ["Airparif", "Qualité air Île-de-France", "Horaire", "API/CSV"],
        ["OpenAQ", "Qualité air mondiale (aggrégateur)", "Temps réel", "API REST"],
        ["Atmo France", "Indices ATMO nationaux", "Quotidienne", "CSV"],
    ]
    elements.append(colored_table(src_headers, src_rows, styles,
                                  col_widths=[3.8*cm, 5*cm, 3*cm, 5.2*cm]))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (13, "Comment tes données sont-elles mises à jour en temps réel ?",
         [Paragraph(
             "Deux mécanismes : (1) <b>API temps réel</b> — OpenAQ et SPF appelées à chaque requête "
             "si la donnée a plus de X heures (cache TTL dans /data/cache_realtime/). "
             "(2) <b>Pipeline batch</b> — scripts data/pipelines/run_all.py planifiables (cron) "
             "pour réindexation incrémentale.",
             styles['Body'])]),

        (14, "Ton index a 572K documents — comment chunkes-tu les données ?",
         [Paragraph(
             "Stratégie <b>« row »</b> : chaque ligne CSV = 1 document, max 512 tokens. "
             "Pour les données tabulaires de santé/pollution, c'est naturel (1 mesure = 1 ligne = 1 chunk). "
             "Chaque chunk porte des métadonnées : source, date, domaine, preview.",
             styles['Body']),
          Paragraph(
              "Alternative non implémentée : chunking par fenêtre glissante pour les textes longs — "
              "non nécessaire ici car les sources sont principalement des données structurées/tabulaires.",
              styles['Note'])]),

        (15, "Quelle est la qualité de tes données sources ?",
         [Paragraph(
             "Sources officielles gouvernementales : data.gouv.fr (SPF, DREES), Airparif (autorité légale "
             "de surveillance qualité air IDF), OpenAQ (aggrégateur international de réseaux officiels). "
             "<b>Pas de scraping Wikipedia ou sources non-vérifiées.</b>",
             styles['Body']),
          Paragraph(
              "Limitations : données ODISSE/SPF avec délais de publication (1–7 jours), "
              "données médicales granulaires avec lacunes sur certaines zones rurales.",
              styles['Note'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 7 : Comparaisons ──────────────────────────────────────────────────
def build_section7(styles):
    elements = []
    elements.append(section_header("7. Questions — Comparaisons avec d'autres approches", styles))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (16, "Pourquoi ne pas utiliser LangChain ou LlamaIndex directement ?",
         [Paragraph(
             "LangChain/LlamaIndex sont des frameworks haut-niveau avec des choix intégrés "
             "(chunking, retrieval, prompts). Pour une étude comparative, il faut <b>contrôler "
             "chaque composant indépendamment</b> — embeddings, retrieval, reranking, prompts — "
             "pour isoler l'impact de chaque variable.",
             styles['Body']),
          Paragraph(
              "Utiliser LangChain masquerait les différences architecturales. "
              "L'implémentation from-scratch est un <b>choix délibéré de rigueur scientifique</b>.",
              styles['Body'])]),

        (17, "Comment se compare ton système à Perplexity AI ou un moteur de recherche classique ?",
         [Paragraph(
             "<b>Perplexity :</b> recherche web généraliste, pas de spécialisation données françaises "
             "officielles, pas de contrôle sur les sources, non reproductible scientifiquement.",
             styles['Body']),
          Paragraph(
              "<b>Moteur de recherche classique :</b> retourne des liens, pas de synthèse.",
              styles['Body']),
          Paragraph(
              "<b>Mon système :</b> corpus fermé et contrôlé (sources officielles françaises), "
              "réponse synthétisée en langage naturel, sources citées ligne par ligne, "
              "0% hallucination mesurée. Cas d'usage différent : réponse fiable avec traçabilité.",
              styles['Body'])]),

        (18, "As-tu comparé des embeddings différents ?",
         [Paragraph(
             "Oui : <b>OpenAI text-embedding-3-small</b> (généraliste, 1536-dim), "
             "<b>CamemBERT-bio</b> (spécialisé biomédical français), "
             "<b>Solon-embeddings-large</b> (optimisé français).",
             styles['Body']),
          Paragraph(
              "Résultat : OpenAI donne les meilleures performances globales (avg relevance 0.52–0.62) "
              "mais CamemBERT-bio performe mieux sur la terminologie médicale pure. "
              "Compromis coût/performance en faveur d'OpenAI pour un déploiement général.",
              styles['Body'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 8 : Limites & Perspectives ───────────────────────────────────────
def build_section8(styles):
    elements = []
    elements.append(section_header("8. Questions — Limites & Perspectives", styles, colors.HexColor('#7f8c8d')))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (19, "Quelles sont les limites de ton système ?",
         [Paragraph("Cinq limites principales identifiées :", styles['Body']),
          Paragraph("<b>1. Évaluation partielle :</b> score qualité manuel sur 70 questions, "
                    "pas de benchmark public de référence (BioASQ, BEIR).", styles['BulletItem']),
          Paragraph("<b>2. Embedding unique :</b> un seul modèle d'embedding pour les deux domaines "
                    "— un embedding spécialisé par domaine pourrait améliorer le recall.", styles['BulletItem']),
          Paragraph("<b>3. Pas de mémoire conversationnelle :</b> chaque requête est indépendante, "
                    "pas de contexte de conversation multi-tour.", styles['BulletItem']),
          Paragraph("<b>4. Latence :</b> 4–9s pour RAG spécialisé — acceptable en démo, "
                    "problématique en production haute-fréquence.", styles['BulletItem']),
          Paragraph("<b>5. Couverture géographique :</b> données Airparif limitées à l'Île-de-France, "
                    "données SPF au niveau national (pas toujours commune par commune).", styles['BulletItem'])]),

        (20, "Que ferais-tu si tu avais 3 mois de plus ?",
         [Paragraph("1. <b>Indexation incrémentale</b> sans reconstruction complète de l'index.", styles['BulletItem']),
          Paragraph("2. <b>GraphRAG</b> — modéliser les relations entre entités (polluant → maladie → région) "
                    "pour des requêtes de corrélation plus riches.", styles['BulletItem']),
          Paragraph("3. <b>Fine-tuning embedding</b> bilingue santé+environnement sur les données du corpus.", styles['BulletItem']),
          Paragraph("4. <b>Évaluation humaine structurée</b> avec plusieurs annotateurs et accord inter-annotateur "
                    "(Cohen's kappa).", styles['BulletItem']),
          Paragraph("5. <b>Déploiement production</b> : Redis pour cache, API FastAPI devant Streamlit.", styles['BulletItem'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 9 : Questions pièges ──────────────────────────────────────────────
def build_section9(styles):
    elements = []
    elements.append(section_header("9. Questions pièges & difficiles", styles, RED))
    elements.append(Spacer(1, 0.3*cm))

    blocks = [
        (21, "Ton score 0.75 est-il statistiquement significatif ?",
         [Paragraph(
             "Honnêtement, 70 questions est un échantillon limité pour une comparaison statistique "
             "rigoureuse (test de Student, intervalles de confiance). Les différences entre architectures "
             "(0.706 vs 0.754 vs 0.760) sont dans une plage de ±5% qui pourrait être du bruit.",
             styles['Body']),
          Paragraph(
              "<b>Pour valider :</b> 200–500 questions et plusieurs runs avec sampling aléatoire. "
              "C'est une limite assumée dans le cadre d'un Master 2.",
              styles['Body'])]),

        (22, "Pourquoi GPT-3.5 et pas GPT-4 ?",
         [Paragraph(
             "Contrainte de coût : GPT-4 coûte ~20× plus cher (input: $30/1M tokens vs $1.5/1M). "
             "Pour 70 questions × 4 architectures = 280 runs, le coût total avec GPT-3.5 est ~$0.30. "
             "Avec GPT-4 : ~$6.",
             styles['Body']),
          Paragraph(
              "De plus, l'hypothèse testée est l'<b>impact de l'architecture RAG</b>, pas la puissance "
              "du LLM — GPT-3.5 est un LLM contrôlé suffisant.",
              styles['Body'])]),

        (23, "La détection d'hallucinations par pattern matching — n'est-ce pas trop simpliste ?",
         [Paragraph(
             "Oui, c'est une heuristique. La détection rigoureuse nécessiterait une vérification "
             "factuelle (NLI — Natural Language Inference) entre la réponse et les documents sources. "
             "Des outils comme SELFCHECKGPT ou FactScore font ça.",
             styles['Body']),
          Paragraph(
              "Le pattern matching est un proxy simple et transparent, utilisé comme outil de "
              "screening. Les questions-pièges manuellement annotées sont le vrai gold standard.",
              styles['Body'])]),

        (24, "Pourquoi ne pas utiliser un LLM-as-a-judge pour évaluer les réponses ?",
         [Paragraph(
             "Approche valide (MT-Bench, Alpaca Eval). Non implémentée par contrainte de coût "
             "(chaque évaluation = un appel supplémentaire) et pour éviter les biais circulaires "
             "(GPT-3.5 évalué par GPT-4 introduit un biais de préférence modèle).",
             styles['Body']),
          Paragraph(
              "Le choix annotation humaine + heuristiques est plus <b>transparent et reproductible</b> "
              "pour un mémoire académique.",
              styles['Body'])]),

        (25, "La RGPD — as-tu pensé aux données personnelles ?",
         [Paragraph(
             "Les sources utilisées sont toutes des données <b>agrégées anonymisées</b> : statistiques "
             "nationales/régionales, pas de données individuelles. SPF publie des taux pour 100K "
             "habitants, Airparif publie des mesures de stations. La RGPD ne s'applique pas au "
             "corpus actuel.",
             styles['Body']),
          Paragraph(
              "Si étendu à des dossiers patients ou données géolocalisées individuelles, "
              "un DPA (Data Processing Agreement) et minimisation des données seraient nécessaires.",
              styles['Note'])]),
    ]

    for qnum, question, answer_pars in blocks:
        elements.append(question_block(qnum, question, answer_pars, styles))

    elements.append(PageBreak())
    return elements

# ── SECTION 10 : Conseils soutenance ─────────────────────────────────────────
def build_section10(styles):
    elements = []
    elements.append(section_header("10. Conseils pour la soutenance", styles, GREEN_DARK))
    elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph("Structure de réponse recommandée", styles['SubsectionTitle']))
    struct_data = [
        [Paragraph("Étape", styles['TableHeader']),
         Paragraph("Action", styles['TableHeader']),
         Paragraph("Exemple", styles['TableHeader'])],
        [Paragraph("1", styles['TableCellCenter']),
         Paragraph("Reconnaître la limite", styles['TableCell']),
         Paragraph("« Oui, 70 questions c'est limité statistiquement... »", styles['TableCell'])],
        [Paragraph("2", styles['TableCellCenter']),
         Paragraph("Expliquer le choix fait", styles['TableCell']),
         Paragraph("« ...contrainte de temps et coût d'annotation »", styles['TableCell'])],
        [Paragraph("3", styles['TableCellCenter']),
         Paragraph("Proposer l'amélioration", styles['TableCell']),
         Paragraph("« Pour valider : 200+ questions, test de Student »", styles['TableCell'])],
    ]
    struct_tbl = Table(struct_data, colWidths=[1.5*cm, 6*cm, 9.5*cm])
    struct_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), GREEN_DARK),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, GRAY_MED),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,2), (-1,2), GRAY_LIGHT),
    ]))
    elements.append(struct_tbl)
    elements.append(Spacer(1, 0.5*cm))

    elements.append(Paragraph("Ce qui distingue un chercheur d'un applicateur d'algos", styles['SubsectionTitle']))
    elements.append(Paragraph(
        "Le jury a dit <i>« il ne suffira pas d'appliquer les algos tout faits »</i>. "
        "Voici ce que tu as fait qui prouve que tu es chercheur :",
        styles['Body']))
    elements.append(Spacer(1, 0.2*cm))

    distinctions = [
        ("Implémentation from-scratch",
         "Pas de LangChain/LlamaIndex — chaque composant contrôlé indépendamment pour isoler les variables."),
        ("Découverte contre-intuitive",
         "RAG Spécialisé < RAG Optimisé (0.706 vs 0.754) : la sur-spécialisation nuit aux requêtes génériques."),
        ("Isolation des variables",
         "RAG Ollama = même retrieval que RAG Basic, LLM différent → prouve que la qualité du retrieval "
         "est le bottleneck principal, pas le LLM."),
        ("Questions-pièges dans l'évaluation",
         "Dataset avec questions adversariales pour mesurer robustesse, pas juste performance optimiste."),
        ("Analyse coût/performance",
         "Comparaison systématique coût $/query, latence, tokens — pas juste la qualité."),
        ("Reconnaître les limites",
         "70 questions = échantillon limité, pattern matching = heuristique — tu les as identifiées."),
    ]

    for title, desc in distinctions:
        row_tbl = Table([[
            Paragraph(f"✓ {title}", styles['BodyBold']),
            Paragraph(desc, styles['Body'])
        ]], colWidths=[4.5*cm, 12.5*cm])
        row_tbl.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (0,-1), 8),
            ('LINEBELOW', (0,0), (-1,-1), 0.3, GRAY_MED),
        ]))
        elements.append(row_tbl)

    elements.append(Spacer(1, 0.5*cm))

    final_tbl = Table([[
        Paragraph(
            "Le jury te cherche sur la maîtrise conceptuelle, pas sur la mémorisation de chiffres. "
            "Si tu ne sais pas, dis : « Je ne sais pas exactement, mais voici ce que je ferais pour "
            "le vérifier... » C'est la réponse d'un chercheur.",
            ParagraphStyle('Final', fontName='Helvetica-Bold', fontSize=10,
                           textColor=WHITE, leading=15, alignment=TA_CENTER)),
    ]], colWidths=[17*cm])
    final_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BLUE_DARK),
        ('TOPPADDING', (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
    ]))
    elements.append(final_tbl)
    return elements

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import os
    os.makedirs('soutenance', exist_ok=True)
    output = 'soutenance/preparation_soutenance_opendatacopilot.pdf'

    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Préparation Soutenance — OpenDataCopilot",
        author="OpenDataCopilot — Master 2 Data Science",
    )

    styles = build_styles()
    story = []
    story += build_cover(styles)
    story += build_section1(styles)
    story += build_section2(styles)
    story += build_section3(styles)
    story += build_section4(styles)
    story += build_section5(styles)
    story += build_section6(styles)
    story += build_section7(styles)
    story += build_section8(styles)
    story += build_section9(styles)
    story += build_section10(styles)

    doc.build(story)
    print(f"✅  PDF généré : {output}")

if __name__ == '__main__':
    main()
