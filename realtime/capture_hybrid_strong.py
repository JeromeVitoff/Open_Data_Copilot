#!/usr/bin/env python3
"""
Teste questions hybrides FORTES qui nécessitent fusion de sources
"""
from playwright.sync_api import sync_playwright
import time
import re

# NOUVELLES QUESTIONS - FUSION OBLIGATOIRE
QUESTIONS = [
    "Que montrent les données sur la qualité de l'air et la santé à Paris après le premier confinement en juin 2020 ?",
    "Quel est le lien entre les pics de pollution NO2, et les hospitalisations respiratoires à Paris?",
    "Quel est le lien entre les pics de pollution NO2, et les hospitalisations respiratoires en 2025 à Paris?",
    "Quelles observations peut-on faire sur la santé et la qualité de l'air à Paris au printemps 2020 ?",
    "Évolution de la grippe cette saison?",
    "Combien de médecins pour 1000 habitants à Montpellier ?",
    "Quelles sont les principales observations sur la santé publique et la pollution atmosphérique à Paris en 2020 ?",
    "Comment ont évolué la qualité de l'air et les indicateurs de santé COVID à Paris en 2021-2022 ?",
]

def count_sources_advanced(page_content):
    doc_pattern = r'\[Doc \d+\]'
    docs = re.findall(doc_pattern, page_content)

    health_keywords = [
        'spf', 'covid', 'hospitalisation', 'vaccination',
        'cas positifs', 'décès', 'réanimation', 'incidence',
        'santé publique france', 'odisse', 'épidémio'
    ]
    pollution_keywords = [
        'airparif', 'openaq', 'no2', 'pm10', 'pm2.5', 'o3',
        'dioxyde azote', 'particules fines', 'ozone',
        'qualité air', 'polluant', 'concentration'
    ]

    content_lower = page_content.lower()
    health_mentions   = sum(1 for kw in health_keywords   if kw in content_lower)
    pollution_mentions = sum(1 for kw in pollution_keywords if kw in content_lower)

    return {
        'num_docs':          len(set(docs)),
        'has_health':        health_mentions >= 2,
        'has_pollution':     pollution_mentions >= 2,
        'health_strength':   health_mentions,
        'pollution_strength': pollution_mentions,
    }


def check_real_fusion(page):
    try:
        expander = page.locator('summary:has-text("Voir le détail des sources")')
        if expander.count() > 0:
            expander.first.click()
            time.sleep(1)

        content = page.content()
        response_text = page.locator('[data-testid="stMarkdown"]').all_text_contents()
        full_response = ' '.join(response_text).lower()

        fusion_indicators = [
            'confinement' in full_response and ('pollution' in full_response or 'no2' in full_response),
            'covid' in full_response and ('qualité air' in full_response or 'airparif' in full_response),
            ('hospitalisation' in full_response or 'vaccination' in full_response) and 'pollution' in full_response,
        ]
        return content, full_response, any(fusion_indicators)

    except Exception as e:
        print(f"⚠️  Erreur extraction : {e}")
        return "", "", False


def test_question(page, question, index):
    print(f"\n{'='*70}")
    print(f"TEST {index+1}/{len(QUESTIONS)}")
    print(f"{'='*70}")
    print(f"Q: {question[:65]}...")

    try:
        input_field = page.locator('input[type="text"]').first
        input_field.fill(question)
        time.sleep(1)

        page.locator('button:has-text("Rechercher")').first.click()
        page.wait_for_selector('[data-testid="stSpinner"]', state='hidden', timeout=60000)
        time.sleep(12)

        page_content, response_text, has_real_fusion = check_real_fusion(page)
        sources_info = count_sources_advanced(page_content)

        print(f"📊 Docs cités      : {sources_info['num_docs']}")
        print(f"🏥 Santé           : {'✅' if sources_info['has_health'] else '❌'} ({sources_info['health_strength']} mentions)")
        print(f"🌫️  Pollution       : {'✅' if sources_info['has_pollution'] else '❌'} ({sources_info['pollution_strength']} mentions)")
        print(f"🔗 Fusion réelle   : {'✅' if has_real_fusion else '❌'}")

        score = sources_info['num_docs'] * 5
        if sources_info['has_health']:   score += 20
        if sources_info['has_pollution']: score += 20
        if sources_info['has_health'] and sources_info['has_pollution']: score += 40
        if has_real_fusion: score += 50
        score += sources_info['health_strength'] * 2
        score += sources_info['pollution_strength'] * 2

        print(f"⭐ SCORE           : {score}")

        return {
            'question':          question,
            'index':             index,
            'score':             score,
            'num_docs':          sources_info['num_docs'],
            'has_both':          sources_info['has_health'] and sources_info['has_pollution'],
            'has_real_fusion':   has_real_fusion,
            'health_strength':   sources_info['health_strength'],
            'pollution_strength': sources_info['pollution_strength'],
        }

    except Exception as e:
        print(f"❌ Erreur : {e}")
        return {'question': question, 'index': index, 'score': 0, 'num_docs': 0,
                'has_both': False, 'has_real_fusion': False,
                'health_strength': 0, 'pollution_strength': 0}
    finally:
        page.reload()
        page.wait_for_selector('text=OpenDataCopilot', timeout=10000)
        time.sleep(3)


def capture_best_hybrid():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        page.goto('http://localhost:8502')
        page.wait_for_selector('text=OpenDataCopilot', timeout=10000)
        time.sleep(3)

        results = []
        for i, question in enumerate(QUESTIONS):
            result = test_question(page, question, i)
            results.append(result)
            time.sleep(2)

        # Classement
        print(f"\n{'='*70}")
        print(f"CLASSEMENT QUESTIONS")
        print(f"{'='*70}")
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

        for i, r in enumerate(sorted_results[:5], 1):
            fusion_icon = '✅' if r['has_real_fusion'] else '❌'
            print(f"\n{i}. Score {r['score']} - Docs: {r['num_docs']} - Fusion: {fusion_icon}")
            print(f"   {r['question'][:60]}...")

        best = sorted_results[0]

        print(f"\n{'='*70}")
        print(f"🏆 MEILLEURE QUESTION")
        print(f"{'='*70}")
        print(f"Question  : {best['question']}")
        print(f"Score     : {best['score']}")
        print(f"Docs      : {best['num_docs']}")
        print(f"Fusion    : {'✅' if best['has_real_fusion'] else '❌'}")
        print(f"Santé     : {best['health_strength']} mentions")
        print(f"Pollution : {best['pollution_strength']} mentions")

        # Capture
        out_png = 'realtime/screenshots/07_question_hybride_sante_pollution.png'
        out_txt = 'realtime/screenshots/07_question.txt'

        if best['score'] >= 80:
            print(f"\n🎬 CAPTURE EN COURS...")
            input_field = page.locator('input[type="text"]').first
            input_field.fill(best['question'])
            time.sleep(1)
            page.locator('button:has-text("Rechercher")').first.click()
            page.wait_for_selector('[data-testid="stSpinner"]', state='hidden', timeout=60000)
            time.sleep(15)

            try:
                expander = page.locator('summary:has-text("Voir le détail des sources")')
                if expander.count() > 0:
                    expander.first.click()
                    time.sleep(2)
            except Exception:
                pass

            page.evaluate('window.scrollTo(0, 400)')
            time.sleep(2)
            page.screenshot(path=out_png)
            print(f"✅ Capture : {out_png}")

        else:
            # Capture quand même (seuil non atteint mais meilleur disponible)
            print(f"\n⚠️  Score {best['score']} < 80 — capture quand même du meilleur résultat")
            input_field = page.locator('input[type="text"]').first
            input_field.fill(best['question'])
            time.sleep(1)
            page.locator('button:has-text("Rechercher")').first.click()
            page.wait_for_selector('[data-testid="stSpinner"]', state='hidden', timeout=60000)
            time.sleep(15)
            page.evaluate('window.scrollTo(0, 400)')
            time.sleep(2)
            page.screenshot(path=out_png)
            print(f"✅ Capture (sous seuil) : {out_png}")

        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(best['question'])
        print(f"✅ Question sauvegardée : {out_txt}")

        browser.close()
        return best, sorted_results


if __name__ == '__main__':
    print("=" * 70)
    print("RECHERCHE QUESTION HYBRIDE FORTE (FUSION OBLIGATOIRE)")
    print("=" * 70)

    best, all_results = capture_best_hybrid()

    print(f"\n{'='*70}")
    print(f"RÉSULTAT FINAL")
    print(f"{'='*70}")
    print(f"Question retenue : {best['question']}")
    print(f"Score fusion     : {best['score']}")
    print(f"Fusion réelle    : {'OUI ✅' if best['has_real_fusion'] else 'NON ❌'}")
    print(f"Capture          : realtime/screenshots/07_question_hybride_sante_pollution.png")
    print("=" * 70)
