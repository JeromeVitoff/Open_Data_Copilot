"""
Génère les captures écran de l'app Streamlit via Playwright.
Pré-requis : app tournant sur http://localhost:8501
"""
import asyncio
import sys
from playwright.async_api import async_playwright

BASE = "http://localhost:8502"
OUT  = "realtime/screenshots"

SCENARIOS = [
    {
        "name": "01_accueil",
        "desc": "Page d'accueil — état initial",
        "question": None,
    },
    {
        "name": "02_question_historique",
        "desc": "Question historique — COVID mars 2021",
        "question": "Hospitalisations COVID à Paris en mars 2021",
    },
    {
        "name": "03_question_temps_reel",
        "desc": "Question temps réel — qualité air aujourd'hui",
        "question": "Qualité de l'air à Paris aujourd'hui",
    },
    {
        "name": "04_question_hybride",
        "desc": "Question hybride — évolution NO2",
        "question": "Évolution de la pollution NO2 ces dernières années à Paris",
    },
    {
        "name": "05_pollution_actuelle",
        "desc": "Pollution temps réel Île-de-France",
        "question": "Niveau de pollution atmosphérique actuellement en Île-de-France",
    },
]


async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})

        for scenario in SCENARIOS:
            name  = scenario["name"]
            desc  = scenario["desc"]
            q     = scenario["question"]

            print(f"\n📸 Capture : {name} — {desc}")

            # Charger la page fraîche
            await page.goto(BASE, wait_until="networkidle")
            await asyncio.sleep(2)

            if q is None:
                # Juste la page d'accueil
                await page.screenshot(path=f"{OUT}/{name}.png", full_page=True)
                print(f"   ✅ {name}.png")
                continue

            # Remplir la question — sélecteur compatible Streamlit 1.54
            input_sel = 'input[type=text]'
            await page.wait_for_selector(input_sel, timeout=15000)
            inputs = await page.query_selector_all(input_sel)
            target = inputs[-1]  # Dernier input = champ question
            await target.click(click_count=3)  # Sélectionne tout
            await target.type(q)
            await asyncio.sleep(0.5)

            # Cliquer "Envoyer 🚀"
            btns = await page.query_selector_all('button')
            send_btn = None
            for b in btns:
                txt = await b.inner_text()
                if "Rechercher" in txt:
                    send_btn = b
                    break
            if not send_btn:
                print(f"   ❌ Bouton 'Envoyer' introuvable")
                continue
            await send_btn.click()

            # Attendre la fin du traitement :
            # 1) Spinner apparaît → 2) Spinner disparaît → 3) Résultats affichés
            print(f"   ⏳ Attente spinner…")
            # Attendre que le spinner apparaisse
            await page.wait_for_selector('[data-testid="stSpinner"]', timeout=30_000)
            print(f"   ⏳ Spinner actif, attente réponse LLM (max 4 min)…")
            # Attendre que le spinner disparaisse (fin du traitement)
            await page.wait_for_selector(
                '[data-testid="stSpinner"]',
                state="hidden",
                timeout=240_000,
            )
            # Laisser le temps au DOM de se stabiliser
            await asyncio.sleep(2)
            await asyncio.sleep(1)

            await page.screenshot(path=f"{OUT}/{name}.png", full_page=True)
            print(f"   ✅ {name}.png")

        await browser.close()
        print(f"\n✅ {len(SCENARIOS)} captures sauvegardées dans {OUT}/")


if __name__ == "__main__":
    asyncio.run(run())
