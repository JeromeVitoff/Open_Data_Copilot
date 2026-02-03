#!/usr/bin/env python3
"""
Script de téléchargement GÉOD'AIR optimisé pour respecter le rate limit.

RATE LIMIT: 15 requêtes/heure

Stratégie:
- 1 ville = 1 polluant = 2 requêtes (génération + téléchargement)
- Maximum 7 villes par heure avec 1 polluant
- Attente longue entre génération et téléchargement (30s)
- Pas de retry excessif

Usage:
    python data/pipelines/download_geodair_safe.py
    python data/pipelines/download_geodair_safe.py --city Paris
    python data/pipelines/download_geodair_safe.py --test
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "pollution"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GEODAIR_BASE_URL = "https://www.geodair.fr/api-ext"

VILLES = {
    "Paris": "75",
    "Lyon": "69",
    "Marseille": "13",
    "Toulouse": "31",
    "Bordeaux": "33",
    "Lille": "59",
    "Montpellier": "34",
    "Strasbourg": "67",
    "Nantes": "44",
    "Nice": "06",
}

# Polluant NO2 uniquement pour économiser le rate limit
POLLUANT = "03"  # NO2


def get_headers():
    """Retourne les headers pour l'API GÉOD'AIR."""
    api_key = os.getenv("GEODAIR_API_KEY")
    if not api_key:
        raise ValueError("GEODAIR_API_KEY non configurée dans .env")

    return {
        "accept": "text/csv; charset=UTF-8",
        "apikey": api_key,
        "User-Agent": "OpenDataCopilot/1.0",
    }


def test_connection() -> bool:
    """Teste la connexion et vérifie le rate limit."""
    print("🔍 Test connexion API GÉOD'AIR...")

    try:
        response = httpx.get(
            f"{GEODAIR_BASE_URL}/polluant/export",
            headers=get_headers(),
            timeout=15.0,
        )

        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            print(f"✅ Connexion OK - {len(lines)-1} polluants disponibles")
            return True
        elif response.status_code == 429:
            print("❌ Rate limit atteint - Attendez 1 heure")
            return False
        else:
            print(f"❌ Erreur HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def download_city(ville: str, departement: str, days: int = 30) -> pd.DataFrame | None:
    """
    Télécharge les données pour une ville.

    Utilise 2 requêtes: 1 génération + 1 téléchargement.
    """
    headers = get_headers()

    date_fin = datetime.now()
    date_debut = date_fin - timedelta(days=days)

    params = {
        "date_debut": date_debut.strftime("%d/%m/%Y 00:00"),
        "date_fin": date_fin.strftime("%d/%m/%Y 23:59"),
        "type_donnee": "a2",  # Moyenne journalière
        "polluant": POLLUANT,
        "departement": departement,
    }

    print(f"\n📍 {ville} (dép. {departement})")
    print(f"   Période: {date_debut.strftime('%Y-%m-%d')} → {date_fin.strftime('%Y-%m-%d')}")

    # Étape 1: Génération du fichier
    print("   1️⃣ Génération du fichier...")
    try:
        response = httpx.get(
            f"{GEODAIR_BASE_URL}/statistique/export",
            headers=headers,
            params=params,
            timeout=60.0,
        )

        if response.status_code == 429:
            print("   ❌ Rate limit atteint sur génération")
            return None
        elif response.status_code != 200:
            print(f"   ❌ Erreur génération: HTTP {response.status_code}")
            return None

        file_id = response.text.strip()
        print(f"   ✅ File ID: {file_id[:50]}...")

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None

    # Étape 2: Attendre que le fichier soit prêt
    print("   2️⃣ Attente 30 secondes...")
    time.sleep(30)

    # Étape 3: Téléchargement
    print("   3️⃣ Téléchargement...")
    try:
        response = httpx.get(
            f"{GEODAIR_BASE_URL}/download",
            headers=headers,
            params={"id": file_id},
            timeout=90.0,
        )

        if response.status_code == 429:
            print("   ❌ Rate limit atteint sur téléchargement")
            print("   💡 Conseil: Attendez 1 heure et réessayez")
            return None
        elif response.status_code == 404:
            print("   ❌ Fichier non trouvé (expiré?)")
            return None
        elif response.status_code != 200:
            print(f"   ❌ Erreur: HTTP {response.status_code}")
            return None

        content = response.text.strip()
        lines = content.split("\n")

        if len(lines) > 1 and ";" in lines[0]:
            print(f"   ✅ CSV reçu: {len(lines)} lignes")

            # Parser le CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), sep=";", encoding="utf-8-sig")
            return df
        else:
            print(f"   ⚠️ Contenu non-CSV: {content[:100]}...")
            return None

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Télécharge les données GÉOD'AIR")
    parser.add_argument("--city", "-c", help="Ville spécifique à télécharger")
    parser.add_argument("--test", "-t", action="store_true", help="Test connexion uniquement")
    parser.add_argument("--days", "-d", type=int, default=30, help="Nombre de jours")
    parser.add_argument("--all", "-a", action="store_true", help="Toutes les villes (attention rate limit!)")

    args = parser.parse_args()

    print("=" * 60)
    print("🌬️ GÉOD'AIR - Téléchargement sécurisé")
    print("=" * 60)
    print(f"⚠️  Rate limit: 15 requêtes/heure")
    print(f"📊 Polluant: NO2 uniquement (pour économiser le quota)")
    print()

    # Test connexion
    if not test_connection():
        print("\n❌ Connexion impossible ou rate limit atteint")
        print("   Réessayez dans 1 heure.")
        return 1

    if args.test:
        return 0

    # Déterminer les villes à télécharger
    if args.city:
        if args.city not in VILLES:
            print(f"❌ Ville '{args.city}' non supportée")
            print(f"   Villes disponibles: {list(VILLES.keys())}")
            return 1
        cities = {args.city: VILLES[args.city]}
    elif args.all:
        print("\n⚠️  Téléchargement de toutes les villes")
        print("   Cela nécessite ~20 requêtes (dépasse le rate limit!)")
        print("   Le script s'arrêtera si le rate limit est atteint.")
        cities = VILLES
    else:
        # Par défaut: 5 premières villes (10 requêtes)
        cities = dict(list(VILLES.items())[:5])
        print(f"\n📍 Villes à télécharger: {list(cities.keys())}")
        print(f"   (utilisez --all pour toutes les villes)")

    # Téléchargement
    results = {}
    total_records = 0

    for ville, departement in cities.items():
        df = download_city(ville, departement, days=args.days)

        if df is not None:
            # Sauvegarder
            month = datetime.now().strftime("%Y-%m")
            filepath = DATA_DIR / f"geodair_{ville.lower()}_{month}.csv"
            df.to_csv(filepath, index=False, sep=";", encoding="utf-8-sig")

            results[ville] = len(df)
            total_records += len(df)
            print(f"   💾 Sauvegardé: {filepath.name}")
        else:
            results[ville] = 0

        # Pause entre les villes (respect rate limit)
        if ville != list(cities.keys())[-1]:
            print("\n   ⏳ Pause 5 secondes...")
            time.sleep(5)

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)

    success = sum(1 for v in results.values() if v > 0)
    failed = len(results) - success

    print(f"✅ Réussis: {success}")
    print(f"❌ Échoués: {failed}")
    print(f"📈 Total enregistrements: {total_records}")

    if results:
        print("\nDétail par ville:")
        for ville, count in results.items():
            status = "✅" if count > 0 else "❌"
            print(f"   {status} {ville}: {count} mesures")

    print(f"\n📁 Fichiers dans: {DATA_DIR}")

    return 0 if success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
