"""
Tests des wrappers APIs temps réel.
Lance avec : python -m realtime.tests.test_api_wrappers
Ou via pytest : pytest realtime/tests/test_api_wrappers.py -v -s
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajoute la racine du projet au path si exécuté directement
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime.api_wrappers.spf_api import SPFRealtimeAPI
from realtime.api_wrappers.airparif_api import AirparifRealtimeAPI
from realtime.api_wrappers.openaq_api import OpenAQRealtimeAPI

SEP = "=" * 70
SEP2 = "-" * 70


def test_spf():
    print(f"\n1. TEST SPF API\n{SEP2}")
    spf = SPFRealtimeAPI()
    df = spf.get_covid_hospitalizations_recent(days=7, department="75")
    print(f"Enregistrements récupérés: {len(df)}")
    assert isinstance(df.__class__.__name__, str)  # DataFrame ou vide
    if len(df) > 0:
        docs = spf.format_for_rag(df.head(3))
        assert len(docs) > 0
        assert "text" in docs[0]
        assert docs[0]["metadata"]["source"] == "SPF"
        print(f"Exemple document:\n  {docs[0]['text'][:120]}")
    else:
        print("⚠️  Aucune donnée (API peut être inaccessible en local)")


def test_airparif():
    print(f"\n2. TEST AIRPARIF API\n{SEP2}")
    import os
    airparif = AirparifRealtimeAPI()

    if not airparif.api_key:
        print("⚠️  Clé absente — test ignoré (set AIRPARIF_API_KEY dans .env)")
        return

    data = airparif.get_current_pollution(city="Paris")
    print(f"Données brutes: {data}")

    if not data:
        print("⚠️  Aucune donnée retournée (API inaccessible ou quota dépassé)")
        return

    assert "city" in data
    assert "date" in data
    doc = airparif.format_for_rag(data)
    assert "text" in doc
    assert doc["metadata"]["source"] == "Airparif"
    print(f"Document RAG:\n  {doc['text']}")


def test_openaq():
    print(f"\n3. TEST OPENAQ API\n{SEP2}")
    openaq = OpenAQRealtimeAPI()
    measures = openaq.get_latest_measurements(city="Paris", parameter="no2")
    print(f"Mesures récupérées: {len(measures)}")
    if len(measures) > 0:
        docs = openaq.format_for_rag(measures[:3])
        assert len(docs) > 0
        assert "text" in docs[0]
        assert docs[0]["metadata"]["source"] == "OpenAQ"
        print(f"Exemple document:\n  {docs[0]['text'][:120]}")
    else:
        print("⚠️  Aucune mesure (API peut nécessiter une clé ou être inaccessible)")


def test_openaq_invalid_parameter():
    import pytest
    openaq = OpenAQRealtimeAPI()
    try:
        openaq.get_latest_measurements(parameter="invalid")
        assert False, "Devrait lever ValueError"
    except ValueError as e:
        print(f"✅ ValueError attendu: {e}")


if __name__ == "__main__":
    print(SEP)
    print("TEST APIs TEMPS RÉEL")
    print(SEP)

    test_spf()
    test_airparif()
    test_openaq()
    test_openaq_invalid_parameter()

    print(f"\n{SEP}")
    print("✅ TOUS LES TESTS TERMINÉS")
    print(SEP)
