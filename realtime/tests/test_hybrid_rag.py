"""
Tests RAG Hybride — valide la fusion historique + temps réel.
Lance avec :
    source venv/bin/activate && set -a && source .env && set +a
    python -m realtime.tests.test_hybrid_rag
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime.hybrid_rag import HybridRAG

SEP  = "=" * 70
SEP2 = "-" * 70

# ── Cas de test ───────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    {
        "question": "Hospitalisations COVID à Paris en mars 2021",
        "expected_type": "historical",
        "desc": "Question historique → FAISS uniquement",
    },
    {
        "question": "Qualité de l'air à Paris aujourd'hui",
        "expected_type": "realtime",
        "desc": "Question temps réel → FAISS + APIs pollution",
    },
    {
        "question": "Évolution de la pollution NO2 ces dernières années à Paris",
        "expected_type": "mixed",
        "desc": "Question mixte → FAISS + APIs",
    },
]


def print_result(i: int, test: dict, result: dict) -> None:
    ta = result["temporal_analysis"]
    print(f"\n{'='*70}")
    print(f"TEST {i} — {test['expected_type'].upper()} : {test['desc']}")
    print("="*70)
    print(f"  Type détecté       : {ta['type']}")
    print(f"  Score RT           : {ta['realtime_score']:.2f}")
    print(f"  Sources historiques: {result['num_historical']}")
    print(f"  Sources temps réel : {result['num_realtime']}")
    print(f"  Total documents    : {len(result['sources'])}")

    print(f"\n💬 RÉPONSE :")
    print(result["answer"])

    print(f"\n📚 SOURCES (5 premières) :")
    for j, src in enumerate(result["sources"][:5], 1):
        stype  = src["metadata"].get("source_type", "?")
        source = src["metadata"].get("source", "?")
        date   = src["metadata"].get("date", "N/A")
        print(f"  {j}. [{stype}] {source} ({date})")
        print(f"     {src['text'][:100]}…")


def validate(result: dict, expected_type: str) -> bool:
    ta = result["temporal_analysis"]
    ok = True

    if expected_type == "historical":
        if result["num_historical"] == 0:
            print("  ❌ Aucun document historique récupéré")
            ok = False
        if result["num_realtime"] > 0:
            print(f"  ⚠️  {result['num_realtime']} doc(s) temps réel inattendus (type historique)")

    elif expected_type == "realtime":
        if not ta["needs_realtime_data"]:
            print(f"  ⚠️  Détecteur n'a pas déclenché temps réel (score={ta['realtime_score']:.2f})")
        if result["num_historical"] == 0:
            print("  ❌ Aucun document historique (FAISS toujours consulté)")
            ok = False

    elif expected_type == "mixed":
        if result["num_historical"] == 0:
            print("  ❌ Aucun document historique")
            ok = False

    if result["answer"] and len(result["answer"]) > 20:
        print("  ✅ Réponse LLM générée")
    else:
        print("  ❌ Réponse LLM vide ou trop courte")
        ok = False

    return ok


# ── Test cache ────────────────────────────────────────────────────────────────

def test_cache(rag: HybridRAG) -> None:
    print(f"\n{SEP}")
    print("TEST CACHE")
    print(SEP)

    question = "Qualité de l'air à Paris actuellement"
    print("Premier appel (remplissage cache)…")
    rag.query(question, k_historical=2, k_realtime=3)

    print("\nDeuxième appel (lecture cache)…")
    import time
    t0 = time.time()
    rag.query(question, k_historical=2, k_realtime=3)
    elapsed = time.time() - t0
    print(f"  ✅ Deuxième appel terminé en {elapsed:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("TEST RAG HYBRIDE")
    print(SEP)

    print("\n🔧 Initialisation HybridRAG…")
    rag = HybridRAG()

    results_ok = []
    for i, test in enumerate(TEST_QUESTIONS, 1):
        result = rag.query(test["question"], k_historical=3, k_realtime=5)
        print_result(i, test, result)
        ok = validate(result, test["expected_type"])
        results_ok.append(ok)
        print(f"\n  {'✅ OK' if ok else '❌ ÉCHEC'}")

    test_cache(rag)

    print(f"\n{SEP}")
    print("BILAN")
    print(SEP)
    total = len(results_ok)
    passed = sum(results_ok)
    print(f"  {passed}/{total} tests validés")
    for i, (test, ok) in enumerate(zip(TEST_QUESTIONS, results_ok), 1):
        status = "✅" if ok else "❌"
        print(f"  {status} Test {i} — {test['desc']}")
    print(f"\n{'✅ JOUR 2 TERMINÉ !' if passed == total else '⚠️  Certains tests à revoir'}")
    print(SEP)
