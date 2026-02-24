"""
OpenDataCopilot - Pipeline ODISSE (Santé publique France)
=========================================================

Télécharge les datasets de santé publique depuis le portail ODISSE
(https://odisse.santepubliquefrance.fr) via l'API ODS v2.1.

Datasets ciblés (ordonnés par volume) :
1. covid-19-synthese-des-indicateurs-de-suivi-de-la-pandemie-dep  (121 200 rec)
2. arboviroses-donnees-declaration-obligatoire                      ( 44 016 rec)
3. infections-sexuellement-transmissibles-*-dep                     ( 33 732 rec)
4. infections-respiratoires-aigues-ira-*-region                    ( 28 800 rec)
5. maladie-veineuse-thrombo-embolique-incidence-hospitaliere-dep   ( 25 650 rec)
6. traumatisme-passages-aux-urgences-et-actes-sos-medecins-region  ( 23 040 rec)
7. vih-depistages-rembourses-departement                            ( 16 995 rec)
8. antibiotiques-consommation-en-medecine-de-ville-departement     ( 14 400 rec)
9. maladies-cardio-neuro-vasculaires-taux-standardises-epci         ( 11 250 rec)
10. gestes-auto-infliges-hospitalisations-departement               ( 11 124 rec)

Total estimé : ~330 000 nouveaux documents

Usage:
    python -m data.pipelines.fetch_odisse
    python -m data.pipelines.fetch_odisse --force
    python -m data.pipelines.fetch_odisse --list
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sante_odisse"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

ODISSE_BASE_URL = "https://odisse.santepubliquefrance.fr"
ODISSE_API = f"{ODISSE_BASE_URL}/api/explore/v2.1"

# Datasets prioritaires (ID ODISSE → métadonnées)
PRIORITY_DATASETS: list[dict[str, Any]] = [
    {
        "id": "covid-19-synthese-des-indicateurs-de-suivi-de-la-pandemie-dep",
        "name": "COVID-19 indicateurs de suivi par département",
        "theme": "covid",
        "expected_records": 121200,
        "priority": 1,
    },
    {
        "id": "arboviroses-donnees-declaration-obligatoire",
        "name": "Arboviroses (Dengue, Chikungunya, Zika) par département",
        "theme": "maladies_infectieuses",
        "expected_records": 44016,
        "priority": 2,
    },
    {
        "id": "infections-sexuellement-transmissibles-donnees-de-depistages-rembourses-dep",
        "name": "IST - Dépistages remboursés par département",
        "theme": "maladies_infectieuses",
        "expected_records": 33732,
        "priority": 3,
    },
    {
        "id": "infections-respiratoires-aigues-ira-passages-aux-urgences-et-actes-sos-medecins-region",
        "name": "IRA - Passages urgences et SOS Médecins par région",
        "theme": "maladies_infectieuses",
        "expected_records": 28800,
        "priority": 4,
    },
    {
        "id": "maladie-veineuse-thrombo-embolique-incidence-hospitaliere-dep",
        "name": "Maladie veineuse thrombo-embolique - incidence hospitalière par département",
        "theme": "maladies_chroniques",
        "expected_records": 25650,
        "priority": 5,
    },
    {
        "id": "traumatisme-passages-aux-urgences-et-actes-sos-medecins-region",
        "name": "Traumatismes - Passages urgences et SOS Médecins par région",
        "theme": "traumatismes",
        "expected_records": 23040,
        "priority": 6,
    },
    {
        "id": "vih-depistages-rembourses-departement",
        "name": "VIH - Dépistages remboursés par département",
        "theme": "maladies_infectieuses",
        "expected_records": 16995,
        "priority": 7,
    },
    {
        "id": "antibiotiques-consommation-en-medecine-de-ville-departement",
        "name": "Antibiotiques - Consommation en médecine de ville par département",
        "theme": "medicaments",
        "expected_records": 14400,
        "priority": 8,
    },
    {
        "id": "maladies-cardio-neuro-vasculaires-taux-standardises-epci",
        "name": "Maladies cardio-neuro-vasculaires - Taux standardisés par EPCI",
        "theme": "maladies_chroniques",
        "expected_records": 11250,
        "priority": 9,
    },
    {
        "id": "gestes-auto-infliges-hospitalisations-departement",
        "name": "Gestes auto-infligés - Hospitalisations par département",
        "theme": "sante_mentale",
        "expected_records": 11124,
        "priority": 10,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_metadata() -> dict[str, Any]:
    """Charge les métadonnées de téléchargement existantes."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloads": {}, "last_updated": None}


def _save_metadata(metadata: dict[str, Any]) -> None:
    """Sauvegarde les métadonnées de téléchargement."""
    metadata["last_updated"] = datetime.now().isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _get_record_count(dataset_id: str) -> int:
    """Récupère le nombre de records d'un dataset via l'API."""
    url = f"{ODISSE_API}/catalog/datasets/{dataset_id}/records"
    try:
        resp = requests.get(url, params={"limit": 1}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("total_count", 0)
    except Exception as e:
        logger.warning(f"Impossible de récupérer le compte pour {dataset_id}: {e}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Téléchargement
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(
    dataset: dict[str, Any],
    output_dir: Path,
    force: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """
    Télécharge un dataset ODISSE au format CSV.

    Args:
        dataset: Métadonnées du dataset (id, name, theme, ...)
        output_dir: Répertoire de sortie
        force: Force le re-téléchargement même si le fichier existe
        session: Session requests optionnelle

    Returns:
        Résultat: {"status": "success/skipped/error", "records": N, "file": path}
    """
    dataset_id = dataset["id"]
    output_file = output_dir / f"{dataset_id}.csv"

    # Vérifier si déjà téléchargé
    if output_file.exists() and not force:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"   ⏭️  {dataset_id[:50]}")
        logger.info(f"       → Déjà présent ({size_mb:.1f} MB), skip (--force pour forcer)")
        return {
            "status": "skipped",
            "file": str(output_file),
            "size_mb": size_mb,
        }

    if session is None:
        session = requests.Session()

    # URL d'export CSV (toujours tous les records)
    export_url = (
        f"{ODISSE_API}/catalog/datasets/{dataset_id}/exports/csv"
        f"?delimiter=%3B&use_labels=true&lang=fr"
    )

    logger.info(f"   ⬇️  {dataset['name'][:60]}")
    logger.info(f"       ID: {dataset_id[:60]}")
    logger.info(f"       Attendu: ~{dataset.get('expected_records', '?'):,} records")

    start_time = time.time()

    try:
        resp = session.get(
            export_url,
            stream=True,
            timeout=(30, 300),  # connect=30s, read=300s (gros fichiers)
            headers={"Accept": "text/csv"},
        )
        resp.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)

        # Téléchargement en streaming avec barre de progression
        total_bytes = 0
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(output_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

        elapsed = time.time() - start_time
        size_mb = total_bytes / (1024 * 1024)
        speed_mbps = size_mb / elapsed if elapsed > 0 else 0

        # Compter les lignes (approximation du nombre de records)
        with open(output_file, "r", encoding="utf-8-sig", errors="replace") as f:
            num_lines = sum(1 for _ in f) - 1  # -1 pour l'en-tête

        logger.info(
            f"       ✅ {size_mb:.1f} MB | {num_lines:,} records | "
            f"{elapsed:.1f}s | {speed_mbps:.1f} MB/s"
        )

        return {
            "status": "success",
            "file": str(output_file),
            "size_mb": size_mb,
            "records": num_lines,
            "elapsed_s": elapsed,
        }

    except requests.exceptions.Timeout:
        logger.error(f"       ❌ Timeout pour {dataset_id}")
        if output_file.exists():
            output_file.unlink()
        return {"status": "error", "error": "Timeout (> 300s)"}

    except requests.exceptions.HTTPError as e:
        logger.error(f"       ❌ HTTP {e.response.status_code} pour {dataset_id}")
        return {"status": "error", "error": f"HTTP {e.response.status_code}"}

    except Exception as e:
        logger.error(f"       ❌ Erreur inattendue: {e}")
        if output_file.exists():
            output_file.unlink()
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def download_all(
    force: bool = False,
    datasets: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Télécharge tous les datasets ODISSE prioritaires.

    Args:
        force: Force le re-téléchargement
        datasets: Liste de datasets (défaut: PRIORITY_DATASETS)

    Returns:
        Résultats: {"success": [...], "skipped": [...], "failed": [...], "stats": {...}}
    """
    if datasets is None:
        datasets = PRIORITY_DATASETS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata()

    results: dict[str, list] = {"success": [], "skipped": [], "failed": []}
    total_records = 0

    logger.info(f"\n📊 ODISSE : {len(datasets)} datasets à traiter")
    logger.info(f"   Sortie : {OUTPUT_DIR}")
    logger.info(f"   Forcer : {'OUI' if force else 'NON'}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "OpenDataCopilot/1.0 (academic research)",
    })

    for i, dataset in enumerate(datasets, 1):
        logger.info(f"\n[{i}/{len(datasets)}] Priorité {dataset.get('priority', '?')}")
        result = download_dataset(dataset, OUTPUT_DIR, force=force, session=session)

        dataset_id = dataset["id"]

        if result["status"] == "success":
            results["success"].append(dataset_id)
            total_records += result.get("records", 0)
            metadata["downloads"][dataset_id] = {
                "name": dataset["name"],
                "theme": dataset["theme"],
                "file": result["file"],
                "size_mb": result.get("size_mb", 0),
                "records": result.get("records", 0),
                "downloaded_at": datetime.now().isoformat(),
            }
        elif result["status"] == "skipped":
            results["skipped"].append(dataset_id)
            # Compter les records du fichier existant
            output_file = OUTPUT_DIR / f"{dataset_id}.csv"
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8-sig", errors="replace") as f:
                    num_lines = sum(1 for _ in f) - 1
                total_records += num_lines
        else:
            results["failed"].append(dataset_id)

        # Pause légère entre les téléchargements
        if i < len(datasets):
            time.sleep(0.5)

    _save_metadata(metadata)

    results["stats"] = {
        "total_downloaded": len(results["success"]),
        "total_skipped": len(results["skipped"]),
        "total_failed": len(results["failed"]),
        "total_records": total_records,
        "output_dir": str(OUTPUT_DIR),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"📊 ODISSE - RÉSUMÉ")
    logger.info(f"   ✅ Téléchargés : {len(results['success'])}")
    logger.info(f"   ⏭️  Ignorés     : {len(results['skipped'])}")
    logger.info(f"   ❌ Échecs      : {len(results['failed'])}")
    logger.info(f"   📄 Records     : ~{total_records:,}")
    logger.info(f"{'='*60}")

    return results


def list_datasets() -> None:
    """Affiche la liste des datasets disponibles."""
    print(f"\n{'='*70}")
    print(f"ODISSE — {len(PRIORITY_DATASETS)} datasets prioritaires")
    print(f"{'='*70}")
    for ds in PRIORITY_DATASETS:
        print(f"\n[{ds['priority']:2}] {ds['name'][:60]}")
        print(f"     ID     : {ds['id']}")
        print(f"     Thème  : {ds['theme']}")
        print(f"     Records: ~{ds['expected_records']:,}")


def main() -> int:
    """Point d'entrée CLI."""
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    parser = argparse.ArgumentParser(
        description="Télécharge les données ODISSE (Santé publique France)"
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Forcer le re-téléchargement")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Lister les datasets disponibles")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Télécharger un dataset spécifique (par ID)")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return 0

    if args.dataset:
        # Chercher dans la liste ou créer un dataset ad hoc
        target = next(
            (d for d in PRIORITY_DATASETS if d["id"] == args.dataset),
            {"id": args.dataset, "name": args.dataset, "theme": "unknown",
             "expected_records": 0, "priority": 99},
        )
        result = download_dataset(target, OUTPUT_DIR, force=args.force)
        return 0 if result["status"] in ("success", "skipped") else 1

    results = download_all(force=args.force)
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
