"""
OpenDataCopilot - Pipeline Airparif Historiques 2020-2023
=========================================================

Télécharge les données historiques de qualité de l'air depuis le portail
Open Data Airparif (https://data-airparif-asso.opendata.arcgis.com).

L'API ArcGIS Hub est utilisée pour découvrir dynamiquement les datasets
disponibles pour les années 2020-2023. Les fichiers CSV sont ensuite
téléchargés depuis arcgis.com.

Stratégie :
- Découverte dynamique via l'API de recherche ArcGIS Hub
- Priorité aux fichiers agrégés par polluant (NO2, PM10, PM25, O3, CO, NOX, NO)
- Fallback sur les fichiers par station individuels si insuffisant
- Pause entre les requêtes pour respecter les rate limits

Usage:
    python -m data.pipelines.fetch_airparif_history
    python -m data.pipelines.fetch_airparif_history --years 2020 2021 2022 2023
    python -m data.pipelines.fetch_airparif_history --force
    python -m data.pipelines.fetch_airparif_history --list
"""

import json
import re
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pollution_airparif_hist"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

AIRPARIF_SEARCH_URL = (
    "https://data-airparif-asso.opendata.arcgis.com"
    "/api/search/v1/collections/all/items"
)
ARCGIS_DOWNLOAD_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"
)

# Polluants à cibler (en priorité les fichiers agrégés)
TARGET_POLLUTANTS = ["NO2", "PM10", "PM25", "O3", "CO", "NOX", "NO"]

# Années cibles
DEFAULT_YEARS = [2020, 2021, 2022, 2023]

# Rate limiting (Airparif Hub : 10 req/s)
REQUEST_DELAY = 0.3  # secondes entre requêtes API
DOWNLOAD_DELAY = 1.0  # secondes entre téléchargements


# ─────────────────────────────────────────────────────────────────────────────
# Découverte des datasets
# ─────────────────────────────────────────────────────────────────────────────

def _extract_year_from_title(title: str) -> int | None:
    """Extrait l'année depuis le titre d'un dataset Airparif.

    Gère les patterns :
    - "2023 NO2 boulevard périphérique"  → word boundary classique
    - "2023_OPERA"                       → underscore après l'année
    - "Stats 2023"                       → année en fin
    """
    # Lookahead/lookbehind sur les CHIFFRES uniquement (pas _)
    match = re.search(r'(?<!\d)(20[12]\d)(?!\d)', title)
    if match:
        return int(match.group(1))
    return None


def _is_aggregate_file(title: str) -> bool:
    """Vérifie si le fichier est un agrégat polluant (pas une station individuelle)."""
    # Les agrégats suivent le pattern "YYYY_NO2", "YYYY PM10 boulevard périphérique", etc.
    # Les stations individuelles : "YYYY_OPERA", "YYYY_STDEN", etc.
    for pollutant in TARGET_POLLUTANTS:
        if pollutant.upper() in title.upper():
            return True
    return False


def _is_station_file(title: str) -> bool:
    """Vérifie si c'est un fichier de station individuelle."""
    # Pattern : "YYYY_STATIONCODE" où STATIONCODE n'est pas un polluant
    if re.match(r'^\d{4}_[A-Z0-9-]+$', title.strip()):
        return not _is_aggregate_file(title)
    return False


def discover_datasets(
    years: list[int] | None = None,
    max_pages: int = 20,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """
    Découvre les datasets Airparif pour les années cibles via l'API ArcGIS Hub.

    Utilise la pagination par lien 'next' (API OGC standard, 1-indexée).

    Args:
        years: Années à cibler (défaut: 2020-2023)
        max_pages: Nombre maximum de pages à parcourir
        session: Session requests

    Returns:
        Liste de datasets: [{"id": ..., "title": ..., "year": ..., "type": ...}]
    """
    if years is None:
        years = DEFAULT_YEARS
    if session is None:
        session = requests.Session()

    years_set = set(years)
    discovered: list[dict] = []
    page_size = 50  # Taille de page conservative
    total_fetched = 0

    logger.info(f"   🔍 Découverte des datasets Airparif {years}...")

    # Première page (pas de startindex)
    next_url: str | None = AIRPARIF_SEARCH_URL
    next_params: dict | None = {"limit": page_size}

    for page in range(max_pages):
        try:
            if next_params is not None:
                resp = session.get(next_url, params=next_params, timeout=20)
            else:
                # Suivre le lien 'next' tel quel (contient déjà les params)
                resp = session.get(next_url, timeout=20)

            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            total_matched = data.get("numberMatched", 0)

            if not features:
                break

            for feature in features:
                item_id = feature.get("id", "")
                props = feature.get("properties", {})
                title = props.get("title", "")
                ftype = props.get("type", "")

                if ftype.upper() != "CSV":
                    continue

                year = _extract_year_from_title(title)

                if year not in years_set:
                    continue

                is_aggregate = _is_aggregate_file(title)
                is_station = _is_station_file(title)

                discovered.append({
                    "id": item_id,
                    "title": title,
                    "year": year,
                    "size_bytes": props.get("size", 0),
                    "is_aggregate": is_aggregate,
                    "is_station": is_station,
                    "description": (props.get("description") or "")[:100],
                })

            total_fetched += len(features)

            logger.debug(
                f"   Page {page + 1}: {len(features)} items | "
                f"trouvés: {len(discovered)} | "
                f"total: {total_fetched}/{total_matched}"
            )

            # Trouver le lien 'next' dans la réponse
            next_link = next(
                (lnk["href"] for lnk in data.get("links", [])
                 if lnk.get("rel") == "next"),
                None,
            )

            if not next_link or total_fetched >= total_matched:
                break

            # Passer au lien next (paramètres déjà inclus dans l'URL)
            next_url = next_link
            next_params = None  # Ne pas re-passer les params, ils sont dans l'URL

            time.sleep(REQUEST_DELAY)

        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Erreur API page {page + 1}: {e}")
            break

    # Trier : agrégats d'abord, puis stations, puis par année
    discovered.sort(key=lambda x: (not x["is_aggregate"], x["year"], x["title"]))

    logger.info(
        f"   ✅ {len(discovered)} datasets trouvés "
        f"({sum(1 for d in discovered if d['is_aggregate'])} agrégats, "
        f"{sum(1 for d in discovered if d['is_station'])} stations)"
    )

    return discovered


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
    Télécharge un fichier CSV Airparif.

    Args:
        dataset: Métadonnées du dataset (id, title, year, ...)
        output_dir: Répertoire de sortie
        force: Force le re-téléchargement
        session: Session requests

    Returns:
        Résultat: {"status": ..., "records": N, "file": path}
    """
    item_id = dataset["id"]
    title = dataset["title"]

    # Nom de fichier propre
    safe_name = re.sub(r'[^\w\-_]', '_', title).strip('_')
    output_file = output_dir / f"airparif_{safe_name}.csv"

    if output_file.exists() and not force:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"       ⏭️  {title[:50]} → déjà présent ({size_mb:.1f} MB)")
        return {"status": "skipped", "file": str(output_file), "size_mb": size_mb}

    if session is None:
        session = requests.Session()

    download_url = ARCGIS_DOWNLOAD_URL.format(item_id=item_id)

    try:
        resp = session.get(
            download_url,
            stream=True,
            timeout=(30, 120),
        )
        resp.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)

        total_bytes = 0
        with open(output_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

        size_mb = total_bytes / (1024 * 1024)

        # Compter les lignes
        try:
            with open(output_file, "r", encoding="utf-8-sig", errors="replace") as f:
                num_lines = sum(1 for _ in f) - 1
        except Exception:
            num_lines = -1

        logger.info(
            f"       ✅ {title[:45]:<45} | "
            f"{size_mb:.1f} MB | {num_lines:,} rows"
        )

        return {
            "status": "success",
            "file": str(output_file),
            "size_mb": size_mb,
            "records": num_lines,
        }

    except requests.exceptions.HTTPError as e:
        logger.warning(f"       ❌ HTTP {e.response.status_code}: {title[:40]}")
        return {"status": "error", "error": f"HTTP {e.response.status_code}"}

    except requests.exceptions.Timeout:
        logger.warning(f"       ❌ Timeout: {title[:40]}")
        if output_file.exists():
            output_file.unlink()
        return {"status": "error", "error": "Timeout"}

    except Exception as e:
        logger.error(f"       ❌ Erreur: {e}")
        if output_file.exists():
            output_file.unlink()
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def download_all(
    years: list[int] | None = None,
    force: bool = False,
    aggregate_only: bool = False,
    max_files: int | None = None,
) -> dict[str, Any]:
    """
    Découvre et télécharge les datasets Airparif pour les années cibles.

    Args:
        years: Années cibles (défaut: 2020-2023)
        force: Force le re-téléchargement
        aggregate_only: Télécharger uniquement les fichiers agrégés
        max_files: Limite le nombre de fichiers téléchargés

    Returns:
        Résultats: {"success": [...], "skipped": [...], "failed": [...], "stats": {...}}
    """
    if years is None:
        years = DEFAULT_YEARS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "OpenDataCopilot/1.0 (academic research)",
    })

    logger.info(f"\n🌬️  Airparif Historiques : années {years}")
    logger.info(f"   Sortie : {OUTPUT_DIR}")

    # Découverte
    datasets = discover_datasets(years=years, session=session)

    if aggregate_only:
        datasets = [d for d in datasets if d["is_aggregate"]]
        logger.info(f"   → Filtre agrégats uniquement : {len(datasets)} datasets")

    if max_files is not None:
        datasets = datasets[:max_files]
        logger.info(f"   → Limité à {max_files} fichiers")

    if not datasets:
        logger.warning("   ⚠️  Aucun dataset trouvé pour les années sélectionnées")
        return {"success": [], "skipped": [], "failed": [], "stats": {"total_records": 0}}

    logger.info(f"   📦 {len(datasets)} datasets à télécharger\n")

    results: dict[str, list] = {"success": [], "skipped": [], "failed": []}
    total_records = 0
    metadata_entries = {}

    for i, dataset in enumerate(datasets, 1):
        file_type = "agrégat" if dataset["is_aggregate"] else "station"
        logger.info(
            f"[{i:3}/{len(datasets)}] {dataset['year']} | {file_type} | "
            f"{dataset['title'][:45]}"
        )

        result = download_dataset(dataset, OUTPUT_DIR, force=force, session=session)

        if result["status"] == "success":
            results["success"].append(dataset["title"])
            total_records += result.get("records", 0)
            metadata_entries[dataset["id"]] = {
                "title": dataset["title"],
                "year": dataset["year"],
                "is_aggregate": dataset["is_aggregate"],
                "file": result["file"],
                "size_mb": result.get("size_mb", 0),
                "records": result.get("records", 0),
                "downloaded_at": datetime.now().isoformat(),
            }
        elif result["status"] == "skipped":
            results["skipped"].append(dataset["title"])
            output_file = OUTPUT_DIR / f"airparif_{re.sub(r'[^\w\-_]', '_', dataset['title']).strip('_')}.csv"
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8-sig", errors="replace") as f:
                    num_lines = sum(1 for _ in f) - 1
                total_records += max(0, num_lines)
        else:
            results["failed"].append(dataset["title"])

        time.sleep(DOWNLOAD_DELAY)

    # Sauvegarder les métadonnées
    metadata = _load_metadata()
    metadata.setdefault("downloads", {}).update(metadata_entries)
    metadata["last_updated"] = datetime.now().isoformat()
    metadata["years_covered"] = list(years)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    results["stats"] = {
        "total_downloaded": len(results["success"]),
        "total_skipped": len(results["skipped"]),
        "total_failed": len(results["failed"]),
        "total_records": total_records,
        "output_dir": str(OUTPUT_DIR),
        "years": years,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"🌬️  Airparif Historiques - RÉSUMÉ")
    logger.info(f"   ✅ Téléchargés : {len(results['success'])}")
    logger.info(f"   ⏭️  Ignorés     : {len(results['skipped'])}")
    logger.info(f"   ❌ Échecs      : {len(results['failed'])}")
    logger.info(f"   📄 Records     : ~{total_records:,}")
    logger.info(f"{'='*60}")

    return results


def _load_metadata() -> dict[str, Any]:
    """Charge les métadonnées existantes."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloads": {}, "last_updated": None}


def main() -> int:
    """Point d'entrée CLI."""
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    parser = argparse.ArgumentParser(
        description="Télécharge les données historiques Airparif 2020-2023"
    )
    parser.add_argument(
        "--years", "-y", type=int, nargs="+", default=DEFAULT_YEARS,
        help=f"Années à télécharger (défaut: {DEFAULT_YEARS})"
    )
    parser.add_argument("--force", "-f", action="store_true",
                        help="Forcer le re-téléchargement")
    parser.add_argument("--aggregate-only", "-a", action="store_true",
                        help="Télécharger uniquement les fichiers agrégés par polluant")
    parser.add_argument("--max-files", "-m", type=int, default=None,
                        help="Limite le nombre de fichiers")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Lister les datasets disponibles sans télécharger")
    args = parser.parse_args()

    if args.list:
        logger.info(f"Découverte des datasets Airparif pour {args.years}...")
        datasets = discover_datasets(years=args.years)
        for ds in datasets:
            flag = "AGG" if ds["is_aggregate"] else "STA"
            size_kb = ds.get("size_bytes", 0) / 1024
            print(f"  [{flag}] {ds['year']} | {ds['title']:<50} | {size_kb:.0f} KB")
        return 0

    results = download_all(
        years=args.years,
        force=args.force,
        aggregate_only=args.aggregate_only,
        max_files=args.max_files,
    )
    return 0 if not results.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
