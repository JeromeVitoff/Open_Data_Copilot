"""
OpenDataCopilot - Orchestrateur de téléchargement des données
=============================================================

Ce script orchestre le téléchargement de toutes les données :
- Données de santé publique (data.gouv.fr)
- Données de pollution (Airparif, OpenAQ)

Usage:
    python -m data.pipelines.run_all
    python -m data.pipelines.run_all --force
    python -m data.pipelines.run_all --sante-only
    python -m data.pipelines.run_all --pollution-only
    python -m data.pipelines.run_all --test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_FILE = DATA_DIR / "download_report.json"

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add(
    DATA_DIR / "logs" / "download_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


def print_banner() -> None:
    """Affiche la bannière du script."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██████╗ ██████╗ ███████╗███╗   ██╗██████╗  █████╗ ████████╗ █████╗  ║
║  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗ ║
║  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║  ██║███████║   ██║   ███████║ ║
║  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║  ██║██╔══██║   ██║   ██╔══██║ ║
║  ╚██████╔╝██║     ███████╗██║ ╚████║██████╔╝██║  ██║   ██║   ██║  ██║ ║
║   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ║
║                         COPILOT                                       ║
║                                                                       ║
║           Pipeline de téléchargement des données                      ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_odisse_pipeline(force: bool = False) -> dict[str, Any]:
    """
    Exécute le pipeline de téléchargement ODISSE (Santé publique France).

    Args:
        force: Force le re-téléchargement

    Returns:
        Résultats du pipeline
    """
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  PIPELINE ODISSE (Santé publique France)                 ║")
    logger.info("╚" + "═" * 58 + "╝")

    try:
        from data.pipelines.fetch_odisse import download_all

        results = download_all(force=force)
        return {
            "status": "success",
            "pipeline": "odisse",
            "downloaded": len(results.get("success", [])),
            "skipped": len(results.get("skipped", [])),
            "failed": len(results.get("failed", [])),
            "total_records": results.get("stats", {}).get("total_records", 0),
            "details": results,
        }
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        return {"status": "error", "pipeline": "odisse", "error": str(e)}
    except Exception as e:
        logger.error(f"Erreur pipeline ODISSE: {e}")
        return {"status": "error", "pipeline": "odisse", "error": str(e)}


def run_airparif_history_pipeline(
    force: bool = False,
    years: list[int] | None = None,
    aggregate_only: bool = True,
) -> dict[str, Any]:
    """
    Exécute le pipeline de téléchargement Airparif historiques 2020-2023.

    Args:
        force: Force le re-téléchargement
        years: Années cibles (défaut: 2020-2023)
        aggregate_only: Télécharger uniquement les fichiers agrégés

    Returns:
        Résultats du pipeline
    """
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  PIPELINE AIRPARIF HISTORIQUES 2020-2023                 ║")
    logger.info("╚" + "═" * 58 + "╝")

    try:
        from data.pipelines.fetch_airparif_history import download_all

        results = download_all(
            years=years or [2020, 2021, 2022, 2023],
            force=force,
            aggregate_only=aggregate_only,
        )
        return {
            "status": "success",
            "pipeline": "airparif_history",
            "downloaded": len(results.get("success", [])),
            "skipped": len(results.get("skipped", [])),
            "failed": len(results.get("failed", [])),
            "total_records": results.get("stats", {}).get("total_records", 0),
            "details": results,
        }
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        return {"status": "error", "pipeline": "airparif_history", "error": str(e)}
    except Exception as e:
        logger.error(f"Erreur pipeline Airparif historiques: {e}")
        return {"status": "error", "pipeline": "airparif_history", "error": str(e)}


def run_sante_pipeline(force: bool = False, priority_only: bool = True) -> dict[str, Any]:
    """
    Exécute le pipeline de téléchargement des données de santé.

    Args:
        force: Force le re-téléchargement
        priority_only: Télécharge uniquement les datasets prioritaires

    Returns:
        Résultats du pipeline
    """
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  PIPELINE SANTÉ PUBLIQUE                                 ║")
    logger.info("╚" + "═" * 58 + "╝")

    try:
        from data.pipelines.download_sante_publique import download_all

        results = download_all(force=force, priority_only=priority_only)
        return {
            "status": "success",
            "pipeline": "sante",
            "downloaded": len(results.get("success", [])),
            "skipped": len(results.get("skipped", [])),
            "failed": len(results.get("failed", [])),
            "details": results,
        }
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        return {"status": "error", "pipeline": "sante", "error": str(e)}
    except Exception as e:
        logger.error(f"Erreur pipeline santé: {e}")
        return {"status": "error", "pipeline": "sante", "error": str(e)}


def run_pollution_pipeline(
    force: bool = False,
    days: int = 30,
    source: str = "all",
) -> dict[str, Any]:
    """
    Exécute le pipeline de récupération des données de pollution.

    Args:
        force: Force le re-téléchargement
        days: Nombre de jours de données
        source: Source ('airparif', 'openaq', 'all')

    Returns:
        Résultats du pipeline
    """
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  PIPELINE POLLUTION (GÉOD'AIR + Airparif)                ║")
    logger.info("╚" + "═" * 58 + "╝")

    try:
        from data.pipelines.fetch_pollution import fetch_pollution_data

        results = fetch_pollution_data(source=source, days=days, force=force)
        return {
            "status": "success",
            "pipeline": "pollution",
            "files_created": len(results.get("success", [])),
            "total_records": results.get("total_records", 0),
            "failed": len(results.get("failed", [])),
            "details": results,
        }
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        return {"status": "error", "pipeline": "pollution", "error": str(e)}
    except Exception as e:
        logger.error(f"Erreur pipeline pollution: {e}")
        return {"status": "error", "pipeline": "pollution", "error": str(e)}


def test_connections() -> dict[str, Any]:
    """Teste les connexions aux APIs."""
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  TEST DES CONNEXIONS API                                 ║")
    logger.info("╚" + "═" * 58 + "╝")

    results = {}

    try:
        from data.pipelines.fetch_pollution import test_api_connection

        for source in ["geodair", "airparif", "openaq"]:
            success, message = test_api_connection(source)
            results[source] = {"success": success, "message": message}
            if success:
                logger.success(f"  {source.upper()}: {message}")
            else:
                logger.warning(f"  {source.upper()}: {message}")

    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        results["error"] = str(e)

    # Test data.gouv.fr
    try:
        import httpx

        response = httpx.get(
            "https://www.data.gouv.fr/api/1/datasets/",
            params={"page_size": 1},
            timeout=10.0,
        )
        if response.status_code == 200:
            logger.success("  DATA.GOUV.FR: Connexion OK")
            results["data_gouv"] = {"success": True, "message": "Connexion OK"}
        else:
            logger.warning(f"  DATA.GOUV.FR: Erreur HTTP {response.status_code}")
            results["data_gouv"] = {"success": False, "message": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.warning(f"  DATA.GOUV.FR: {e}")
        results["data_gouv"] = {"success": False, "message": str(e)}

    return results


def generate_report(results: dict[str, Any]) -> None:
    """Génère un rapport de téléchargement."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "total_pipelines": len(results),
            "successful": sum(1 for r in results.values() if r.get("status") == "success"),
            "failed": sum(1 for r in results.values() if r.get("status") == "error"),
        },
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nRapport sauvegardé: {REPORT_FILE}")


def print_summary(results: dict[str, Any]) -> None:
    """Affiche le résumé final."""
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  RÉSUMÉ FINAL                                            ║")
    logger.info("╚" + "═" * 58 + "╝")

    total_files = 0
    total_records = 0
    errors = []

    for pipeline, result in results.items():
        if result.get("status") == "success":
            files = result.get("downloaded", 0) + result.get("files_created", 0)
            records = result.get("total_records", 0)
            total_files += files
            total_records += records
            logger.success(f"  {pipeline.upper()}: {files} fichiers")
        elif result.get("status") == "error":
            errors.append(f"{pipeline}: {result.get('error', 'Unknown')}")
            logger.error(f"  {pipeline.upper()}: ÉCHEC")

    logger.info("")
    logger.info(f"  Total fichiers téléchargés: {total_files}")
    logger.info(f"  Total enregistrements: {total_records}")

    if errors:
        logger.warning("")
        logger.warning("  Erreurs rencontrées:")
        for error in errors:
            logger.warning(f"    - {error}")

    logger.info("")
    logger.info("  Données stockées dans:")
    logger.info(f"    - Santé:             {DATA_DIR / 'raw' / 'sante'}")
    logger.info(f"    - Santé ODISSE:      {DATA_DIR / 'raw' / 'sante_odisse'}")
    logger.info(f"    - Pollution:         {DATA_DIR / 'raw' / 'pollution'}")
    logger.info(f"    - Airparif hist:     {DATA_DIR / 'raw' / 'pollution_airparif_hist'}")


def main() -> int:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Télécharge toutes les données pour OpenDataCopilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m data.pipelines.run_all                  # Tout télécharger
  python -m data.pipelines.run_all --test           # Tester les connexions
  python -m data.pipelines.run_all --sante-only     # Santé uniquement
  python -m data.pipelines.run_all --pollution-only # Pollution uniquement
  python -m data.pipelines.run_all --force          # Forcer re-téléchargement
  python -m data.pipelines.run_all --days 7         # 7 jours de pollution
        """,
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force le re-téléchargement de tous les fichiers",
    )
    parser.add_argument(
        "--odisse",
        action="store_true",
        help="Télécharge les données ODISSE (Santé publique France)",
    )
    parser.add_argument(
        "--odisse-only",
        action="store_true",
        help="Télécharge uniquement les données ODISSE",
    )
    parser.add_argument(
        "--airparif-history",
        action="store_true",
        help="Télécharge les données Airparif historiques 2020-2023",
    )
    parser.add_argument(
        "--airparif-history-only",
        action="store_true",
        help="Télécharge uniquement les données Airparif historiques",
    )
    parser.add_argument(
        "--all-new",
        action="store_true",
        help="Télécharge TOUTES les nouvelles sources (ODISSE + Airparif hist)",
    )
    parser.add_argument(
        "--sante-only",
        action="store_true",
        help="Télécharge uniquement les données de santé",
    )
    parser.add_argument(
        "--pollution-only",
        action="store_true",
        help="Télécharge uniquement les données de pollution",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Télécharge tous les datasets (pas seulement les prioritaires)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Nombre de jours de données pollution (défaut: 30)",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Teste uniquement les connexions aux APIs",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Mode silencieux (moins de logs)",
    )

    args = parser.parse_args()

    # Créer le dossier de logs
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)

    # Afficher la bannière
    if not args.quiet:
        print_banner()

    start_time = datetime.now()
    results = {}

    # Test de connexion uniquement
    if args.test:
        test_connections()
        return 0

    # Exécuter les pipelines
    try:
        run_all_new = args.all_new

        # Pipelines sources existantes
        if not args.pollution_only and not args.odisse_only and not args.airparif_history_only:
            results["sante"] = run_sante_pipeline(
                force=args.force,
                priority_only=not args.all_datasets,
            )

        if not args.sante_only and not args.odisse_only and not args.airparif_history_only:
            results["pollution"] = run_pollution_pipeline(
                force=args.force,
                days=args.days,
            )

        # Nouvelles sources
        if args.odisse or args.odisse_only or run_all_new:
            results["odisse"] = run_odisse_pipeline(force=args.force)

        if args.airparif_history or args.airparif_history_only or run_all_new:
            results["airparif_history"] = run_airparif_history_pipeline(
                force=args.force,
                aggregate_only=True,
            )

    except KeyboardInterrupt:
        logger.warning("\nInterruption par l'utilisateur")
        return 1

    # Générer le rapport
    generate_report(results)

    # Afficher le résumé
    print_summary(results)

    # Durée totale
    duration = datetime.now() - start_time
    logger.info(f"\n  Durée totale: {duration.total_seconds():.1f} secondes")

    # Code de retour
    has_errors = any(r.get("status") == "error" for r in results.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
