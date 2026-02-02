"""
OpenDataCopilot - Pipeline de téléchargement des données de Santé Publique
===========================================================================

Ce script télécharge les données de santé publique depuis data.gouv.fr :
- Données hospitalières COVID-19 (quotidien)
- Données épidémiologiques SurSaUD (quotidien)
- Démographie des professionnels de santé RPPS (annuel)
- Indicateurs de suivi épidémique

Usage:
    python -m data.pipelines.download_sante_publique
    python -m data.pipelines.download_sante_publique --dataset covid_hospitalisations
    python -m data.pipelines.download_sante_publique --all --force
    python -m data.pipelines.download_sante_publique --list
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from tqdm import tqdm

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sante"
METADATA_FILE = DATA_RAW_DIR / "metadata.json"

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# ═══════════════════════════════════════════════════════════════
# Catalogue des datasets disponibles sur data.gouv.fr
# URLs vérifiées et à jour (Février 2025)
# ═══════════════════════════════════════════════════════════════

DATASETS = {
    # ─────────────────────────────────────────────────────────────
    # PRIORITÉ 1 : Données hospitalières COVID-19
    # ─────────────────────────────────────────────────────────────
    "covid_hospitalisations": {
        "name": "Données hospitalières COVID-19",
        "description": "Nombre quotidien de personnes hospitalisées, en réanimation, décédées par département et sexe",
        "url": "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "priority": 1,
        "columns": ["dep", "sexe", "jour", "hosp", "rea", "HospConv", "SSR_USLD", "autres", "rad", "dc"],
    },
    "covid_hospitalisations_etablissement": {
        "name": "Données hospitalières COVID-19 par établissement",
        "description": "Données COVID par établissement hospitalier",
        "url": "https://www.data.gouv.fr/fr/datasets/r/41b9bd2a-b5b6-4271-8878-e45a8f1cfe47",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "priority": 2,
        "columns": ["dep", "jour", "hosp", "rea", "rad", "dc"],
    },
    "covid_hospitalisations_classe_age": {
        "name": "Données hospitalières COVID-19 par classe d'âge",
        "description": "Hospitalisations COVID par tranche d'âge et région",
        "url": "https://www.data.gouv.fr/fr/datasets/r/08c18e08-6780-452d-9b8c-ae244ad529b3",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "priority": 2,
        "columns": ["reg", "cl_age90", "jour", "hosp", "rea", "HospConv", "SSR_USLD", "autres", "rad", "dc"],
    },

    # ─────────────────────────────────────────────────────────────
    # PRIORITÉ 2 : Surveillance syndromique SurSaUD
    # URLs mises à jour - Données urgences hospitalières
    # ─────────────────────────────────────────────────────────────
    "sursaud_urgences": {
        "name": "Données quotidiennes urgences SOS Médecins",
        "description": "Passages aux urgences pour suspicion COVID-19",
        "url": "https://www.data.gouv.fr/fr/datasets/r/6fadff46-9efd-4c53-942a-54aca783c30c",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France - SurSaUD",
        "priority": 1,
        "columns": ["dep", "date_de_passage", "sursaud_cl_age_corona", "nbre_pass_corona", "nbre_pass_tot"],
    },
    "covid_tests_dep": {
        "name": "Tests de dépistage COVID-19 par département",
        "description": "Nombre de tests réalisés et positifs par département",
        "url": "https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "priority": 1,
        "columns": ["dep", "jour", "P", "T", "cl_age90"],
    },

    # ─────────────────────────────────────────────────────────────
    # PRIORITÉ 3 : Démographie des professionnels de santé
    # URLs data.gouv.fr vérifiées
    # ─────────────────────────────────────────────────────────────
    "professionnels_sante_dep": {
        "name": "Patientèle médecin traitant - Médecins généralistes par territoire",
        "description": "Patientèle moyenne des médecins généralistes par département et région",
        "url": "https://data.ameli.fr/api/explore/v2.1/catalog/datasets/patientele-medecintraitant-generalistes-annuelle/exports/csv?use_labels=true",
        "format": "csv",
        "frequency": "annual",
        "source": "Assurance Maladie (AMELI)",
        "priority": 1,
        "columns": ["annee", "region", "departement", "patientele_moyenne", "nb_medecins"],
    },
    "medecins_generalistes": {
        "name": "Médecins généralistes par commune",
        "description": "Nombre de médecins généralistes libéraux et mixtes",
        "url": "https://www.data.gouv.fr/fr/datasets/r/a1efcd60-6e86-4b1a-84de-a6f30e9e9375",
        "format": "csv",
        "frequency": "annual",
        "source": "Assurance Maladie",
        "priority": 2,
        "columns": ["annee", "code_commune", "libelle", "nb_medecins"],
    },

    # ─────────────────────────────────────────────────────────────
    # Indicateurs de suivi épidémique - URLs mises à jour
    # ─────────────────────────────────────────────────────────────
    "indicateurs_suivi_covid": {
        "name": "Indicateurs de suivi épidémique COVID",
        "description": "Taux d'incidence et positivité par département",
        "url": "https://www.data.gouv.fr/fr/datasets/r/5c4e1452-3850-4b59-b11c-3dd51d7fb8b5",
        "format": "csv",
        "frequency": "weekly",
        "source": "Santé Publique France",
        "priority": 2,
        "columns": ["dep", "semaine", "tx_incid", "tx_pos"],
    },

    # ─────────────────────────────────────────────────────────────
    # Données complémentaires
    # ─────────────────────────────────────────────────────────────
    "capacites_hospitalieres": {
        "name": "Capacités hospitalières par département",
        "description": "Nombre de lits et places par type d'hospitalisation",
        "url": "https://www.data.gouv.fr/fr/datasets/r/1a3a7f7c-8a8e-4e4c-8c8a-3f3b9a9a0d9b",
        "format": "csv",
        "frequency": "annual",
        "source": "DREES - SAE",
        "priority": 3,
        "columns": ["annee", "departement", "type_hospi", "nb_lits", "nb_places"],
    },
    "vaccination_covid": {
        "name": "Données de vaccination COVID-19",
        "description": "Nombre de vaccinations par département et tranche d'âge",
        "url": "https://www.data.gouv.fr/fr/datasets/r/900da9b0-8987-4ba7-b117-7aea0e53f530",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "priority": 3,
        "columns": ["dep", "jour", "clage_vacsi", "n_dose1", "n_complet", "n_rappel"],
    },
}

# Liste des datasets prioritaires (téléchargés par défaut)
PRIORITY_DATASETS = [
    "covid_hospitalisations",
    "covid_tests_dep",
    "sursaud_urgences",
    "professionnels_sante_dep",
]


def compute_file_hash(filepath: Path) -> str:
    """Calcule le hash MD5 d'un fichier."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_metadata() -> dict[str, Any]:
    """Charge les métadonnées des téléchargements précédents."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {"downloads": {}, "last_run": None}


def save_metadata(metadata: dict[str, Any]) -> None:
    """Sauvegarde les métadonnées."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    metadata["last_run"] = datetime.now().isoformat()
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 8192,
    timeout: float = 120.0,
) -> tuple[bool, str]:
    """
    Télécharge un fichier avec barre de progression.

    Args:
        url: URL du fichier à télécharger
        destination: Chemin de destination
        chunk_size: Taille des chunks pour le téléchargement
        timeout: Timeout en secondes

    Returns:
        Tuple (success, error_message)
    """
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "OpenDataCopilot/1.0 (Master2 Data Science Project)"},
        ) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            destination.parent.mkdir(parents=True, exist_ok=True)

            with (
                open(destination, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"  {destination.name}",
                    leave=False,
                ) as pbar,
            ):
                for chunk in response.iter_bytes(chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)

        return True, ""

    except httpx.HTTPStatusError as e:
        return False, f"Erreur HTTP {e.response.status_code}"
    except httpx.TimeoutException:
        return False, "Timeout dépassé"
    except httpx.RequestError as e:
        return False, f"Erreur de connexion: {type(e).__name__}"
    except OSError as e:
        return False, f"Erreur d'écriture: {e}"


def download_dataset(
    dataset_id: str,
    force: bool = False,
    verbose: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Télécharge un dataset spécifique.

    Args:
        dataset_id: Identifiant du dataset (clé dans DATASETS)
        force: Force le re-téléchargement même si le fichier existe
        verbose: Affiche les logs détaillés

    Returns:
        Tuple (success, info_dict)
    """
    if dataset_id not in DATASETS:
        if verbose:
            logger.error(f"Dataset inconnu: {dataset_id}")
        return False, {"error": "Dataset inconnu"}

    dataset = DATASETS[dataset_id]
    metadata = load_metadata()

    # Nom du fichier de destination
    filename = f"{dataset_id}.{dataset['format']}"
    destination = DATA_RAW_DIR / filename

    info = {
        "dataset_id": dataset_id,
        "name": dataset["name"],
        "file": str(destination),
        "source": dataset["source"],
    }

    # Vérifier si on doit télécharger
    if destination.exists() and not force:
        last_download = metadata.get("downloads", {}).get(dataset_id, {})
        if last_download:
            if verbose:
                logger.info(f"'{dataset['name']}' déjà téléchargé le {last_download.get('date', 'N/A')[:10]}")
            info["status"] = "skipped"
            info["reason"] = "already_exists"
            return True, info

    if verbose:
        logger.info(f"Téléchargement: {dataset['name']}")

    success, error = download_file(dataset["url"], destination)

    if success:
        # Mettre à jour les métadonnées
        file_hash = compute_file_hash(destination)
        file_size = destination.stat().st_size

        metadata.setdefault("downloads", {})[dataset_id] = {
            "date": datetime.now().isoformat(),
            "file": str(destination.relative_to(PROJECT_ROOT)),
            "size_bytes": file_size,
            "hash_md5": file_hash,
            "source_url": dataset["url"],
            "source": dataset["source"],
        }
        save_metadata(metadata)

        if verbose:
            logger.success(f"  OK: {filename} ({file_size / 1024 / 1024:.2f} MB)")

        info["status"] = "downloaded"
        info["size_bytes"] = file_size
        info["hash_md5"] = file_hash
    else:
        if verbose:
            logger.error(f"  ÉCHEC: {error}")
        info["status"] = "failed"
        info["error"] = error

    return success, info


def download_all(force: bool = False, priority_only: bool = False) -> dict[str, Any]:
    """
    Télécharge tous les datasets disponibles.

    Args:
        force: Force le re-téléchargement
        priority_only: Télécharge uniquement les datasets prioritaires

    Returns:
        Dictionnaire avec résultats et statistiques
    """
    datasets_to_download = PRIORITY_DATASETS if priority_only else list(DATASETS.keys())

    results = {
        "success": [],
        "failed": [],
        "skipped": [],
        "total": len(datasets_to_download),
    }

    logger.info("=" * 60)
    logger.info("TÉLÉCHARGEMENT DONNÉES SANTÉ PUBLIQUE")
    logger.info("=" * 60)
    logger.info(f"Datasets à traiter: {len(datasets_to_download)}")
    logger.info("")

    for dataset_id in datasets_to_download:
        success, info = download_dataset(dataset_id, force=force, verbose=True)

        if info.get("status") == "downloaded":
            results["success"].append(info)
        elif info.get("status") == "skipped":
            results["skipped"].append(info)
        else:
            results["failed"].append(info)

    # Résumé
    logger.info("")
    logger.info("=" * 60)
    logger.info("RÉSUMÉ SANTÉ PUBLIQUE")
    logger.info("=" * 60)
    logger.info(f"  Téléchargés: {len(results['success'])}")
    logger.info(f"  Ignorés (déjà présents): {len(results['skipped'])}")
    logger.info(f"  Échecs: {len(results['failed'])}")

    if results["failed"]:
        logger.warning("Datasets en échec:")
        for item in results["failed"]:
            logger.warning(f"  - {item['dataset_id']}: {item.get('error', 'Unknown')}")

    return results


def list_datasets() -> None:
    """Affiche la liste des datasets disponibles."""
    print("\n" + "=" * 70)
    print("DATASETS SANTÉ PUBLIQUE DISPONIBLES")
    print("=" * 70)

    # Grouper par priorité
    by_priority: dict[int, list] = {}
    for dataset_id, info in DATASETS.items():
        priority = info.get("priority", 3)
        by_priority.setdefault(priority, []).append((dataset_id, info))

    for priority in sorted(by_priority.keys()):
        print(f"\n{'='*30} PRIORITÉ {priority} {'='*30}")
        for dataset_id, info in by_priority[priority]:
            marker = "★" if dataset_id in PRIORITY_DATASETS else " "
            print(f"\n{marker} {dataset_id}")
            print(f"  Nom: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Fréquence: {info['frequency']}")
            print(f"  Source: {info['source']}")

    print("\n" + "=" * 70)
    print("★ = Dataset prioritaire (téléchargé par défaut)")
    print("=" * 70 + "\n")


def main() -> dict[str, Any] | None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Télécharge les données de Santé Publique depuis data.gouv.fr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m data.pipelines.download_sante_publique --list
  python -m data.pipelines.download_sante_publique --dataset covid_hospitalisations
  python -m data.pipelines.download_sante_publique --priority
  python -m data.pipelines.download_sante_publique --all
  python -m data.pipelines.download_sante_publique --all --force
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="ID du dataset à télécharger",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Télécharge tous les datasets",
    )
    parser.add_argument(
        "--priority", "-p",
        action="store_true",
        help="Télécharge uniquement les datasets prioritaires",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force le re-téléchargement même si le fichier existe",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Liste les datasets disponibles",
    )

    args = parser.parse_args()

    # Créer le répertoire de destination
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets()
        return None
    elif args.all:
        return download_all(force=args.force, priority_only=False)
    elif args.priority:
        return download_all(force=args.force, priority_only=True)
    elif args.dataset:
        success, info = download_dataset(args.dataset, force=args.force)
        return {"success": [info] if success else [], "failed": [] if success else [info]}
    else:
        # Par défaut, télécharger les datasets prioritaires
        logger.info("Téléchargement des datasets prioritaires...")
        return download_all(force=args.force, priority_only=True)


if __name__ == "__main__":
    main()
