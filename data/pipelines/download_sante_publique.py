"""
OpenDataCopilot - Pipeline de téléchargement des données de Santé Publique
===========================================================================

Ce script télécharge les données de santé publique depuis data.gouv.fr :
- Données hospitalières COVID-19 (quotidien)
- Données épidémiologiques (hebdomadaire)
- Démographie des professionnels de santé (annuel)

Usage:
    python -m data.pipelines.download_sante_publique
    python -m data.pipelines.download_sante_publique --dataset covid
    python -m data.pipelines.download_sante_publique --all --force
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
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "sante" / "metadata.json"

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# ═══════════════════════════════════════════════════════════════
# Catalogue des datasets disponibles sur data.gouv.fr
# ═══════════════════════════════════════════════════════════════

DATASETS = {
    "covid_hospitalisations": {
        "name": "Données hospitalières COVID-19",
        "description": "Nombre quotidien de personnes hospitalisées, en réanimation, décédées par département",
        "url": "https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "columns": [
            "dep",
            "sexe",
            "jour",
            "hosp",
            "rea",
            "HospConv",
            "SSR_USLD",
            "autres",
            "rad",
            "dc",
        ],
    },
    "covid_hospitalisations_age": {
        "name": "Données hospitalières COVID-19 par classe d'âge",
        "description": "Hospitalisations COVID par tranche d'âge et département",
        "url": "https://www.data.gouv.fr/fr/datasets/r/08c18e08-6780-452d-9b8c-ae244ad529b3",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "columns": ["reg", "cl_age90", "jour", "hosp", "rea", "HospConv", "SSR_USLD", "autres", "rad", "dc"],
    },
    "covid_nouveaux_cas": {
        "name": "Données des tests COVID-19",
        "description": "Nombre quotidien de tests et taux de positivité par département",
        "url": "https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "columns": ["dep", "jour", "P", "T", "cl_age90", "pop"],
    },
    "urgences_sos_medecins": {
        "name": "Données des urgences et SOS Médecins",
        "description": "Passages aux urgences et actes SOS Médecins pour suspicion COVID",
        "url": "https://www.data.gouv.fr/fr/datasets/r/eceb9fb4-3ebc-4da3-828d-f5939712571a",
        "format": "csv",
        "frequency": "daily",
        "source": "Santé Publique France",
        "columns": [
            "dep",
            "date_de_passage",
            "sursaud_cl_age_corona",
            "nbre_pass_corona",
            "nbre_pass_tot",
            "nbre_hospit_corona",
            "nbre_pass_corona_h",
            "nbre_pass_corona_f",
            "nbre_pass_tot_h",
            "nbre_pass_tot_f",
            "nbre_acte_corona",
            "nbre_acte_tot",
            "nbre_acte_corona_h",
            "nbre_acte_corona_f",
            "nbre_acte_tot_h",
            "nbre_acte_tot_f",
        ],
    },
    "indicateurs_suivi": {
        "name": "Indicateurs de suivi épidémique",
        "description": "Taux d'incidence, R effectif, taux de positivité par département",
        "url": "https://www.data.gouv.fr/fr/datasets/r/4acad602-d8b1-4516-bc71-7d5574f60cdc",
        "format": "csv",
        "frequency": "weekly",
        "source": "Santé Publique France",
        "columns": [
            "extract_date",
            "dep",
            "region",
            "libelle_dep",
            "libelle_reg",
            "tx_incid",
            "R",
            "taux_occupation_sae",
            "tx_pos",
            "tx_incid_couleur",
            "R_couleur",
            "taux_occupation_sae_couleur",
            "tx_pos_couleur",
            "nb_orange",
            "nb_rouge",
        ],
    },
}


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
    return {"downloads": {}}


def save_metadata(metadata: dict[str, Any]) -> None:
    """Sauvegarde les métadonnées."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Télécharge un fichier avec barre de progression.

    Args:
        url: URL du fichier à télécharger
        destination: Chemin de destination
        chunk_size: Taille des chunks pour le téléchargement

    Returns:
        True si le téléchargement a réussi
    """
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            destination.parent.mkdir(parents=True, exist_ok=True)

            with (
                open(destination, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=destination.name,
                ) as pbar,
            ):
                for chunk in response.iter_bytes(chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)

        return True

    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP {e.response.status_code}: {url}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Erreur de connexion: {e}")
        return False


def download_dataset(
    dataset_id: str,
    force: bool = False,
) -> bool:
    """
    Télécharge un dataset spécifique.

    Args:
        dataset_id: Identifiant du dataset (clé dans DATASETS)
        force: Force le re-téléchargement même si le fichier existe

    Returns:
        True si le téléchargement a réussi ou si le fichier est déjà à jour
    """
    if dataset_id not in DATASETS:
        logger.error(f"Dataset inconnu: {dataset_id}")
        logger.info(f"Datasets disponibles: {', '.join(DATASETS.keys())}")
        return False

    dataset = DATASETS[dataset_id]
    metadata = load_metadata()

    # Nom du fichier de destination
    filename = f"{dataset_id}.{dataset['format']}"
    destination = DATA_RAW_DIR / filename

    # Vérifier si on doit télécharger
    if destination.exists() and not force:
        last_download = metadata.get("downloads", {}).get(dataset_id, {})
        if last_download:
            logger.info(
                f"'{dataset['name']}' déjà téléchargé le {last_download.get('date', 'N/A')}"
            )
            logger.info("Utilisez --force pour re-télécharger")
            return True

    logger.info(f"Téléchargement: {dataset['name']}")
    logger.info(f"Source: {dataset['source']}")
    logger.info(f"URL: {dataset['url']}")

    success = download_file(dataset["url"], destination)

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
        }
        save_metadata(metadata)

        logger.success(
            f"Téléchargé: {filename} ({file_size / 1024 / 1024:.2f} MB)"
        )
    else:
        logger.error(f"Échec du téléchargement: {dataset['name']}")

    return success


def download_all(force: bool = False) -> dict[str, bool]:
    """
    Télécharge tous les datasets disponibles.

    Args:
        force: Force le re-téléchargement

    Returns:
        Dictionnaire {dataset_id: success}
    """
    results = {}

    logger.info("=" * 60)
    logger.info("Téléchargement de tous les datasets Santé Publique")
    logger.info("=" * 60)

    for dataset_id in DATASETS:
        logger.info("-" * 40)
        results[dataset_id] = download_dataset(dataset_id, force=force)

    # Résumé
    logger.info("=" * 60)
    logger.info("RÉSUMÉ")
    logger.info("=" * 60)

    success_count = sum(results.values())
    total_count = len(results)

    for dataset_id, success in results.items():
        status = "OK" if success else "ÉCHEC"
        logger.info(f"  {dataset_id}: {status}")

    logger.info(f"\nTotal: {success_count}/{total_count} datasets téléchargés")

    return results


def list_datasets() -> None:
    """Affiche la liste des datasets disponibles."""
    print("\n" + "=" * 70)
    print("DATASETS SANTÉ PUBLIQUE DISPONIBLES")
    print("=" * 70)

    for dataset_id, info in DATASETS.items():
        print(f"\n{dataset_id}")
        print("-" * len(dataset_id))
        print(f"  Nom: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Fréquence: {info['frequency']}")
        print(f"  Format: {info['format']}")
        print(f"  Source: {info['source']}")

    print("\n" + "=" * 70)


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Télécharge les données de Santé Publique France depuis data.gouv.fr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python -m data.pipelines.download_sante_publique --list
  python -m data.pipelines.download_sante_publique --dataset covid_hospitalisations
  python -m data.pipelines.download_sante_publique --all
  python -m data.pipelines.download_sante_publique --all --force
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="ID du dataset à télécharger",
        choices=list(DATASETS.keys()),
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Télécharge tous les datasets",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force le re-téléchargement même si le fichier existe",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Liste les datasets disponibles",
    )

    args = parser.parse_args()

    # Créer le répertoire de destination
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets()
    elif args.all:
        download_all(force=args.force)
    elif args.dataset:
        download_dataset(args.dataset, force=args.force)
    else:
        # Par défaut, télécharger les hospitalisations COVID
        logger.info("Aucun argument fourni, téléchargement du dataset par défaut...")
        download_dataset("covid_hospitalisations", force=args.force)


if __name__ == "__main__":
    main()
