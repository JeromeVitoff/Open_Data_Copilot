"""
Configuration pour la baseline sans RAG.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BaselineConfig:
    """Configuration de la baseline."""

    # Modèle OpenAI
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 500

    # Coûts OpenAI (prix par 1000 tokens)
    input_cost_per_1k: float = 0.0015  # $0.0015 / 1K input tokens
    output_cost_per_1k: float = 0.002  # $0.002 / 1K output tokens

    # Prompt système
    system_prompt: str = """Tu es un assistant qui répond aux questions sur la santé publique et la pollution en France.

IMPORTANT :
- Si tu ne connais pas une information précise, dis-le clairement
- Ne fournis JAMAIS de chiffres ou statistiques sans être absolument certain
- Pour les données récentes (2024-2025), admets les limites de tes connaissances
- Préfère dire "je ne sais pas" plutôt que d'inventer des données

Domaines de compétence :
- COVID-19 : hospitalisations, tests, vaccination
- Urgences hospitalières et capacités
- Démographie médicale (médecins, spécialistes)
- Qualité de l'air en Île-de-France (Airparif)
- Pollution atmosphérique (NO2, PM2.5, PM10, O3)

Rappel : Tes connaissances s'arrêtent à ta date de formation. Pour des données en temps réel, indique que tu ne peux pas y accéder."""

    # Chemins
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent / "results")

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
