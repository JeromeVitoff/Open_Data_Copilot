"""
Embeddings médicaux avec CamemBERT-bio.

CamemBERT-bio (almanach/camembert-bio-base) est un CamemBERT fine-tuné
sur un corpus biomédical français — il comprend la terminologie médicale
française mieux que les embeddings génériques OpenAI.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class MedicalEmbeddings:
    """
    Embeddings spécialisés domaine médical français.

    Utilise CamemBERT-bio (almanach/camembert-bio-base) fine-tuné
    sur corpus biomédical français (PubMed FR, thèses médicales...).

    Args:
        model_name: Modèle HuggingFace (défaut: camembert-bio-base)
        device: 'cuda', 'cpu', ou None (auto-détection)
        normalize: Normaliser les embeddings (recommandé pour cosine sim)
    """

    def __init__(
        self,
        model_name: str = "almanach/camembert-bio-base",
        device: str | None = None,
        normalize: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Chargement embeddings medicaux : {model_name}")
        print(f"Device : {device}")

        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Modele charge — dimension : {self.embedding_dim} | device : {self.model.device}")

    def embed_query(self, text: str) -> np.ndarray:
        """
        Encode une requête unique.

        Args:
            text: Texte de la requête

        Returns:
            np.ndarray de shape (embedding_dim,)
        """
        return self.model.encode(
            text,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode un batch de documents.

        Args:
            texts: Liste de textes
            batch_size: Taille du batch GPU
            show_progress: Afficher la barre de progression

        Returns:
            np.ndarray de shape (len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=False,
            normalize_embeddings=self.normalize,
        )

    def get_embedding_dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        return self.embedding_dim

    def __repr__(self) -> str:
        return f"MedicalEmbeddings(model={self.model_name}, dim={self.embedding_dim})"
