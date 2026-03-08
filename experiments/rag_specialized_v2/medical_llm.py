"""
LLM médical avec BioMistral-7B.

BioMistral-7B est un Mistral-7B fine-tuné sur PubMed Central (10M+ articles).
Utilisé en fallback si CUDA disponible et modèle téléchargé.

Pour utiliser via Ollama (plus simple) :
    ollama run mistral  # fallback si biomistral non disponible
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BioMistralLLM:
    """
    LLM spécialisé biomédical — BioMistral/BioMistral-7B.

    Requiert ~14 GB VRAM (float16) ou ~28 GB RAM (float32 CPU).

    Args:
        model_name: Modèle HuggingFace (défaut: BioMistral/BioMistral-7B)
        device_map: 'auto', 'cuda', 'cpu'
        load_in_4bit: Quantification 4-bit (réduit la mémoire, nécessite bitsandbytes)
    """

    def __init__(
        self,
        model_name: str = "BioMistral/BioMistral-7B",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        print(f"Chargement LLM medical : {model_name}")
        print(f"CUDA disponible : {torch.cuda.is_available()}")

        self.model_name = model_name

        kwargs: dict = {
            "device_map": device_map,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                kwargs.pop("torch_dtype", None)
                print("Quantification 4-bit activee (bitsandbytes)")
            except ImportError:
                print("bitsandbytes non disponible, chargement sans quantification")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        print(f"BioMistral charge — dtype : {self.model.dtype}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Génère une réponse à partir du prompt.

        Args:
            prompt: Texte complet (system + user)
            max_new_tokens: Longueur max de la réponse générée
            temperature: Température (0.0 = déterministe)

        Returns:
            Réponse textuelle générée
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Décoder uniquement les tokens générés (pas le prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
