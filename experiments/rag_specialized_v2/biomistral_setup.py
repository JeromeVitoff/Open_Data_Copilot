#!/usr/bin/env python3
"""
Téléchargement et test minimal BioMistral-7B.

Vérifie que le modèle charge correctement sur le GPU disponible
et génère une réponse cohérente en français médical.

Usage:
    python -m experiments.rag_specialized_v2.biomistral_setup
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "BioMistral/BioMistral-7B"

print("=" * 70)
print("INSTALLATION BIOMISTRAL-7B")
print("=" * 70)

# --- GPU info ---
print(f"\nCUDA disponible : {torch.cuda.is_available()}")
print(f"GPU count       : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    vram = props.total_memory / 1024**3
    print(f"  GPU {i} : {torch.cuda.get_device_name(i)} ({vram:.1f} GB)")

# --- Chargement ---
print(f"\nChargement {MODEL_NAME}...")
print("  Premiere fois : ~14 GB a telecharger (~5-10 min)")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"  Tokenizer charge ({time.time()-t0:.1f}s)")

t1 = time.time()
# Restreindre aux A10 (Ampere) pour éviter FlashAttention sur RTX 2080 Ti (Turing)
# GPU 0 et 1 = A10 22GB — largement suffisant pour BioMistral-7B float16 (~14GB)
max_memory = {0: "20GiB", 1: "20GiB", 2: "0GiB", 3: "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # Désactive FlashAttention (incompatible Turing)
)
load_time = time.time() - t1
print(f"  Modele charge en {load_time:.1f}s")
print(f"  Device : {model.device}")
print(f"  Dtype  : {model.dtype}")

# Mémoire GPU utilisée
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        if used > 0:
            print(f"  GPU {i} VRAM : {used:.1f}/{total:.1f} GB utilise")

# --- Test génération ---
print("\nTEST GENERATION")
test_prompt = (
    "Tu es un assistant expert en santé publique française. Réponds en français, précisément.\n\n"
    "Question : Qu'est-ce qu'une infection respiratoire aiguë (IRA) et comment est-elle surveillée en France ?\n\n"
    "Réponse :"
)

inputs = tokenizer(test_prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

t_gen = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
gen_time = time.time() - t_gen

generated = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(generated, skip_special_tokens=True).strip()

print(f"\nReponse ({gen_time:.1f}s, {200/gen_time:.1f} tok/s) :")
print("-" * 60)
print(response[:400])
print("-" * 60)

print(f"\nBIOMISTRAL-7B OPERATIONNEL")
print(f"  Temps chargement : {load_time:.1f}s")
print(f"  Latence 200 tok  : {gen_time:.1f}s ({200/gen_time:.1f} tok/s)")
print(f"\nProchaine etape : python -m experiments.rag_specialized_v2.biomistral_quick_test")
