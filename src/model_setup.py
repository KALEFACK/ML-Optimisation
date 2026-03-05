"""
Module de configuration des modèles.
Optimisation pour CPU et support du transfert cross-lingue (P03).
"""

import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Caractéristiques 
MODELS_CONFIG = {
    "distilbert": {
        "id": "distilbert-base-uncased",
        "params": "66M",
        "memory": "500 Mo",
        "type": "M01 (CPU+)",
        "langue": "Anglais (Source)"
    },
    "camembert": {
        "id": "camembert-base",
        "params": "110M",
        "memory": "1.2 Go",
        "type": "M04 (GPU Rec.)",
        "langue": "Français (Cible/Baseline)"
    }
}

# ──────────────────────────────────────────────
# 1. ENVIRONNEMENT ET REPRODUCTIBILITÉ
# ──────────────────────────────────────────────

def setup_reproducibility(seed=42):
    """Fixe les graines pour la rigueur méthodologique."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Détecte le matériel et optimise les threads CPU ."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Recommandation : ajuster selon le processeur (standard=4)
        torch.set_num_threads(4)
    print(f"Dispositif utilisé : {device}")
    return device

# ──────────────────────────────────────────────
# 2. CHARGEMENT ADAPTATIF
# ──────────────────────────────────────────────

def load_model(model_key, num_labels=2, device=None, use_quantization=False):
    """
    Charge le modèle et le tokenizer avec adaptation automatique.
    Amélioration : gestion de la précision (float32 vs float16).
    """
    if model_key not in MODELS_CONFIG:
        raise ValueError("model_key doit être 'distilbert' ou 'camembert'")
    
    cfg = MODELS_CONFIG[model_key]
    if device is None:
        device = get_device()

    print(f"Chargement de {cfg['type']} ({cfg['id']})...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['id'])
    
    # Sélection du dtype selon le matériel pour optimiser la RAM [1]
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['id'],
        num_labels=num_labels,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        ignore_mismatched_sizes=True
    )
    
    model = model.to(device)
    
    # Application de la quantification si demandée
    if use_quantization and device.type == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Optimisation : Quantification dynamique appliquée.")

    return model, tokenizer

# ──────────────────────────────────────────────
# 3. STRATÉGIES DE STABILISATION (P03)
# ──────────────────────────────────────────────

def freeze_encoder(model, model_key):
    """
    Gèle l'encodeur pour stabiliser le transfert sur CPU.
    Réduit le nombre de paramètres entraînables pour accélérer la convergence.
    """
    keyword = "distilbert" if model_key == "distilbert" else "roberta"
    for name, param in model.named_parameters():
        if keyword in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stabilisation : Encodeur gelé. Params entraînables : {trainable:,}")
    return model

# ──────────────────────────────────────────────
# 4. ANALYSE COMPARATIVE (P03) 
# ──────────────────────────────────────────────

def print_model_comparison():
    """Affiche le tableau comparatif pour le rapport (P03)."""
    print("\n" + "="*65)
    print("COMPARAISON TECHNIQUE DES MODÈLES (PROBLÉMATIQUE P03)")
    print("="*65)
    print(f"{'Critère':<20} | {'DistilBERT (M01)':<20} | {'CamemBERT (M04)':<20}")
    print("-" * 65)
    print(f"{'Langue Source':<20} | {'Anglais':<20} | {'Français':<20}")
    print(f"{'Paramètres':<20} | {MODELS_CONFIG['distilbert']['params']:<20} | {MODELS_CONFIG['camembert']['params']:<20}")
    print(f"{'Mémoire estimée':<20} | {MODELS_CONFIG['distilbert']['memory']:<20} | {MODELS_CONFIG['camembert']['memory']:<20}")
    print(f"{'Tokenizer':<20} | {'WordPiece':<20} | {'SentencePiece':<20}")
    print("="*65 + "\n")

if __name__ == "__main__":
    setup_reproducibility()
    print_model_comparison()