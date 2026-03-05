"""
Module de configuration des modèles - Groupe G11.
Optimisation pour CPU et support du transfert cross-lingue (P03).
"""

import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.quantization import quantize_dynamic

# Configuration des modèles selon les spécifications du projet [Source 112]
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
    """Fixe les graines pour la rigueur méthodologique [Source 124]."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Recommandé pour la reproductibilité totale sur CPU
    torch.use_deterministic_algorithms(False) 

def get_device():
    """Détecte le matériel et optimise les threads CPU [Source 114]."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Optimisation cruciale pour les Transformers sur CPU
        torch.set_num_threads(4) 
    return device

# ──────────────────────────────────────────────
# 2. CHARGEMENT ET INITIALISATION
# ──────────────────────────────────────────────

def load_model(model_key, device, verbose=False):
    """Charge le modèle et le tokenizer pour l'analyse initiale."""
    if model_key not in MODELS_CONFIG:
        raise ValueError("model_key doit être 'distilbert' ou 'camembert'")
    
    cfg = MODELS_CONFIG[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg['id'])
    
    # Utilisation de float32 sur CPU pour la stabilité [Source 114]
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['id'], 
        num_labels=2,
        torch_dtype=torch.float32 if device.type == 'cpu' else torch.float16
    )
    model.to(device)
    
    if verbose:
        print(f"  → Modèle {model_key} ({cfg['type']}) chargé sur {device}")
    return model, tokenizer, cfg['id']

def fresh_model_fn(model_key, device):
    """
    Retourne une fonction d'initialisation (model_init).
    Indispensable pour le Random Search afin de réinitialiser les poids
    à chaque nouvel essai (Trial) [Sources 124, 151].
    """
    cfg = MODELS_CONFIG[model_key]
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg['id'], 
            num_labels=2,
            torch_dtype=torch.float32 if device.type == 'cpu' else torch.float16
        )
        return model.to(device)
        
    return model_init

# ──────────────────────────────────────────────
# 3. OPTIMISATIONS ET STABILISATION (P03)
# ──────────────────────────────────────────────

def quantize_model(model):
    """
    Réduction de l'empreinte mémoire d'environ 4x via quantification dynamique.
    Essentiel pour CamemBERT sur CPU [Source 125].
    """
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def freeze_encoder(model, model_key):
    """
    Gèle l'encodeur pour stabiliser le transfert cross-lingue (P03).
    Seule la tête de classification reste entraînable.
    """
    # DistilBERT utilise 'distilbert', CamemBERT est basé sur 'roberta'
    keyword = "distilbert" if model_key == "distilbert" else "roberta"
    for name, param in model.named_parameters():
        if keyword in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  → Stabilisation : Encodeur gelé. Params entraînables : {trainable:,}")
    return model

# ──────────────────────────────────────────────
# 4. AFFICHAGE TECHNIQUE
# ──────────────────────────────────────────────

def print_model_comparison():
    """Affiche le tableau comparatif pour la problématique P03 [Source 116]."""
    print("\n" + "="*65)
    print("COMPARAISON TECHNIQUE DES MODÈLES (PROBLÉMATIQUE P03)")
    print("="*65)
    print(f"{'Critère':<20} | {'DistilBERT (M01)':<20} | {'CamemBERT (M04)':<20}")
    print("-" * 65)
    print(f"{'Langue Source':<20} | {MODELS_CONFIG['distilbert']['langue']:<20} | {MODELS_CONFIG['camembert']['langue']:<20}")
    print(f"{'Paramètres':<20} | {MODELS_CONFIG['distilbert']['params']:<20} | {MODELS_CONFIG['camembert']['params']:<20}")
    print(f"{'Mémoire estimée':<20} | {MODELS_CONFIG['distilbert']['memory']:<20} | {MODELS_CONFIG['camembert']['memory']:<20}")
    print(f"{'Tokenizer':<20} | {'WordPiece':<20} | {'SentencePiece':<20}")
    print("="*65 + "\n")