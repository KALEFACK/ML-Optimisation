# main.py
# Allociné (D05) | DistilBERT vs CamemBERT | P03 Cross-Lingue
# Optimisé pour CPU : Threads, Quantification et Label Smoothing

import os
import json
import numpy as np
import torch
from torch.quantization import quantize_dynamic

from src.data_loader   import (load_allocine_dataset, create_balanced_subset, 
                                tokenize_dataset, analyze_tokenizer_comparison)
from src.model_setup   import (load_model, fresh_model_fn, 
                                print_model_comparison, get_device)
from src.optimization  import (random_search, print_comparison_summary)
from src.visualisation import (compute_loss_landscape, compute_sharpness,
                                plot_loss_landscapes, plot_convergence,
                                plot_random_search_comparison, 
                                plot_tokenizer_analysis)

# ──────────────────────────────────────────────
# 1. CONFIGURATION ET RÉGLAGES CPU
# ──────────────────────────────────────────────

CONFIG = {
    # Sous-échantillonnage équilibré pour CPU
    "n_per_class_train": 200,   # Total 400
    "n_per_class_val":    50,   # Total 100
    "n_per_class_test":   50,   # Total 100

    # Hyperparamètres et Random Search
    "n_trials":   5,            # À augmenter pour le rapport final
    "max_length": 128,          # Sequence truncation pour RAM limitée
    
    # Améliorations de généralisation 
    "label_smoothing": 0.1,     # Favorise les minima plats
    "max_grad_norm": 1.0,       # Stabilisation des gradients Transformers
    
    # Répertoires et Reproductibilité
    "output_dir":  "./runs",
    "figures_dir": "./figures",
    "seed": 42,
}

# Initialisation de la reproductibilité [Sources 151, 185]
os.makedirs(CONFIG["output_dir"],  exist_ok=True)
os.makedirs(CONFIG["figures_dir"], exist_ok=True)
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# Optimisation du matériel : Threads CPU
device = get_device()
if device.type == 'cpu':
    torch.set_num_threads(4) 
    print(f"★ Optimisation : Utilisation de 4 threads CPU")

# ──────────────────────────────────────────────
# ÉTAPE 0 & 1 : PRÉSENTATION ET DONNÉES
# ──────────────────────────────────────────────

print_model_comparison()
raw_dataset = load_allocine_dataset()

# Création de sous-ensembles équilibrés pour éviter les biais
train_raw = create_balanced_subset(raw_dataset, "train",      CONFIG["n_per_class_train"], seed=42)
val_raw   = create_balanced_subset(raw_dataset, "validation", CONFIG["n_per_class_val"],   seed=43)
test_raw  = create_balanced_subset(raw_dataset, "test",       CONFIG["n_per_class_test"],  seed=44)

# ──────────────────────────────────────────────
# ÉTAPE 2 & 3 : MODÈLES ET ANALYSE TOKENIZER 
# ──────────────────────────────────────────────

# Chargement avec quantification dynamique pour réduction mémoire ~4x 
model_db, tok_db, _ = load_model("distilbert", device)
model_cb, tok_cb, _ = load_model("camembert",  device)

# Analyse de la fragmentation linguistique (Point clé P03)
df_tok = analyze_tokenizer_comparison(tok_db, tok_cb, 
                                      save_csv=f"{CONFIG['output_dir']}/tokenizer_analysis.csv")
plot_tokenizer_analysis(df_tok, save_path=f"{CONFIG['figures_dir']}/tokenizer_analysis.png")

# ──────────────────────────────────────────────
# ÉTAPE 4 & 5 : TOKENISATION ET RANDOM SEARCH
# ──────────────────────────────────────────────

train_db = tokenize_dataset(train_raw, tok_db, CONFIG["max_length"])
val_db   = tokenize_dataset(val_raw,   tok_db, CONFIG["max_length"])
train_cb = tokenize_dataset(train_raw, tok_cb, CONFIG["max_length"])
val_cb   = tokenize_dataset(val_raw,   tok_cb, CONFIG["max_length"])

# Exécution du Random Search sous conditions équitables
# Inclut AdamW, Cosine Annealing et Warmup en interne 
# ──────────────────────────────────────────────
# ÉTAPE 5 : RANDOM SEARCH — CORRIGÉE
# ──────────────────────────────────────────────

# L'entraînement doit se faire en float32 sur CPU.

results_db, best_db, trainer_db = random_search(
    model_fn  = fresh_model_fn("distilbert", device), # Utilisez la fonction simple
    train_ds  = train_db,
    val_ds    = val_db,
    model_key = "distilbert",
    n_trials  = CONFIG["n_trials"],
    output_dir= CONFIG["output_dir"],
)

results_cb, best_cb, trainer_cb = random_search(
    model_fn  = fresh_model_fn("camembert", device), # Utilisez la fonction simple
    train_ds  = train_cb,
    val_ds    = val_cb,
    model_key = "camembert",
    n_trials  = CONFIG["n_trials"],
    output_dir= CONFIG["output_dir"],
)
# ──────────────────────────────────────────────
# ÉTAPE 6 : LOSS LANDSCAPE ET SHARPNESS
# ──────────────────────────────────────────────

# Évaluation sur petit subset pour économiser le CPU [Source 120]
alphas_db, losses_db = compute_loss_landscape(trainer_db.model, val_db, n_points=12, epsilon=0.05)
alphas_cb, losses_cb = compute_loss_landscape(trainer_cb.model, val_cb, n_points=12, epsilon=0.05)

# Calcul de la métrique de platitude (Equation 1)
sharp_db = compute_sharpness(alphas_db, losses_db)
sharp_cb = compute_sharpness(alphas_cb, losses_cb)

print(f"\n➤ Sharpness DistilBERT (EN) : {sharp_db:.6f}")
print(f"➤ Sharpness CamemBERT  (FR) : {sharp_cb:.6f}")

# ──────────────────────────────────────────────
# ÉTAPE 7 & 8 : CONVERGENCE ET VISUALISATION FINALE
# ──────────────────────────────────────────────

plot_loss_landscapes(alphas_db, losses_db, alphas_cb, losses_cb, 
                     save_path=f"{CONFIG['figures_dir']}/loss_landscape.png")

plot_convergence(best_db.get("convergence_history", []), best_cb.get("convergence_history", []),
                 save_path=f"{CONFIG['figures_dir']}/convergence.png")

plot_random_search_comparison(results_db, results_cb, 
                              save_path=f"{CONFIG['figures_dir']}/random_search.png")

summary = print_comparison_summary(results_db, results_cb)
print("\n★ PIPELINE G11 TERMINÉ — RÉSULTATS EXPORTÉS ★")