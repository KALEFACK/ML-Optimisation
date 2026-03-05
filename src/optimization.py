"""
Module d'optimisation .
Implémentation du Random Search optimisée pour CPU et transfert cross-lingue (P03).
"""

import os
import json
import time
import random
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, f1_score

# ──────────────────────────────────────────────
# 1. ESPACE DE RECHERCHE 
# ──────────────────────────────────────────────

# Défini selon le Tableau 4 des sources pour une comparaison équitable (P03)
SEARCH_SPACE = {
    "learning_rate": {"type": "log_uniform", "low": 1e-6,  "high": 5e-4},
    "weight_decay":  {"type": "log_uniform", "low": 1e-8,  "high": 1e-2}, # 1e-8 simule le 0
    "batch_size":    {"type": "categorical", "choices": [6-8]},
    "warmup_steps":  {"type": "categorical", "choices": [9]},
    "num_epochs":    {"type": "int",         "low": 2,     "high": 5},
}

def sample_hyperparameters(seed=None):
    """Échantillonne aléatoirement les HP pour explorer l'espace de haute dimension."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    params = {}
    for name, cfg in SEARCH_SPACE.items():
        if cfg["type"] == "log_uniform":
            params[name] = float(np.exp(np.random.uniform(
                np.log(cfg["low"]), np.log(cfg["high"])
            )))
        elif cfg["type"] == "categorical":
            params[name] = random.choice(cfg["choices"])
        elif cfg["type"] == "int":
            params[name] = random.randint(cfg["low"], cfg["high"])
    return params

# ──────────────────────────────────────────────
# 2. MÉTRIQUES ET SUIVI DE CONVERGENCE 
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Calcule l'Accuracy et le F1-score ."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds, average="binary")),
    }

class ConvergenceCallback(TrainerCallback):
    """Protocole P03 : Enregistre l'évolution pour mesurer le nombre d'itérations."""
    def __init__(self):
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.history.append({
                "step":      state.global_step,
                "eval_f1":   metrics.get("eval_f1", 0.0),
                "eval_loss": metrics.get("eval_loss", 0.0),
            })

# ──────────────────────────────────────────────
# 3. ENTRAÎNEMENT D'UN ESSAI (TRIAL) 
# ──────────────────────────────────────────────

def train_one_trial(model_fn, train_ds, val_ds, params, trial_id, model_key, output_dir="./runs"):
    """Exécute un fine-tuning avec optimisations pour la stabilité et le CPU."""
    conv_cb = ConvergenceCallback()
    model = model_fn()
    t_start = time.time()

    # Arguments optimisés selon les sources pour Transformers sur CPU
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{model_key}_trial_{trial_id}",
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["num_epochs"],
        warmup_steps=params["warmup_steps"],
        
        # Stabilité : Gradient Clipping pour éviter les explosions [Source 143, 173]
        max_grad_norm=1.0, 
        # Mémoire : Accumulation pour simuler des batchs plus grands sur RAM limitée [Source 108]
        gradient_accumulation_steps=2,
        
        evaluation_strategy="steps",
        eval_steps=50, # Mesure fine pour le protocole P03
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        
        # Configuration CPU
        no_cuda=True,
        fp16=False, # float32 recommandé pour stabilité CPU [Source 114]
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            conv_cb
        ],
    )

    trainer.train()
    eval_results = trainer.evaluate()
    elapsed = time.time() - t_start

    result = {
        "trial_id": trial_id,
        "params": params,
        "f1": eval_results.get("eval_f1", 0.0),
        "accuracy": eval_results.get("eval_accuracy", 0.0),
        "time_sec": elapsed,
        "steps_to_optimum": conv_cb.history[-1]["step"] if conv_cb.history else 0,
        "history": conv_cb.history
    }
    return result, trainer

# ──────────────────────────────────────────────
# 4. RANDOM SEARCH & RÉSUMÉ 
# ──────────────────────────────────────────────

def random_search(model_fn, train_ds, val_ds, model_key, n_trials=5, output_dir="./runs"):
    """Boucle principale de recherche avec garantie de reproductibilité."""
    all_results = []
    best_result = None
    best_trainer = None

    for i in range(n_trials):
        # Utilisation de seeds fixes pour comparer DistilBERT et CamemBERT sur les mêmes HP 
        params = sample_hyperparameters(seed=i * 100)
        result, trainer = train_one_trial(model_fn, train_ds, val_ds, params, i+1, model_key, output_dir)
        
        all_results.append(result)
        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = result
            best_trainer = trainer

    # Sauvegarde JSON pour le rapport final
    with open(f"{output_dir}/results_{model_key}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results, best_result, best_trainer