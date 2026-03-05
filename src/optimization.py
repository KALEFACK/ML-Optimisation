"""
Module d'optimisation - Groupe G11.
Implémente le Random Search et l'analyse des performances (P03).
"""

import numpy as np
import json
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    """Calcule le F1-score (métrique principale G11) et l'Accuracy [Source 109]."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions, average="binary"),
        "accuracy": accuracy_score(labels, predictions)
    }

def sample_hyperparameters():
    """
    Échantillonne les hyperparamètres et les convertit en types Python natifs.
    Indispensable pour la compatibilité JSON et la sauvegarde des logs [Source 117].
    """
    return {
        # float(...) convertit le np.float64 en float Python standard
        "learning_rate": float(10**np.random.uniform(-6, -3.3)), 
        "weight_decay": float(10**np.random.uniform(-4, -2)),   
        
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        
        # int(...) convertit le np.int32 en int Python standard
        "warmup_steps": int(np.random.choice([1])) 
    }

def random_search(model_fn, train_ds, val_ds, model_key, n_trials, output_dir):
    """
    Exécute une recherche aléatoire pour trouver les meilleurs HP [Source 149].
    Garantit la reproductibilité via des conditions équitables.
    """
    results = []
    best_f1 = -1
    best_trainer = None
    best_trial_data = None

    print(f"\n🚀 Lancement du Random Search pour {model_key} ({n_trials} essais)...")

    for i in range(n_trials):
        hp = sample_hyperparameters()
        print(f"  Trial {i+1}/{n_trials} | LR: {hp['learning_rate']:.2e} | WD: {hp['weight_decay']:.2e}")

        # Initialisation d'un modèle neuf à chaque essai [Source 124]
        model = model_fn()

        args = TrainingArguments(
            output_dir=f"{output_dir}/{model_key}_trial_{i}",
            learning_rate=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
            num_train_epochs=hp["num_train_epochs"],
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            warmup_steps=hp["warmup_steps"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            # Optimisations CPU [Source 108]
            fp16=False, 
            no_cuda=True if not torch.cuda.is_available() else False
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate()
        
        trial_data = {
            "trial": i,
            "hp": hp,
            "f1": eval_result["eval_f1"],
            "accuracy": eval_result["eval_accuracy"],
            "convergence_history": trainer.state.log_history
        }
        results.append(trial_data)

        if eval_result["eval_f1"] > best_f1:
            best_f1 = eval_result["eval_f1"]
            best_trainer = trainer
            best_trial_data = trial_data

    # Sauvegarde des résultats
    with open(f"{output_dir}/random_search_{model_key}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results, best_trial_data, best_trainer

def print_comparison_summary(results_db, results_cb):
    """
    FONCTION MANQUANTE : Compare DistilBERT vs CamemBERT (P03) [Source 123].
    """
    # Extraction des meilleurs F1-scores
    best_f1_db = max([r['f1'] for r in results_db])
    best_f1_cb = max([r['f1'] for r in results_cb])
    delta = best_f1_cb - best_f1_db

    summary = {
        "distilbert": {"f1": best_f1_db},
        "camembert":  {"f1": best_f1_cb},
        "delta_f1":    delta
    }

    print("\n" + "="*40)
    print("RÉSUMÉ DE LA COMPARAISON (P03)")
    print("="*40)
    print(f"Meilleur F1 DistilBERT (EN) : {best_f1_db:.4f}")
    print(f"Meilleur F1 CamemBERT  (FR) : {best_f1_cb:.4f}")
    print(f"Écart de performance       : {delta:+.4f}")
    
    if delta > 0:
        print("→ CamemBERT (natif) surpasse le transfert anglais.")
    else:
        print("→ DistilBERT (anglais) est compétitif malgré la langue.")
    print("="*40)

    return summary