"""
Module de gestion des données pour le Projet Transformers.
Problématique P03 : Transfert Cross-Lingue (Allociné D05).
Optimisé pour exécution sur CPU.
"""

import numpy as np
import pandas as pd
import re
import torch
from datasets import load_dataset

# Configuration selon les sources [4]
DATASET_NAME = "allocine"
LABEL_NAMES  = {0: "négatif", 1: "positif"}

# ──────────────────────────────────────────────
# 1. NETTOYAGE ET NORMALISATION
# ──────────────────────────────────────────────

def clean_text(text):
    """
    Amélioration : Normalise le texte pour stabiliser l'apprentissage.
    Supprime les balises HTML <br /> fréquentes dans Allociné et gère les espaces.
    """
    text = text.replace("<br />", " ")
    text = re.sub(r"\s+", " ", text) # Supprime les espaces multiples
    return text.strip()

# ──────────────────────────────────────────────
# 2. CHARGEMENT ET PRÉTRAITEMENT
# ──────────────────────────────────────────────

def load_allocine_dataset():
    """
    Charge le dataset Allociné (D05) et applique le nettoyage. [3]
    """
    print(f"Chargement du dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME)
    
    # Application du nettoyage sur tous les splits
    dataset = dataset.map(lambda x: {"review": clean_text(x["review"])})
    
    for split in ["train", "validation", "test"]:
        print(f"  [{split}] : {len(dataset[split])} exemples")
    return dataset

# ──────────────────────────────────────────────
# 3. ÉCHANTILLONNAGE ÉQUILIBRÉ
# ──────────────────────────────────────────────

def create_balanced_subset(dataset, split, n_per_class=100, seed=42):
    """
    Crée un sous-ensemble équilibré pour l'entraînement sur CPU.
    Utilise n_per_class=100 comme recommandé pour la validation rapide sur CPU [2].
    """
    full_split = dataset[split]
    
    # Filtrage par classe pour garantir l'équilibre
    pos_indices = [i for i, x in enumerate(full_split) if x["label"] == 1]
    neg_indices = [i for i, x in enumerate(full_split) if x["label"] == 0]
    
    # Échantillonnage aléatoire reproductible
    np.random.seed(seed)
    selected_pos = np.random.choice(pos_indices, n_per_class, replace=False)
    selected_neg = np.random.choice(neg_indices, n_per_class, replace=False)
    
    indices = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(indices)
    
    subset = full_split.select(indices)
    print(f"  Sous-ensemble [{split}] créé : {len(subset)} exemples ({n_per_class}/classe).")
    return subset

# ──────────────────────────────────────────────
# 4. TOKENISATION ADAPTATIVE
# ──────────────────────────────────────────────

def tokenize_dataset(dataset_split, tokenizer, max_length=128):
    """
    Tokenise avec troncation (truncation) pour économiser la RAM CPU.
    max_length=128 est le compromis standard pour les Transformers.
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples["review"],
            padding="max_length",
            truncation=True, # Indispensable pour la mémoire vive limitée
            max_length=max_length
        )

    tokenized = dataset_split.map(tokenize_fn, batched=True)
    # Renommage crucial pour le Trainer HuggingFace
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

# ──────────────────────────────────────────────
# 5. ANALYSE DU TOKENIZER (P03)
# ──────────────────────────────────────────────

def analyze_tokenizer_comparison(tok_distilbert, tok_camembert):
    """
    Protocole P03 : Analyse l'adaptation du tokenizer anglais au français.
    Mesure la fragmentation (ratio tokens/mots).
    """
    phrases = [
        "Ce film est extraordinairement bien réalisé.",
        "Une œuvre cinématographique d'une beauté époustouflante.",
        "Le scénario est médiocre et l'histoire incompréhensible."
    ]
    
    results = []
    for p in phrases:
        # Tokenisation brute (sans padding)
        t_db = tok_distilbert.tokenize(p)
        t_cb = tok_camembert.tokenize(p)
        
        n_mots = len(p.split())
        results.append({
            "Phrase": p[:30] + "...",
            "Mots": n_mots,
            "Tokens DistilBERT": len(t_db),
            "Tokens CamemBERT": len(t_cb),
            "Ratio DB": round(len(t_db)/n_mots, 2),
            "Ratio CB": round(len(t_cb)/n_mots, 2)
        })
    
    df = pd.DataFrame(results)
    print("\n--- Analyse de Fragmentation (P03) ---")
    print(df.to_string(index=False))
    print(f"\nFragmentation moyenne DistilBERT: {df['Ratio DB'].mean():.2f}")
    print(f"Fragmentation moyenne CamemBERT: {df['Ratio CB'].mean():.2f}")
    return df

if __name__ == "__main__":
    # Test du module
    ds = load_allocine_dataset()
    train_sub = create_balanced_subset(ds, "train", n_per_class=100)