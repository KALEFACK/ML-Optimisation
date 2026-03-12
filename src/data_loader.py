"""
Module de chargement et traitement des données.
Support pour Allociné (D05) et analyse de fragmentation (P03).
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

def load_allocine_dataset():
    """Charge le dataset Allociné depuis HuggingFace."""
    print("  → Chargement du dataset Allociné (D05)...")
    return load_dataset("allocine")

def create_balanced_subset(dataset, split_name, n_per_class, seed=42):
    """
    Crée un sous-ensemble équilibré pour le CPU.
    Garantit la rigueur méthodologique.
    """
    ds = dataset[split_name]
    
    # Extraction des indices par classe
    idx_neg = [i for i, x in enumerate(ds['label']) if x == 0]
    idx_pos = [i for i, x in enumerate(ds['label']) if x == 1]
    
    # Échantillonnage aléatoire reproductible
    np.random.seed(seed)
    selected_neg = np.random.choice(idx_neg, n_per_class, replace=False)
    selected_pos = np.random.choice(idx_pos, n_per_class, replace=False)
    
    indices = np.concatenate([selected_neg, selected_pos])
    return ds.select(indices)

def analyze_tokenizer_comparison(tok_db, tok_cb, save_csv=None):
    """
    Analyse la fragmentation linguistique pour P03.
    Compare DistilBERT (anglais) vs CamemBERT (français).
    """
    exemples = [
        "Ce film est absolument magnifique et touchant.",
        "Une perte de temps totale, l'intrigue est creuse.",
        "Les acteurs sont bons mais la réalisation laisse à désirer.",
        "Un chef-d'œuvre du cinéma français contemporain."
    ]
    
    stats = []
    for txt in exemples:
        # Calcul du ratio tokens/mots
        nb_mots = len(txt.split())
        tokens_db = len(tok_db.tokenize(txt))
        tokens_cb = len(tok_cb.tokenize(txt))
        
        stats.append({
            "Texte": txt[:30] + "...",
            "DistilBERT_ratio": tokens_db / nb_mots,
            "CamemBERT_ratio": tokens_cb / nb_mots
        })
    
    df = pd.DataFrame(stats)
    
    # Moyenne pour le graphique
    df_plot = pd.DataFrame({
        'Model': ['DistilBERT', 'CamemBERT'],
        'Tokens_per_Word': [df['DistilBERT_ratio'].mean(), df['CamemBERT_ratio'].mean()]
    })
    
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"  → Analyse du tokenizer sauvegardée dans {save_csv}")
        
    return df_plot

def tokenize_dataset(dataset, tokenizer, max_length=128):
    """Tokenise le dataset avec troncature pour le CPU ."""
    def _tokenize_fn(batch):
        return tokenizer(
            batch["review"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    return dataset.map(_tokenize_fn, batched=True)