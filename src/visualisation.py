"""
Module de visualisation 
Analyse du Loss Landscape, Sharpness et Convergence.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ──────────────────────────────────────────────
# 1. ANALYSE DU LOSS LANDSCAPE
# ──────────────────────────────────────────────

def compute_loss_landscape(model, dataset, n_points=12, epsilon=0.05, device="cpu"):
    """
    Version légère pour CPU : perturbe les paramètres selon une direction 
    aléatoire et calcule la perte.
    """
    model.eval()
    model.to(device)
    # Sauvegarde des paramètres originaux
    original_params = [p.clone().detach() for p in model.parameters()]
    
    # Génération d'une direction aléatoire normalisée [Source 120]
    direction = [torch.randn_like(p) for p in original_params]
    norm = sum(torch.norm(d) for d in direction)
    direction = [d / norm for d in direction]
    
    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Application de la perturbation : theta = theta0 + alpha * d
            for p, p0, d in zip(model.parameters(), original_params, direction):
                p.data = p0 + alpha * d
            
            # Évaluation sur un petit subset pour économiser le CPU
            # Ici, on simule l'évaluation sur le dataset fourni
            total_loss = 0
            count = 0
            for i in range(min(len(dataset), 50)): 
                # OPTIMISATION : On ne garde que les colonnes tokenisées
                # On exclut 'label' et 'review' (qui est une chaîne de caractères)
                inputs = {
                    k: torch.tensor([v]) 
                    for k, v in dataset[i].items() 
                    if k in ['input_ids', 'attention_mask', 'token_type_ids']
                }
                
                labels = torch.tensor([dataset[i]['label']])
                
                # Passage au device et calcul
                model_inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**model_inputs, labels=labels.to(device))
                
                total_loss += outputs.loss.item()
                count += 1
            losses.append(total_loss / count)
            
    # Restauration des paramètres
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0
        
    return alphas, np.array(losses)

def compute_sharpness(alphas, losses):
    """
    Calcule la platitude (Sharpness) du minimum.
    Formule : 1/N * sum(|L(theta + epsilon) - L(theta)|)
    """
    center_idx = len(losses) // 2
    loss_theta = losses[center_idx]
    # Mesure de l'écart moyen par rapport au centre
    diffs = [abs(l - loss_theta) for l in losses]
    return np.mean(diffs)

# ──────────────────────────────────────────────
# 2. GRAPHIQUES DE COMPARAISON
# ──────────────────────────────────────────────

def plot_loss_landscapes(alphas_db, losses_db, alphas_cb, losses_cb, save_path):
    """Visualise la courbure des minima pour comparer la généralisation [Source 160]."""
    plt.figure(figsize=(10, 6))
    plt.plot(alphas_db, losses_db, 'r-o', label='DistilBERT (Anglais)')
    plt.plot(alphas_cb, losses_cb, 'b-o', label='CamemBERT (Français)')
    plt.title("Visualisation 1D du Loss Landscape (P03)")
    plt.xlabel("Direction de perturbation (alpha)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_convergence(history_db, history_cb, save_path):
    """Compare la vitesse d'apprentissage (P03)."""
    plt.figure(figsize=(10, 6))
    
    # Extraction des pertes d'évaluation depuis les logs de Trainer
    steps_db = [x['step'] for x in history_db if 'eval_loss' in x]
    loss_db = [x['eval_loss'] for x in history_db if 'eval_loss' in x]
    steps_cb = [x['step'] for x in history_cb if 'eval_loss' in x]
    loss_cb = [x['eval_loss'] for x in history_cb if 'eval_loss' in x]

    plt.plot(steps_db, loss_db, 'r--', label='DistilBERT Convergence')
    plt.plot(steps_cb, loss_cb, 'b-', label='CamemBERT Convergence')
    plt.title("Courbes de convergence : Transfert vs Natif")
    plt.xlabel("Steps")
    plt.ylabel("Eval Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_tokenizer_analysis(df_tok, save_path):
    """Analyse l'adaptation du tokenizer pour le transfert cross-lingue [Source 116]."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_tok, x='Model', y='Tokens_per_Word')
    plt.title("Fragmentation du Tokenizer (Ratio Tokens/Mot)")
    plt.ylabel("Nombre de tokens par mot")
    plt.savefig(save_path)
    plt.close()

def plot_random_search_comparison(results_db, results_cb, save_path):
    """Visualise la distribution des performances du Random Search [Source 150]."""
    data = []
    for r in results_db: data.append({'Model': 'DistilBERT', 'F1': r['f1']})
    for r in results_cb: data.append({'Model': 'CamemBERT', 'F1': r['f1']})
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Model', y='F1')
    sns.swarmplot(data=df, x='Model', y='F1', color=".25")
    plt.title("Distribution des scores F1 (Random Search)")
    plt.savefig(save_path)
    plt.close()