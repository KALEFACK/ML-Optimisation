"""
Module de visualisation - Groupe G11.
Analyses pour la problématique P03 : Transfert Cross-Lingue (Allociné).
Optimisé pour CPU et rapports haute résolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import DataLoader

# Couleurs cohérentes pour le rapport [Source 123]
COLORS = {
    "distilbert": "#2563eb",   # Bleu (Modèle anglais)
    "camembert":  "#16a34a",   # Vert (Modèle français)
}

# ──────────────────────────────────────────────
# 1. ANALYSE DU LOSS LANDSCAPE (GÉNÉRALISATION)
# ──────────────────────────────────────────────

def _eval_loss_on_subset(model, dataset, n_samples=64, batch_size=8, device="cpu"):
    """Évalue la perte sur un subset avec batching pour l'efficacité CPU [Source 114]."""
    model.eval()
    model.to(device)
    
    indices = torch.randperm(len(dataset))[:n_samples]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size)
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**inputs)
            total_loss += outputs.loss.item() * inputs["input_ids"].size(0)
            
    return total_loss / n_samples

def compute_loss_landscape(model, dataset, n_points=12, epsilon=0.05, device="cpu"):
    """Perturbe les paramètres pour mesurer la platitude du minimum [Source 120, 159]."""
    model.eval()
    orig_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée [Source 120]
    direction = [torch.randn_like(p) for p in orig_params]
    total_norm = torch.sqrt(sum(torch.sum(d**2) for d in direction))
    direction = [d / total_norm for d in direction]

    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        for p, p0, d in zip(model.parameters(), orig_params, direction):
            p.data = p0 + alpha * d
        losses.append(_eval_loss_on_subset(model, dataset, device=device))

    # Restauration des paramètres [Source 121]
    for p, p0 in zip(model.parameters(), orig_params):
        p.data = p0.clone()

    return alphas, np.array(losses)

def compute_sharpness(alphas, losses):
    """Calcule la métrique de platitude (Sharpness) [Source 121, 162]."""
    l_theta_star = losses[len(losses) // 2]
    return float(np.mean(np.abs(losses - l_theta_star)))

def plot_loss_landscapes(alphas_db, losses_db, alphas_cb, losses_cb, save_path=None):
    """Compare la robustesse des minima (Plat vs Pointus) [Source 160, 194]."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, alphas, losses, key in [(axes, alphas_db, losses_db, "distilbert"), 
                                    (axes[6], alphas_cb, losses_cb, "camembert")]:
        sharp = compute_sharpness(alphas, losses)
        color = COLORS[key]
        label = "DistilBERT (M01)" if key == "distilbert" else "CamemBERT (M04)"
        
        ax.plot(alphas, losses, "o-", color=color, linewidth=2.5, label=f"{label}\nSharpness={sharp:.4f}")
        ax.axvline(0, color="#ef4444", linestyle="--", label="Optimum θ*")
        ax.fill_between(alphas, losses.min(), losses.max(), color=color, alpha=0.05)
        ax.set_title(f"Landscape : {label}", fontweight="bold")
        ax.set_xlabel("Perturbation (α)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes.set_ylabel("Cross-Entropy Loss")
    plt.suptitle("Analyse de Généralisation : Géométrie des Minima (P03)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    return fig

# ──────────────────────────────────────────────
# 2. ANALYSE DE CONVERGENCE ET PERFORMANCE (P03)
# ──────────────────────────────────────────────

def plot_convergence(history_db, history_cb, save_path=None):
    """Visualise la vitesse d'apprentissage pour le transfert cross-lingue [Source 117]."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for key, history in [("distilbert", history_db), ("camembert", history_cb)]:
        if not history: continue
        steps = [h["step"] for h in history]
        f1s = [h["eval_f1"] for h in history]
        losses = [h["eval_loss"] for h in history]
        
        axes.plot(steps, f1s, "o-", color=COLORS[key], label=key.capitalize())
        axes[6].plot(steps, losses, "s--", color=COLORS[key], label=key.capitalize())

    axes.set_title("Évolution du F1-score", fontweight="bold")
    axes[6].set_title("Évolution de la Perte", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Steps (Itérations)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Courbes de Convergence : Mesure de la vitesse d'adaptation (P03)", fontweight="bold")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    return fig

# ──────────────────────────────────────────────
# 3. BILAN DU RANDOM SEARCH ET TOKENIZER
# ──────────────────────────────────────────────

def plot_random_search_comparison(res_db, res_cb, save_path=None):
    """Analyse globale de l'efficacité et de la sensibilité [Source 117, 150]."""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Performance vs Learning Rate (Échelle Log) [Source 117]
    ax1 = fig.add_subplot(gs)
    for res, key in [(res_db, "distilbert"), (res_cb, "camembert")]:
        lrs = [r["params"]["learning_rate"] for r in res]
        f1s = [r["f1"] for r in res]
        ax1.scatter(lrs, f1s, color=COLORS[key], s=100, label=key.capitalize(), alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_title("Sensibilité au Learning Rate", fontweight="bold")
    ax1.set_xlabel("LR (Log Scale)")
    ax1.set_ylabel("F1-score")
    ax1.legend()

    # Efficacité temporelle (CPU) [Source 112]
    ax2 = fig.add_subplot(gs[6])
    times = [np.mean([r["time_sec"]/60 for r in res_db]), np.mean([r["time_sec"]/60 for r in res_cb])]
    ax2.bar(["DistilBERT", "CamemBERT"], times, color=[COLORS["distilbert"], COLORS["camembert"]], alpha=0.7)
    ax2.set_title("Temps d'entraînement moyen (min)", fontweight="bold")

    # Résumé des scores finaux
    ax3 = fig.add_subplot(gs[1, :])
    best_db = max([r["f1"] for r in res_db])
    best_cb = max([r["f1"] for r in res_cb])
    ax3.barh(["DistilBERT (EN)", "CamemBERT (FR)"], [best_db, best_cb], color=[COLORS["distilbert"], COLORS["camembert"]])
    ax3.set_xlim(0, 1.0)
    ax3.set_title("Meilleure Performance Finale (F1-score)", fontweight="bold")
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    return fig

def plot_tokenizer_analysis(df, save_path=None):
    """Visualise la fragmentation linguistique du tokenizer [Source 116]."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["ratio_distilbert"], 0.4, color=COLORS["distilbert"], label="DistilBERT (EN)")
    ax.bar(x + 0.2, df["ratio_camembert"], 0.4, color=COLORS["camembert"], label="CamemBERT (FR)")
    ax.axhline(1.0, color="black", linestyle="--", label="Ratio idéal (1 token/mot)")
    
    ax.set_xticks(x)
    ax.set_xticklabels([p[:20] for p in df["phrase"]], rotation=25)
    ax.set_ylabel("Ratio Tokens / Mots")
    ax.set_title("Analyse de la Fragmentation du Tokenizer (P03)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    return fig