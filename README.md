# ML-Optimisation — Transfert Cross-Lingue (P03)

Étude du transfert linguistique par fine-tuning de modèles Transformers.  
**Question centrale :** Un modèle pré-entraîné en anglais (DistilBERT) peut-il atteindre une efficacité comparable à un modèle natif (CamemBERT) sur le dataset Allociné (D05) ?

---

## Structure du projet

```
projet_Transformers/
├── main.py                   # Point d'entrée principal du pipeline
├── app.py                    # Interface applicative
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Prétraitement et sous-échantillonnage
│   ├── model_setup.py        # Initialisation et quantification
│   ├── optimization.py       # Random Search des hyperparamètres
│   └── visualisation.py      # Loss landscape et sharpness
├── notebooks/
│   ├── exploration.ipynb     # Analyse du tokenizer (fragmentation/UNK)
│   └── analysis.ipynb        # Analyses complémentaires
├── figures/                  # Graphiques générés automatiquement
│   ├── tokenizer_analysis.png
│   ├── loss_landscape.png
│   ├── convergence.png
│   ├── random_search.png
│   ├── confusion_matrices.png
│   └── eda_distribution.png
├── runs/                     # Checkpoints et logs des trials
└── mlop/                     # Environnement virtuel
```

---

## Installation

**Prérequis :** Python 3.9+ et un environnement isolé (recommandé).

```bash
# Créer et activer l'environnement
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Ordre d'exécution

Le pipeline complet s'exécute dans l'ordre suivant.

### Étape 1 — Pipeline principal (recommandé)

Lance toutes les étapes dans l'ordre : chargement des données, analyse du tokenizer, Random Search, loss landscape et visualisations.

```bash
python main.py
```

---

### Étape 2 — Exécution modulaire (optionnelle)

Pour relancer une étape spécifique indépendamment :

**2a. Chargement et analyse des données**
```bash
# Vérifie le dataset et génère l'analyse de fragmentation du tokenizer
python -c "from src.data_loader import load_allocine_dataset; load_allocine_dataset()"
```

**2b. Comparaison technique des modèles**
```bash
# Affiche le tableau comparatif DistilBERT vs CamemBERT
python -c "from src.model_setup import print_model_comparison; print_model_comparison()"
```

**2c. Random Search — DistilBERT (Source EN)**
```bash
python src/optimization.py --dataset D05 --model M01 --search_method random --n_iter 6
```

**2d. Random Search — CamemBERT (Natif FR)**
```bash
python src/optimization.py --dataset D05 --model M04 --search_method random --n_iter 6
```

**2e. Analyse géométrique du loss landscape**
```bash
python src/visualisation.py --checkpoint runs/best_model_distilbert.pt --n_points 12
python src/visualisation.py --checkpoint runs/best_model_camembert.pt  --n_points 12
```

---

## Configuration

Tous les paramètres sont centralisés dans `main.py` sous la variable `CONFIG` :

| Paramètre           | Valeur par défaut | Description |

| `n_per_class_train` |      500        | Échantillons par classe (train) |
| `n_per_class_val`   |      100        | Échantillons par classe (validation) |
| `n_per_class_test`  |      100        | Échantillons par classe (test) |
| `n_trials`          |       6         | Nombre d'essais du Random Search |
| `max_length`        |      12         | Troncature des séquences (tokens) |
| `label_smoothing`   |      0.1        | Régularisation pour minima plats |
| `max_grad_norm`     |      1.0        | Stabilisation des gradients |
| `seed`              |       42`       | Graine pour la reproductibilité |

---

## Reproductibilité

Les trois sources d'aléa sont fixées au démarrage via `setup_reproducibility()` :

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Les splits utilisent des seeds distinctes (`42 / 43 / 44`) pour garantir des sous-ensembles non corrélés.

---

## Optimisations CPU

Ce projet est conçu pour fonctionner sur CPU (RAM < 8 Go) :

- **Threading** : `torch.set_num_threads(4)` — adapter au nombre de cœurs physiques
- **Troncature** : `max_length=128` réduit quadratiquement la complexité de l'attention
- **Quantification dynamique** : conversion `fp32 → qint8` pour CamemBERT (~4x de réduction mémoire)
- **Sous-échantillonnage équilibré** : 500 exemples/classe en train au lieu des 160 000 originaux

---

## Résultats attendus

Après exécution complète, les fichiers suivants sont générés :

| Fichier | Contenu |
|---|---|
| `figures/tokenizer_analysis.png` | Ratio tokens/mot — WordPiece vs SentencePiece |
| `figures/random_search.png` | Distribution des F1-scores par modèle |
| `figures/convergence.png` | Courbes de convergence comparées |
| `figures/loss_landscape.png` | Visualisation 1D du paysage de perte |
| `runs/random_search_distilbert.json` | Logs complets des trials DistilBERT |
| `runs/random_search_camembert.json` | Logs complets des trials CamemBERT |
| `runs/summary.json` | Résumé comparatif final (delta F1) |

---

## Références

- Li et al. (2018) — *Visualizing the Loss Landscape of Neural Nets*
- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Martin et al. (2020) — *CamemBERT: a Tasty French Language Model*
- Dataset : [Allociné (D05)](https://huggingface.co/datasets/allocine) via HuggingFace