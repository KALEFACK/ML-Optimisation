# ML-Optimisation
**1. Présentation du Projet et Objectifs (P03)**

Ce projet (P03) est dédié à l'étude du transfert linguistique "cross-lingue" à travers le fine-tuning de modèles Transformers. L'objectif est de quantifier l'efficacité de l'alignement sémantique d'un modèle massivement pré-entraîné sur une langue source (Anglais) lorsqu'il est confronté à une langue cible (Français).

**Question de recherche centrale** 

"Un modèle pré-entraîné en anglais (BERT-base-uncased) peut-il atteindre une efficacité comparable à un modèle natif (CamemBERT) sur le dataset Allociné (D05) ?"

**Objectifs pédagogiques** 
Maîtrise de l'écosystème HuggingFace : Implémentation de pipelines d'entraînement pour la classification de séquences.
Ingénierie sous contraintes : Optimisation des ressources pour une exécution sur architecture CPU (RAM < 8 Go).
Analyse de la généralisation : Évaluation de la robustesse des modèles via l'analyse géométrique du paysage de perte (loss landscape).
--------------------------------------------------------------------------------

**2. Configuration et Environnement**

Dépendances Critiques
L'isolation de l'environnement (via venv ou conda) est impérative pour garantir la reproductibilité.
transformers : Accès aux architectures BERT (M02) et CamemBERT (M04).
torch : Framework de calcul tensoriel.
datasets : Gestion optimisée du dataset Allociné.
sentencepiece : Requis pour le tokenizer de CamemBERT.
scikit-optimize / optuna : Support pour la recherche d'hyperparamètres.
tensorboard : Monitoring en temps réel de la convergence.
Structure du Projet (Best Practices MLOps)
projet_Transformers/
├── requirements.txt
├── src/
│   ├── data_loader.py    # Prétraitement et sous-échantillonnage
│   ├── model_setup.py    # Initialisation et quantification
│   ├── optimization.py   # Scripts de recherche (Random Search)
│   └── visualization.py  # Calcul du loss landscape et sharpness
├── notebooks/
│   └── exploration.ipynb # Analyse du tokenizer (fragmentation/UNK)
├── checkpoints/          # Sauvegarde des poids optimaux (.pt)
├── logs/                 # Traces TensorBoard et logs applicatifs
└── report/
    └── main.pdf          # Rapport scientifique (8-10 pages)
--------------------------------------------------------------------------------

**3. Protocole de Reproductibilité Scientifique**

Fixation des Graines Aléatoires (Seeds)
La stabilité des résultats repose sur le gel des sources d'aléa (initialisation des têtes linéaires, brassage des mini-batches). Le script doit fixer : random.seed, np.random.seed et torch.manual_seed.
Conditions de Comparaison Équitables
Les hyperparamètres suivants sont standardisés pour les deux modèles (BERT vs CamemBERT) :
Paramètre
Configuration
Architecture
Base-Architecture + Classification Head
Split
80% Train / 10% Val / 10% Test (Seed fixée)

Époques
3 époques (Early Stopping actif)

Optimiseur
AdamW 
Métrique
Accuracy / F1-Score
--------------------------------------------------------------------------------

**4. Analyse du Transfert Cross-Lingue (Protocole P03)**

Analyse du Tokenizer : WordPiece vs SentencePiece
L'expert doit quantifier la dégradation de la représentation textuelle en analysant :
Fragmentation : Mesurer le ratio sub-tokens/mots. BERT (WordPiece) risque de fragmenter excessivement le français par rapport à CamemBERT (SentencePiece/BPE).
Outil de vocabulaire : Calculer la fréquence de tokens [UNK] (Unknown) générés par BERT sur le dataset Allociné.
Convergence et Efficacité Wall-clock
L'analyse ne se limite pas aux itérations. Sur CPU, nous mesurerons :
Le temps réel écoulé (Wall-clock time) par époque.
Le nombre d'itérations nécessaires pour atteindre un plateau de performance stable
--------------------------------------------------------------------------------

**5. Optimisations Spécifiques pour l'Exécution sur CPU**

Gestion de la Mémoire et du Threading
Pour saturer intelligemment le processeur sans provoquer d'erreurs d'allocation :
import torch
torch.set_num_threads(4) # À adapter au nombre de cœurs physiques
Techniques de Réduction de Charge
Gradient Accumulation : Permet de simuler un batch_size important (ex: 32) sur une machine limitée à un batch de 4.
Sequence Truncation : Limiter max_length à 128 ou 256 tokens au lieu de 512 pour réduire quadratiquement la complexité de l'attention.
Quantification Dynamique : Conversion des poids nn.Linear de fp32 vers qint8 pour accélérer l'inférence.
--------------------------------------------------------------------------------

**6. Analyse de la Généralisation et du Loss Landscape**

Protocole de Perturbation 1D
Conformément aux travaux de Li et al. (2018), nous explorons la surface de perte autour du minimum local θ 
∗
  en utilisant une direction aléatoire d filter-normalisée pour préserver l'échelle des poids :
f(α)=L(θ*+αd)   avec α∈[−0.05,0.05].
Métrique de Sharpness

Interprétation scientifique : Un minimum "plat" (faible sharpness) est le signe d'une meilleure capacité de généralisation. L'utilisation de petits batch sizes lors de l'entraînement favorise généralement l'émergence de ces minima robustes sur les données de test (D05).
--------------------------------------------------------------------------------

**7. Instructions d'Exécution**

Installation
pip install -r requirements.txt
Lancement de l'Optimisation (Méthode G11 : Random Search)
Pour comparer les modèles sur Allociné :
# Entraînement BERT (Source EN)
python src/optimization.py --dataset D05 --model M02 --search_method random --n_iter 15

# Entraînement CamemBERT (Native FR)
python src/optimization.py --dataset D05 --model M04 --search_method random --n_iter 15
Analyse Géométrique
python src/visualization.py --checkpoint checkpoints/best_model_M04.pt --n_points 20
--------------------------------------------------------------------------------
