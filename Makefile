PYTHON     = python
SRC        = src
RUNS_DIR   = runs
FIGS_DIR   = figures
DASHBOARD  = dashboard/app.py

.DEFAULT_GOAL := help

help:
	@echo "======================================================"
	@echo "  Pipeline G11 - P03 Transfert Cross-Lingue"
	@echo "======================================================"
	@echo "  make install           Installer les dependances"
	@echo "  make all               Pipeline complet (~5h CPU)"
	@echo "  make data              Etape 1 - Charger Allocine"
	@echo "  make tokenizer         Etape 2 - Analyser fragmentation"
	@echo "  make train             Etape 3 - Random Search 2 modeles"
	@echo "  make train-distilbert  Etape 3a - DistilBERT (~2h30)"
	@echo "  make train-camembert   Etape 3b - CamemBERT (~2h30)"
	@echo "  make landscape         Etape 4 - Loss Landscape"
	@echo "  make visualize         Etape 5 - Figures"
	@echo "  make dashboard         Etape 6 - Lancer Streamlit"
	@echo "  make clean             Supprimer cache Python"
	@echo "  make clean-runs        Supprimer checkpoints"
	@echo "  make clean-all         Tout nettoyer"
	@echo "  make git-save          git add + commit + push"

install:
	pip install -r requirements.txt

all: data tokenizer train landscape visualize
	@echo "Pipeline G11 termine"

data:
	@echo "-> [1/5] Chargement Allocine..."
	$(PYTHON) -c "from $(SRC).data_loader import load_allocine_dataset; ds = load_allocine_dataset(); print('Train:', len(ds['train']))"

tokenizer:
	@echo "-> [2/5] Analyse tokenizer..."
	$(PYTHON) -c "from $(SRC).model_setup import load_model, get_device; from $(SRC).data_loader import analyze_tokenizer_comparison; import os; os.makedirs('$(RUNS_DIR)', exist_ok=True); device = get_device(); _, tok_db, _ = load_model('distilbert', device); _, tok_cb, _ = load_model('camembert', device); analyze_tokenizer_comparison(tok_db, tok_cb, save_csv='$(RUNS_DIR)/tokenizer_analysis.csv')"

train: train-distilbert train-camembert
	@echo "Random Search termine"

train-distilbert:
	@echo "-> [3a/5] Random Search DistilBERT..."
	$(PYTHON) -c "from $(SRC).model_setup import get_device, fresh_model_fn, load_model; from $(SRC).data_loader import load_allocine_dataset, create_balanced_subset, tokenize_dataset; from $(SRC).optimization import random_search; import os; os.makedirs('$(RUNS_DIR)', exist_ok=True); device = get_device(); ds = load_allocine_dataset(); _, tok, _ = load_model('distilbert', device); train = tokenize_dataset(create_balanced_subset(ds, 'train', 500), tok); val = tokenize_dataset(create_balanced_subset(ds, 'validation', 100, seed=43), tok); random_search(fresh_model_fn('distilbert', device), train, val, 'distilbert', 6, '$(RUNS_DIR)')"

train-camembert:
	@echo "-> [3b/5] Random Search CamemBERT..."
	$(PYTHON) -c "from $(SRC).model_setup import get_device, fresh_model_fn, load_model; from $(SRC).data_loader import load_allocine_dataset, create_balanced_subset, tokenize_dataset; from $(SRC).optimization import random_search; import os; os.makedirs('$(RUNS_DIR)', exist_ok=True); device = get_device(); ds = load_allocine_dataset(); _, tok, _ = load_model('camembert', device); train = tokenize_dataset(create_balanced_subset(ds, 'train', 500), tok); val = tokenize_dataset(create_balanced_subset(ds, 'validation', 100, seed=43), tok); random_search(fresh_model_fn('camembert', device), train, val, 'camembert', 6, '$(RUNS_DIR)')"

landscape:
	@echo "-> [4/5] Loss Landscape..."
	$(PYTHON) -c "from $(SRC).visualisation import compute_loss_landscape, compute_sharpness; from $(SRC).model_setup import load_model, get_device; from $(SRC).data_loader import load_allocine_dataset, create_balanced_subset, tokenize_dataset; device = get_device(); ds = load_allocine_dataset(); model_db, tok_db, _ = load_model('distilbert', device); model_cb, tok_cb, _ = load_model('camembert', device); val_db = tokenize_dataset(create_balanced_subset(ds, 'validation', 100, seed=43), tok_db); val_cb = tokenize_dataset(create_balanced_subset(ds, 'validation', 100, seed=43), tok_cb); a_db, l_db = compute_loss_landscape(model_db, val_db); a_cb, l_cb = compute_loss_landscape(model_cb, val_cb); print('Sharpness DistilBERT:', round(compute_sharpness(a_db, l_db), 6)); print('Sharpness CamemBERT :', round(compute_sharpness(a_cb, l_cb), 6))"

visualize:
	@echo "-> [5/5] Generation figures..."
	@mkdir -p $(FIGS_DIR)
	$(PYTHON) -c "import json; from $(SRC).visualisation import plot_random_search_comparison; db = json.load(open('$(RUNS_DIR)/random_search_distilbert.json')); cb = json.load(open('$(RUNS_DIR)/random_search_camembert.json')); plot_random_search_comparison(db, cb, '$(FIGS_DIR)/random_search.png'); print('Figures dans $(FIGS_DIR)/')"

dashboard:
	@echo "-> Lancement Streamlit sur http://localhost:8504"
	streamlit run $(DASHBOARD) --server.port 8504

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-runs:
	find $(RUNS_DIR) -type d -name "checkpoint-*" -exec rm -rf {} + 2>/dev/null || true
	find $(RUNS_DIR) -name "events.out.tfevents.*" -delete 2>/dev/null || true

clean-all: clean clean-runs
	rm -rf $(FIGS_DIR)/*.png 2>/dev/null || true

git-save:
	git add .
	git commit -m "chore: sauvegarde automatique"
	git push origin main

.PHONY: help install all data tokenizer train train-distilbert train-camembert \
        landscape visualize dashboard clean clean-runs clean-all git-save