"""
Tableau de Bord Interactif - Groupe G11
Allociné D05 | DistilBERT vs CamemBERT | Problématique P03
Run : streamlit run dashboard_g11.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="G11 · DistilBERT vs CamemBERT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #ffffff; color: #1a1a2e; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f5f7fa !important;
        border-right: 1px solid #dde3ed;
    }
    section[data-testid="stSidebar"] * { color: #1a1a2e !important; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f5f7fa;
        border: 1px solid #dde3ed;
        border-radius: 10px;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        color: #1a1a2e !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #5a6a8a !important;
    }
    [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-bottom: 1px solid #dde3ed;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 6px 6px 0 0;
        color: #5a6a8a;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eef2f8 !important;
        border: 1px solid #dde3ed !important;
        color: #1a1a2e !important;
    }

    /* Titles */
    h1, h2, h3 { color: #1a1a2e !important; font-family: 'JetBrains Mono', monospace !important; }

    /* DataFrames */
    .stDataFrame { background: #f5f7fa !important; border: 1px solid #dde3ed; border-radius: 8px; }

    /* Divider */
    hr { border-color: #dde3ed; }

    /* Card section */
    .card {
        background: #f5f7fa;
        border: 1px solid #dde3ed;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .tag-db { color: #e03e3e; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    .tag-cb { color: #0a9a93; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
    }
    .section-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #5a6a8a;
        border-left: 3px solid #0a9a93;
        padding-left: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
DB_COLOR = "#e03e3e"
CB_COLOR = "#0a9a93"
GOLD     = "#d4820a"
PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f5f7fa",
    font=dict(family="JetBrains Mono, monospace", color="#1a1a2e", size=11),
    legend=dict(bgcolor="#ffffff", bordercolor="#dde3ed", borderwidth=1),
    margin=dict(l=40, r=20, t=40, b=40),
)
GRID = dict(gridcolor="#dde3ed", linecolor="#dde3ed", zerolinecolor="#dde3ed")

def apply_theme(fig):
    """Applique le thème sombre + grille à tous les axes d'une figure."""
    fig.update_layout(**PLOTLY_TEMPLATE)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
# Le dashboard est dans dashboard/ → on remonte d'un niveau pour atteindre la racine du projet
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR    = os.path.join(BASE, "runs")
FIGURES_DIR = os.path.join(BASE, "figures")

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def safe_load_json(filename):
    path = os.path.join(RUNS_DIR, filename)
    if os.path.exists(path):
        return load_json(path)
    return None

def safe_load_csv(filename):
    path = os.path.join(RUNS_DIR, filename)
    if os.path.exists(path):
        return load_csv(path)
    return None

def safe_load_image(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None

# Chargement
results_db_raw = safe_load_json("random_search_distilbert.json")
results_cb_raw = safe_load_json("random_search_camembert.json")
df_tok         = safe_load_csv("tokenizer_analysis.csv")

# ── Parsing des résultats ──────────────────────────────────────────────────
def parse_results(raw):
    if not raw:
        return pd.DataFrame()
    rows = []
    for r in raw:
        row = {
            "trial":    r.get("trial_id", r.get("trial", 0)),
            "f1":       r.get("f1", 0),
            "accuracy": r.get("accuracy", 0),
            "lr":       r.get("params", r.get("hp", {})).get("learning_rate", 0),
            "wd":       r.get("params", r.get("hp", {})).get("weight_decay", 0),
            "epochs":   r.get("params", r.get("hp", {})).get("num_train_epochs", 0),
            "time_min": r.get("time_sec", 0) / 60,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def extract_convergence(raw):
    """Extrait l'historique eval_loss du meilleur trial."""
    if not raw:
        return pd.DataFrame()
    best = max(raw, key=lambda r: r.get("f1", 0))
    history = best.get("convergence_history", [])
    steps, losses, f1s = [], [], []
    for entry in history:
        if "eval_loss" in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry["eval_loss"])
            f1s.append(entry.get("eval_f1", None))
    return pd.DataFrame({"step": steps, "eval_loss": losses, "eval_f1": f1s})

df_db   = parse_results(results_db_raw)
df_cb   = parse_results(results_cb_raw)
conv_db = extract_convergence(results_db_raw)
conv_cb = extract_convergence(results_cb_raw)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Pipeline G11")
    st.markdown("---")
    st.markdown("""
    <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; color: #5a6a8a; letter-spacing: 2px;'>
    DATASET<br>
    </div>
    <div style='font-size: 0.85rem; margin-bottom: 12px;'>Allociné (D05)</div>
    <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; color: #5a6a8a; letter-spacing: 2px;'>
    PROBLÉMATIQUE<br>
    </div>
    <div style='font-size: 0.85rem; margin-bottom: 12px;'>P03 · Cross-Lingue</div>
    <div style='font-family: JetBrains Mono, monospace; font-size: 0.7rem; color: #5a6a8a; letter-spacing: 2px;'>
    SEED<br>
    </div>
    <div style='font-size: 0.85rem; margin-bottom: 12px;'>42 (reproductible)</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Statut des fichiers
    st.markdown("**📁 Fichiers détectés**")
    files = {
        "random_search_distilbert.json": results_db_raw is not None,
        "random_search_camembert.json":  results_cb_raw is not None,
        "tokenizer_analysis.csv":        df_tok is not None,
        "figures/convergence.png":       os.path.exists(os.path.join(FIGURES_DIR, "convergence.png")),
        "figures/loss_landscape.png":    os.path.exists(os.path.join(FIGURES_DIR, "loss_landscape.png")),
        "figures/random_search.png":     os.path.exists(os.path.join(FIGURES_DIR, "random_search.png")),
        "figures/tokenizer_analysis.png":os.path.exists(os.path.join(FIGURES_DIR, "tokenizer_analysis.png")),
    }
    for fname, ok in files.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"`{icon}` `{fname}`")

    st.markdown("---")
    if not df_db.empty:
        n_trials = len(df_db)
        st.metric("Trials DistilBERT", n_trials)
    if not df_cb.empty:
        n_trials_cb = len(df_cb)
        st.metric("Trials CamemBERT", n_trials_cb)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='border-bottom: 1px solid #dde3ed; padding-bottom: 16px; margin-bottom: 24px;'>
  <div style='font-family: JetBrains Mono, monospace; font-size: 0.65rem; color: #5a6a8a; letter-spacing: 4px; margin-bottom: 6px;'>
    GROUPE G11 · ALLOCINÉ D05 · PROBLÉMATIQUE P03
  </div>
  <div style='font-size: 1.6rem; font-weight: 700; font-family: JetBrains Mono, monospace;'>
    <span style='color: #ff6b6b;'>DistilBERT</span>
    <span style='color: #aab4c8; margin: 0 12px;'>vs</span>
    <span style='color: #4ecdc4;'>CamemBERT</span>
    <span style='color: #5a6a8a; font-size: 0.85rem; margin-left: 12px;'>Transfert Cross-Lingue</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MÉTRIQUES GLOBALES 
# ─────────────────────────────────────────────
if not df_db.empty and not df_cb.empty:
    best_f1_db  = df_db["f1"].max()
    best_f1_cb  = df_cb["f1"].max()
    mean_f1_db  = df_db["f1"].mean()
    mean_f1_cb  = df_cb["f1"].mean()
    delta       = best_f1_cb - best_f1_db

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best F1 · CamemBERT", f"{best_f1_cb:.4f}", delta=f"+{best_f1_cb - mean_f1_cb:.4f} vs moy.")
    c2.metric("Best F1 · DistilBERT", f"{best_f1_db:.4f}", delta=f"+{best_f1_db - mean_f1_db:.4f} vs moy.")
    c3.metric("Δ F1 (CB − DB)", f"{delta:+.4f}", delta="CamemBERT gagne" if delta > 0 else "DistilBERT gagne")
    c4.metric("Trials DistilBERT", len(df_db))
    c5.metric("Trials CamemBERT", len(df_cb))
    st.markdown("---")

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Vue Générale",
    "🔤 Tokenizer P03",
    "🏋️ Entraînement",
    "📉 Loss Landscape",
    "🏆 Comparaison Finale",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 · VUE GÉNÉRALE
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">Spécifications Techniques · Comparaison Modèles</div>', unsafe_allow_html=True)

    specs = pd.DataFrame({
        "Critère":   ["Paramètres", "Mémoire estimée", "Tokenizer", "Type", "Langue source", "Optim. CPU"],
        "DistilBERT": ["66M", "500 Mo", "WordPiece", "M01 (CPU+)", "Anglais (Source)", "fp32 + 4 threads"],
        "CamemBERT":  ["110M", "1.2 Go", "SentencePiece", "M04 (GPU Rec.)", "Français (Cible)", "Quantif. dynamique"],
    })
    st.dataframe(specs.set_index("Critère"), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Pipeline · Flux des Étapes</div>', unsafe_allow_html=True)

    steps = [
        ("0", "Présentation", "Specs modèles"),
        ("1", "Données", "Allociné D05\n500 train / 200 val / 200 test"),
        ("2-3", "Modèles", "Chargement +\nQuantification ~4x"),
        ("4", "Tokenisation", "max_length=128\nPadding + Troncature"),
        ("5", "Random Search", f"{len(df_db) if not df_db.empty else '?'} trials/modèle\nAdamW + EarlyStopping"),
        ("6", "Loss Landscape", "n=12 points\nε=0.05 + Sharpness"),
        ("7-8", "Visualisation", "Export figures\n+ Résumé final"),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style='text-align:center; padding: 10px 4px; background:#f5f7fa; border:1px solid #dde3ed; border-top: 2px solid #0a9a93; border-radius: 8px;'>
          <div style='font-family: JetBrains Mono, monospace; font-size: 1rem; color: #0a9a93; font-weight: 700;'>{num}</div>
          <div style='font-size: 0.75rem; font-weight: 600; color: #1a1a2e; margin: 4px 0;'>{title}</div>
          <div style='font-size: 0.62rem; color: #5a6a8a; white-space: pre-line;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Configuration · Hyperparamètres Fixes</div>', unsafe_allow_html=True)
    config_data = {
        "Paramètre": ["n_per_class_train", "n_per_class_val", "n_per_class_test", "max_length", "label_smoothing", "max_grad_norm", "seed"],
        "Valeur":    [500,100,100,128,0.1,1.0,42],
        "Description": [
            "500 négatifs + 500 positifs =500 total",
            "100 néatifs + 100 positifs =200 total",
            "100 négatifs + 100 positifs =200 total",
            "Troncature séquences (RAM CPU)",
            "Minima plats (généralisation)",
            "Stabilisation gradients Transformers",
            "Reproductibilité complète",
        ]
    }
    st.dataframe(pd.DataFrame(config_data).set_index("Paramètre"), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 · TOKENIZER
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Fragmentation Linguistique · Ratio Tokens / Mot</div>', unsafe_allow_html=True)

    col_img, col_data = st.columns([1, 1])

    # Image de l'analyse de tokenisation
    img_tok = safe_load_image("tokenizer_analysis.png")
    if img_tok:
        col_img.markdown("**Figure de l'analyse de tokenisation:**")
        col_img.image(img_tok, use_container_width=True)
    else:
        col_img.warning("tokenizer_analysis.png non trouvé dans ./figures/")

    # Données CSV
    with col_data:
        if df_tok is not None:
            st.markdown("**Données brutes · tokenizer_analysis.csv :**")
            st.dataframe(df_tok, use_container_width=True)

            # Bar chart interactif
            if "DistilBERT_ratio" in df_tok.columns and "CamemBERT_ratio" in df_tok.columns:
                fig = go.Figure()
                x_labels = df_tok["Texte"].tolist() if "Texte" in df_tok.columns else [f"Ex. {i+1}" for i in range(len(df_tok))]
                fig.add_bar(name="DistilBERT", x=x_labels, y=df_tok["DistilBERT_ratio"], marker_color=DB_COLOR)
                fig.add_bar(name="CamemBERT",  x=x_labels, y=df_tok["CamemBERT_ratio"],  marker_color=CB_COLOR)
                apply_theme(fig)

                fig.update_layout(
                    title="Ratio Tokens/Mot par Exemple",
                    barmode="group",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Moyennes
                mean_db = df_tok["DistilBERT_ratio"].mean()
                mean_cb = df_tok["CamemBERT_ratio"].mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Moy. DistilBERT", f"{mean_db:.3f}x")
                c2.metric("Moy. CamemBERT",  f"{mean_cb:.3f}x")
                c3.metric("Écart Fragmentation", f"{mean_db - mean_cb:+.3f}x", delta="WordPiece fragmente +")
        else:
            st.warning("tokenizer_analysis.csv non trouvé dans ./runs/")

    st.markdown("---")
    st.info("""
    **Interprétation P03** : WordPiece (DistilBERT, entraîné sur l'anglais) fragmente davantage les mots français 
    → plus de tokens  → handicap pour le transfert cross-lingue.
    SentencePiece (CamemBERT) est entraîné sur le français → tokenisation plus naturelle.
    """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 · ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">Random Search · Résultats par Trial</div>', unsafe_allow_html=True)

    if df_db.empty and df_cb.empty:
        st.error("Aucun fichier JSON trouvé dans ./runs/. Vérifiez le chemin.")
    else:
        # ── F1 par trial ──
        fig_f1 = make_subplots(rows=1, cols=2,
            subplot_titles=["DistilBERT · F1 par Trial", "CamemBERT · F1 par Trial"])

        if not df_db.empty:
            fig_f1.add_bar(row=1, col=1, name="DistilBERT F1",
                x=df_db["trial"].astype(str), y=df_db["f1"],
                marker_color=DB_COLOR, showlegend=True,
                text=df_db["f1"].round(4), textposition="outside")
        if not df_cb.empty:
            fig_f1.add_bar(row=1, col=2, name="CamemBERT F1",
                x=df_cb["trial"].astype(str), y=df_cb["f1"],
                marker_color=CB_COLOR, showlegend=True,
                text=df_cb["f1"].round(4), textposition="outside")

        apply_theme(fig_f1)



        fig_f1.update_layout(height=360, title_text="F1-Score par Trial")
        fig_f1.update_yaxes(gridcolor="#1e2d4a")
        fig_f1.update_xaxes(title_text="Trial")
        st.plotly_chart(fig_f1, use_container_width=True)

        # ── LR vs F1 scatter ──
        st.markdown('<div class="section-label">Learning Rate vs F1 · Espace des Hyperparamètres</div>', unsafe_allow_html=True)
        fig_lr = go.Figure()
        if not df_db.empty:
            fig_lr.add_scatter(
                x=df_db["lr"], y=df_db["f1"],
                mode="markers+text",
                text=["T" + str(t) for t in df_db["trial"]],
                textposition="top center",
                marker=dict(color=DB_COLOR, size=12, symbol="circle"),
                name="DistilBERT",
            )
        if not df_cb.empty:
            fig_lr.add_scatter(
                x=df_cb["lr"], y=df_cb["f1"],
                mode="markers+text",
                text=["T" + str(t) for t in df_cb["trial"]],
                textposition="top center",
                marker=dict(color=CB_COLOR, size=12, symbol="diamond"),
                name="CamemBERT",
            )
        apply_theme(fig_lr)

        fig_lr.update_layout(
            title="Learning Rate vs F1 (Random Search)",
            xaxis_title="Learning Rate (log scale)",
            yaxis_title="F1-Score",
            height=360,
        )
        fig_lr.update_xaxes(type="log")
        st.plotly_chart(fig_lr, use_container_width=True)

        # ── Convergence ──
        st.markdown('<div class="section-label">Convergence · Eval Loss du Meilleur Trial</div>', unsafe_allow_html=True)
        fig_conv_plotly = go.Figure()
        has_conv = False
        if not conv_db.empty:
            fig_conv_plotly.add_scatter(
                x=conv_db["step"], y=conv_db["eval_loss"],
                mode="lines+markers", name="DistilBERT (meilleur trial)",
                line=dict(color=DB_COLOR, width=2, dash="dash"),
                marker=dict(size=6),
            )
            has_conv = True
        if not conv_cb.empty:
            fig_conv_plotly.add_scatter(
                x=conv_cb["step"], y=conv_cb["eval_loss"],
                mode="lines+markers", name="CamemBERT (meilleur trial)",
                line=dict(color=CB_COLOR, width=2),
                marker=dict(size=6),
            )
            has_conv = True

        if has_conv:
            apply_theme(fig_conv_plotly)

            fig_conv_plotly.update_layout(
                title="Courbes de Convergence · Eval Loss vs Steps",
                xaxis_title="Steps",
                yaxis_title="Eval Loss",
                height=340,
            )
            st.plotly_chart(fig_conv_plotly, use_container_width=True)
        else:
            img_conv = safe_load_image("convergence.png")
            if img_conv:
                st.image(img_conv, caption="Convergence (figure pipeline)", use_container_width=True)
            else:
                st.warning("Historique de convergence non disponible dans les JSON.")

        # ── Tableaux détaillés ──
        st.markdown('<div class="section-label">Tableaux Détaillés · Hyperparamètres et Scores</div>', unsafe_allow_html=True)
        col_t1, col_t2 = st.columns(2)
        if not df_db.empty:
            with col_t1:
                st.markdown(f"<span class='tag-db'>DistilBERT</span>", unsafe_allow_html=True)
                st.dataframe(df_db.style.highlight_max(subset=["f1","accuracy"], color="#ffe0e0"), use_container_width=True)
        if not df_cb.empty:
            with col_t2:
                st.markdown(f"<span class='tag-cb'>CamemBERT</span>", unsafe_allow_html=True)
                st.dataframe(df_cb.style.highlight_max(subset=["f1","accuracy"], color="#e0f5f4"), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 · LOSS LANDSCAPE
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-label">Loss Landscape 1D · Figures du Pipeline</div>', unsafe_allow_html=True)

    col_ll, col_rs = st.columns(2)
    with col_ll:
        img_ll = safe_load_image("loss_landscape.png")
        if img_ll:
            st.image(img_ll, caption="Loss Landscape · DistilBERT vs CamemBERT", use_container_width=True)
        else:
            st.warning("loss_landscape.png non trouvé dans ./figures/")

    with col_rs:
        img_rs = safe_load_image("random_search.png")
        if img_rs:
            st.image(img_rs, caption="Distribution F1 · Random Search (boxplot)", use_container_width=True)
        else:
            st.warning("random_search.png non trouvé dans ./figures/")

    st.markdown("---")
    st.markdown('<div class="section-label">Interprétation · Sharpness et Généralisation</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns(2)
    col_i1.markdown("""
    **Formule Sharpness** :
    ```
    S = 1/N · Σ |L(θ + α·d) − L(θ₀)|
    ```
    - `θ₀` : paramètres après entraînement  
    - `d` : direction aléatoire normalisée  
    - `α ∈ [-0.05, +0.05]` : intensité perturbation  
    - `n = 12` points d'évaluation  
    """)
    col_i2.markdown("""
    **Interprétation** :
    - **Sharpness bas** → minimum plat → meilleure généralisation
    - **CamemBERT** : minimum plus plat grâce à l'alignement linguistique natif
    - **DistilBERT** : courbure plus prononcée → overfitting possible sur bruit de transfert
    - **label_smoothing = 0.1** favorise les minima plats (codé dans CONFIG)
    """)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 · COMPARAISON FINALE
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-label">Résumé Comparatif P03 · Transfert Cross-Lingue</div>', unsafe_allow_html=True)

    if not df_db.empty and not df_cb.empty:
        best_f1_db = df_db["f1"].max()
        best_f1_cb = df_cb["f1"].max()
        mean_f1_db = df_db["f1"].mean()
        mean_f1_cb = df_cb["f1"].mean()
        delta       = best_f1_cb - best_f1_db

        # ── Bar chart comparatif ──
        summary_data = pd.DataFrame({
            "Métrique":  ["F1 Max", "F1 Moyen", "Accuracy Max"],
            "DistilBERT": [best_f1_db, mean_f1_db, df_db["accuracy"].max()],
            "CamemBERT":  [best_f1_cb, mean_f1_cb, df_cb["accuracy"].max()],
        })
        fig_summary = go.Figure()
        fig_summary.add_bar(name="DistilBERT", x=summary_data["Métrique"], y=summary_data["DistilBERT"],
            marker_color=DB_COLOR, text=summary_data["DistilBERT"].round(4), textposition="outside")
        fig_summary.add_bar(name="CamemBERT",  x=summary_data["Métrique"], y=summary_data["CamemBERT"],
            marker_color=CB_COLOR, text=summary_data["CamemBERT"].round(4), textposition="outside")
        apply_theme(fig_summary)

        fig_summary.update_layout(
            title="Comparaison Finale · Métriques Clés",
            barmode="group",
            height=380,
        )
        fig_summary.update_yaxes(range=[0.6, 1.0], gridcolor="#1e2d4a")
        st.plotly_chart(fig_summary, use_container_width=True)

        # ── Conclusion ──
        winner = "CamemBERT" if delta > 0 else "DistilBERT"
        winner_color = CB_COLOR if delta > 0 else DB_COLOR
        st.markdown(f"""
        <div style='background: #f5f7fa; border: 1px solid #dde3ed; border-left: 4px solid {winner_color}; 
             border-radius: 8px; padding: 20px 24px; margin-top: 16px;'>
          <div style='font-family: JetBrains Mono, monospace; font-size: 0.65rem; color: #5a6a8a; letter-spacing: 3px; margin-bottom: 8px;'>
            RÉSUMÉ P03 · TRANSFERT CROSS-LINGUE
          </div>
          <table style='width:100%; border-collapse: collapse; font-family: JetBrains Mono, monospace; font-size: 0.85rem;'>
            <tr style='border-bottom: 1px solid #1e2d4a;'>
              <td style='padding: 8px 0; color: #5a6a8a;'>Meilleur F1 DistilBERT (EN→FR)</td>
              <td style='color: {DB_COLOR}; font-weight: 700; text-align:right;'>{best_f1_db:.4f}</td>
            </tr>
            <tr style='border-bottom: 1px solid #1e2d4a;'>
              <td style='padding: 8px 0; color: #5a6a8a;'>Meilleur F1 CamemBERT (FR natif)</td>
              <td style='color: {CB_COLOR}; font-weight: 700; text-align:right;'>{best_f1_cb:.4f}</td>
            </tr>
            <tr style='border-bottom: 1px solid #1e2d4a;'>
              <td style='padding: 8px 0; color: #5a6a8a;'>Écart Δ F1</td>
              <td style='color: {GOLD}; font-weight: 700; text-align:right;'>{delta:+.4f}</td>
            </tr>
          </table>
          <div style='margin-top: 14px; color: {winner_color}; font-size: 0.85rem;'>
            {'✦ CamemBERT (natif FR) surpasse le transfert anglais → alignement linguistique confirmé (P03).' 
             if delta > 0 else 
             '✦ DistilBERT (EN→FR) reste compétitif malgré la barrière linguistique → transfert efficace.'}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Distribution F1 (box plot interactif) ──
        st.markdown("---")
        st.markdown('<div class="section-label">Distribution F1 · Tous Trials (Box Plot Interactif)</div>', unsafe_allow_html=True)
        fig_box = go.Figure()
        fig_box.add_box(y=df_db["f1"].tolist(), name="DistilBERT", marker_color=DB_COLOR,
                        boxpoints="all", jitter=0.3, pointpos=-1.8)
        fig_box.add_box(y=df_cb["f1"].tolist(), name="CamemBERT", marker_color=CB_COLOR,
                        boxpoints="all", jitter=0.3, pointpos=-1.8)
        apply_theme(fig_box)

        fig_box.update_layout(
            title="Distribution F1-Score · Ensemble des Trials",
            yaxis_title="F1-Score",
            height=380,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.error("Données JSON non chargées. Vérifiez que ./runs/ contient les fichiers random_search_*.json")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-family: JetBrains Mono, monospace; font-size: 0.6rem; 
            color: #aab4c8; letter-spacing: 3px; padding: 10px 0;'>
  GROUPE G11 · PIPELINE NLP · ALLOCINÉ D05 · SEED 42 · TRANSFERT CROSS-LINGUE P03
</div>
""", unsafe_allow_html=True)