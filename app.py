"""
Application Streamlit — Prévision des Prix de l'Or au Maroc
Auteur : Djibril SALL — ENSAM
Encadrant : Pr. Tawfik Masrour
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import os

# ─── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Prix de l'Or au Maroc",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Thème Plotly global — FIX lisibilité textes ──────────────────────────────
pio.templates["gold_theme"] = go.layout.Template(
    layout=go.Layout(
        font=dict(color="#111827", family="sans-serif", size=13),
        xaxis=dict(
            tickfont=dict(color="#111827", size=12),
            title_font=dict(color="#111827", size=13),
        ),
        yaxis=dict(
            tickfont=dict(color="#111827", size=12),
            title_font=dict(color="#111827", size=13),
        ),
        legend=dict(font=dict(color="#111827", size=12)),
        annotationdefaults=dict(
            font=dict(color="#111827", size=12),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=4,
        ),
    )
)
pio.templates.default = "plotly_white+gold_theme"

# ─── Palette de couleurs ──────────────────────────────────────────────────────
COLORS = {
    "Ensemble_Weighted":    "#1E2761",
    "SARIMAX":              "#C9A84C",
    "ARIMA":                "#4F9CF9",
    "SARIMA":               "#64748B",
    "XGBoost":              "#F59E0B",
    "Hybrid_ARIMA_XGBoost": "#7C3AED",
    "LSTM_univariate":      "#EF4444",
    "Prophet":              "#22C55E",
    "Actual":               "#111827",
}

MODEL_LABELS = {
    "Ensemble_Weighted":    "Ensemble Pondéré ★",
    "SARIMAX":              "SARIMAX",
    "ARIMA":                "ARIMA",
    "SARIMA":               "SARIMA",
    "XGBoost":              "XGBoost",
    "Hybrid_ARIMA_XGBoost": "Hybride ARIMA+XGB",
    "LSTM_univariate":      "LSTM (univarié)",
    "Prophet":              "Prophet",
}

# ─── Chemins des données ──────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
TABLES_DIR    = os.path.join(RESULTS_DIR, "tables")
FORECASTS_DIR = os.path.join(RESULTS_DIR, "forecasts")

# ─── Chargement des données ───────────────────────────────────────────────────
@st.cache_data
def load_historical():
    gold = pd.read_csv(os.path.join(DATA_DIR, "gold_prices.csv"))
    gold.columns = [c.lower() for c in gold.columns]
    gold["date"] = pd.to_datetime(gold["date"])
    fx = pd.read_csv(os.path.join(DATA_DIR, "usd_mad.csv"))
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    fx["usd_mad"] = (
        fx["usd_mad"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace('"', "", regex=False)
        .astype(float)
    )
    df = gold.merge(fx, on="date", how="inner")
    df["gold_price_mad"] = df["gold_price_usd"] * df["usd_mad"]
    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data
def load_forecasts():
    path = os.path.join(FORECASTS_DIR, "future_forecasts_to_2027_12.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_intervals():
    path = os.path.join(FORECASTS_DIR, "forecast_intervals_2027_12.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_backtest_metrics():
    return pd.read_csv(os.path.join(TABLES_DIR, "model_ranking_backtest.csv"))

@st.cache_data
def load_rolling_cv():
    df = pd.read_csv(os.path.join(TABLES_DIR, "rolling_cv_by_fold.csv"))
    df["test_start"] = pd.to_datetime(df["test_start"])
    df["test_end"]   = pd.to_datetime(df["test_end"])
    return df

@st.cache_data
def load_bootstrap_ci():
    return pd.read_csv(os.path.join(TABLES_DIR, "metrics_bootstrap_ci.csv"))

@st.cache_data
def load_events():
    df = pd.read_csv(os.path.join(DATA_DIR, "moroccan_events.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_macro():
    path = os.path.join(DATA_DIR, "macro_indicators.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_sarimax_coefficients():
    path = os.path.join(TABLES_DIR, "sarimax_coefficients.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "param" in df.columns and "estimate" in df.columns:
        df = df.rename(columns={"param": "Paramètre", "estimate": "Estimation"})
    return df

# ─── Helper : applique le style de base sans clés conflictuelles ─────────────
def apply_base_style(fig, height=400, margin=None, xaxis_title=None, yaxis_title=None,
                     xaxis_extra=None, legend_extra=None, hovermode=None):
    """
    Applique plot_bgcolor, paper_bgcolor, font, axes de base.
    Toutes les clés potentiellement dupliquées sont passées en paramètres séparés.
    """
    if margin is None:
        margin = dict(l=0, r=0, t=10, b=0)

    x_cfg = dict(gridcolor="#F1F5F9",
                 tickfont=dict(color="#111827", size=12),
                 title_font=dict(color="#111827", size=13))
    if xaxis_title:
        x_cfg["title_text"] = xaxis_title
    if xaxis_extra:
        x_cfg.update(xaxis_extra)

    y_cfg = dict(gridcolor="#F1F5F9",
                 tickfont=dict(color="#111827", size=12),
                 title_font=dict(color="#111827", size=13))
    if yaxis_title:
        y_cfg["title_text"] = yaxis_title

    leg_cfg = dict(font=dict(color="#111827", size=12))
    if legend_extra:
        leg_cfg.update(legend_extra)

    layout_kwargs = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827", size=13),
        height=height,
        margin=margin,
        xaxis=x_cfg,
        yaxis=y_cfg,
        legend=leg_cfg,
    )
    if hovermode:
        layout_kwargs["hovermode"] = hovermode

    fig.update_layout(**layout_kwargs)


# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E2761 0%, #C9A84C 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #CADCFC; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .kpi-card {
        background: white; border-radius: 10px; padding: 1rem 1.2rem;
        border-left: 4px solid #C9A84C; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 0.5rem;
    }
    .kpi-value { font-size: 1.7rem; font-weight: 700; color: #1E2761; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #64748B; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.04em; }
    .kpi-sub   { font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #1E2761;
        border-bottom: 2px solid #C9A84C; padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .stSelectbox label, .stMultiSelect label { font-weight: 600; color: #1E2761; }
</style>
""", unsafe_allow_html=True)

# ─── En-tête principal ────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🥇 Prévision des Prix de l'Or au Maroc</h1>
    <p>Modélisation mensuelle 2000–2027 &nbsp;|&nbsp; Machine Learning & Économétrie &nbsp;|&nbsp;
       Djibril SALL — ENSAM &nbsp;|&nbsp; Encadrant : Pr. Tawfik Masrour</p>
</div>
""", unsafe_allow_html=True)

# ─── Chargement ───────────────────────────────────────────────────────────────
hist         = load_historical()
forecasts    = load_forecasts()
intervals    = load_intervals()
metrics      = load_backtest_metrics()
rolling      = load_rolling_cv()
bootstrap    = load_bootstrap_ci()
events       = load_events()
macro        = load_macro()
sarimax_coef = load_sarimax_coefficients()

MODEL_COLS = [c for c in forecasts.columns if c != "date"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:3rem;">🥇</div>
        <div style="font-weight:700; color:#1E2761; font-size:1rem;">Gold Forecast Maroc</div>
        <div style="font-size:0.8rem; color:#888;">ENSAM · Djibril SALL</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Paramètres")
    page = st.radio(
        "Navigation",
        ["📊 Tableau de bord", "🔮 Prévisions", "📈 Backtest", "🔬 Analyse", "ℹ️ À propos"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Modèles à afficher**")
    selected_models = st.multiselect(
        "Sélectionner",
        options=MODEL_COLS,
        default=[m for m in ["Ensemble_Weighted", "SARIMAX", "XGBoost"] if m in MODEL_COLS],
        format_func=lambda x: MODEL_LABELS.get(x, x),
        label_visibility="collapsed",
    )
    if not selected_models:
        selected_models = [MODEL_COLS[0]]
    st.markdown("---")
    st.markdown("**Période historique**")
    year_min = int(hist["date"].dt.year.min())
    year_max = int(hist["date"].dt.year.max())
    hist_range = st.slider("Années", year_min, year_max, (2015, year_max))
    st.markdown("---")
    st.caption("🏫 ENSAM Meknes — Recherche académique")
    st.caption("📌 Seed reproductible : 42")

# ─── Filtrage de l'historique ─────────────────────────────────────────────────
hist_filtered = hist[
    (hist["date"].dt.year >= hist_range[0]) &
    (hist["date"].dt.year <= hist_range[1])
].copy()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Tableau de bord":

    last_price   = hist["gold_price_mad"].iloc[-1]
    last_date    = hist["date"].iloc[-1].strftime("%b %Y")
    prev_price   = hist["gold_price_mad"].iloc[-2]
    delta_pct    = (last_price - prev_price) / prev_price * 100
    best_model_col = "Ensemble_Weighted" if "Ensemble_Weighted" in forecasts.columns else MODEL_COLS[0]
    fc_dec27     = forecasts[forecasts["date"] == "2027-12-01"][best_model_col].values
    fc_dec27_val = fc_dec27[0] if len(fc_dec27) else None
    non_baseline = metrics[~metrics["model"].str.startswith("Baseline")]
    best_rmse    = non_baseline.sort_values("rmse_mean").iloc[0] if len(non_baseline) > 0 else metrics.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Dernier prix observé</div>
            <div class="kpi-value">{last_price:,.0f} MAD</div>
            <div class="kpi-sub">{last_date} &nbsp;|&nbsp; {'▲' if delta_pct>0 else '▼'} {abs(delta_pct):.1f}% vs mois préc.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        if fc_dec27_val:
            st.markdown(f"""<div class="kpi-card" style="border-color:#4F9CF9">
                <div class="kpi-label">Prévision Déc. 2027 (Ensemble)</div>
                <div class="kpi-value">{fc_dec27_val:,.0f} MAD</div>
                <div class="kpi-sub">+{(fc_dec27_val/last_price-1)*100:.1f}% vs aujourd'hui</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-color:#22C55E">
            <div class="kpi-label">Meilleur modèle (RMSE)</div>
            <div class="kpi-value">{MODEL_LABELS.get(best_rmse['model'], best_rmse['model'])}</div>
            <div class="kpi-sub">RMSE {best_rmse['rmse_mean']:,.0f} MAD &nbsp;|&nbsp; MAPE {best_rmse['mape_mean']:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        nb_obs = len(hist)
        hausse = (hist["gold_price_mad"].iloc[-1] / hist["gold_price_mad"].iloc[0] - 1) * 100
        st.markdown(f"""<div class="kpi-card" style="border-color:#7C3AED">
            <div class="kpi-label">Série historique</div>
            <div class="kpi-value">{nb_obs} mois</div>
            <div class="kpi-sub">{hist["date"].dt.year.min()}–{hist["date"].dt.year.max()} &nbsp;|&nbsp; Hausse totale +{hausse:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Série historique — Prix de l\'or en MAD</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_filtered["date"], y=hist_filtered["gold_price_mad"],
        name="Prix observé (MAD)", line=dict(color=COLORS["Actual"], width=2),
        hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f} MAD</b><extra></extra>",
    ))
    for m in selected_models:
        if m in forecasts.columns:
            fig.add_trace(go.Scatter(
                x=forecasts["date"], y=forecasts[m],
                name=MODEL_LABELS.get(m, m),
                line=dict(color=COLORS.get(m, "#999"), width=2, dash="dash"),
                hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f} MAD</b><extra></extra>",
            ))
    if "SARIMAX" in selected_models and "upper" in intervals.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([intervals["date"], intervals["date"][::-1]]),
            y=pd.concat([intervals["upper"], intervals["lower"][::-1]]),
            fill="toself", fillcolor="rgba(201,168,76,0.12)",
            line=dict(color="rgba(201,168,76,0)"),
            name="IC 95% SARIMAX", showlegend=True, hoverinfo="skip",
        ))
    last_hist_ts = hist["date"].max().timestamp() * 1000
    fig.add_vline(x=last_hist_ts, line_dash="dot", line_color="#EF4444",
                  annotation_text="Dernier obs.", annotation_position="top right",
                  annotation_font_color="#111827", annotation_bgcolor="rgba(255,255,255,0.85)")
    apply_base_style(fig, height=420, hovermode="x unified",
                     xaxis_title="Date", yaxis_title="MAD / once troy",
                     legend_extra=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Or en USD vs Taux USD/MAD</div>', unsafe_allow_html=True)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=hist_filtered["date"], y=hist_filtered["gold_price_usd"],
                                  name="XAU/USD", line=dict(color="#C9A84C", width=1.5)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=hist_filtered["date"], y=hist_filtered["usd_mad"],
                                  name="USD/MAD", line=dict(color="#4F9CF9", width=1.5)), secondary_y=True)
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#111827", size=13),
            height=280, margin=dict(l=0, r=0, t=5, b=0),
            legend=dict(orientation="h", y=1.05, font=dict(color="#111827")),
        )
        fig2.update_yaxes(title_text="USD/once", secondary_y=False,
                          gridcolor="#F1F5F9",
                          tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
        fig2.update_yaxes(title_text="USD/MAD", secondary_y=True,
                          tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
        fig2.update_xaxes(tickfont=dict(color="#111827"), title_font=dict(color="#111827"),
                          gridcolor="#F1F5F9")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Distribution mensuelle des retours</div>', unsafe_allow_html=True)
        returns = hist_filtered["gold_price_mad"].pct_change().dropna() * 100
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=returns, nbinsx=40, marker_color="#1E2761",
                                    opacity=0.75, name="Rendements mensuels %"))
        fig3.add_vline(x=returns.mean(), line_dash="dash", line_color="#C9A84C",
                       annotation_text=f"Moy. {returns.mean():.2f}%",
                       annotation_font_color="#111827",
                       annotation_bgcolor="rgba(255,255,255,0.85)")
        apply_base_style(fig3, height=280, margin=dict(l=0, r=0, t=5, b=0),
                         xaxis_title="Rendement mensuel (%)", yaxis_title="Fréquence")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button("⬇ Exporter historique CSV",
                           data=hist.to_csv(index=False).encode("utf-8"),
                           file_name="gold_historique_mad.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉVISIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prévisions":
    st.markdown('<div class="section-title">Prévisions futures 2026–2027 — Tous les modèles</div>', unsafe_allow_html=True)

    hist_recent = hist[hist["date"] >= "2022-01-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_recent["date"], y=hist_recent["gold_price_mad"],
                             name="Historique observé", line=dict(color=COLORS["Actual"], width=2.5),
                             hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f} MAD</b><extra></extra>"))
    for m in selected_models:
        if m in forecasts.columns:
            fig.add_trace(go.Scatter(x=forecasts["date"], y=forecasts[m],
                                     name=MODEL_LABELS.get(m, m),
                                     line=dict(color=COLORS.get(m, "#999"), width=2.5),
                                     hovertemplate=f"<b>{MODEL_LABELS.get(m,m)}</b><br>%{{x|%b %Y}}<br>%{{y:,.0f}} MAD<extra></extra>"))
    if "upper" in intervals.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([intervals["date"], intervals["date"][::-1]]),
            y=pd.concat([intervals["upper"], intervals["lower"][::-1]]),
            fill="toself", fillcolor="rgba(201,168,76,0.13)",
            line=dict(color="rgba(201,168,76,0)"), name="IC 95% (SARIMAX)", hoverinfo="skip"))
    last_hist_ts = hist["date"].max().timestamp() * 1000
    fig.add_vline(x=last_hist_ts, line_dash="dot", line_color="#EF4444", line_width=1.5,
                  annotation_text="Fin historique", annotation_position="top left",
                  annotation_font_color="#111827", annotation_bgcolor="rgba(255,255,255,0.85)")
    apply_base_style(fig, height=460, hovermode="x unified",
                     xaxis_title="Date", yaxis_title="MAD / once troy",
                     legend_extra=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Tableau des prévisions (MAD / once troy)</div>', unsafe_allow_html=True)
    fc_display = forecasts.copy()
    fc_display["date"] = fc_display["date"].dt.strftime("%b %Y")
    cols_show = ["date"] + [c for c in selected_models if c in fc_display.columns]
    fc_show = fc_display[cols_show].copy()
    fc_show.columns = ["Date"] + [MODEL_LABELS.get(c, c) for c in cols_show[1:]]
    for col in fc_show.columns[1:]:
        fc_show[col] = fc_show[col].apply(lambda x: f"{x:,.0f}")
    st.dataframe(fc_show, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Variation prévue par modèle (début → fin horizon)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        start_row  = forecasts[forecasts["date"] == forecasts["date"].min()].iloc[0]
        end_row    = forecasts[forecasts["date"] == forecasts["date"].max()].iloc[0]
        variations = {}
        for m in MODEL_COLS:
            if m in start_row and m in end_row and start_row[m] > 0:
                variations[MODEL_LABELS.get(m, m)] = (end_row[m] / start_row[m] - 1) * 100
        bar_colors = [COLORS.get(k, "#999") for k in MODEL_COLS if MODEL_LABELS.get(k, k) in variations]
        fig_var = go.Figure(go.Bar(
            x=list(variations.values()), y=list(variations.keys()),
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in variations.values()],
            textposition="outside",
            textfont=dict(color="#111827", size=12),
        ))
        apply_base_style(fig_var, height=280, margin=dict(l=0, r=70, t=5, b=0),
                         xaxis_title="Variation totale (%)")
        st.plotly_chart(fig_var, use_container_width=True)
    with col2:
        st.markdown("**Lecture rapide**")
        for m, v in sorted(variations.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{m}** : {'▲' if v>0 else '▼'} {abs(v):.1f}%")

    st.markdown("---")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button("⬇ Exporter prévisions CSV",
                           data=forecasts.to_csv(index=False).encode("utf-8"),
                           file_name="gold_previsions_2027.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Backtest":
    st.markdown('<div class="section-title">Résultats du backtest glissant — Rolling CV</div>', unsafe_allow_html=True)

    has_baseline = metrics["model"].str.startswith("Baseline").any()
    m_display = metrics[~metrics["model"].str.startswith("Baseline")].copy() if has_baseline else metrics.copy()
    m_display["Modèle"]         = m_display["model"].map(lambda x: MODEL_LABELS.get(x, x))
    m_display["RMSE moy."]      = m_display["rmse_mean"].apply(lambda x: f"{x:,.0f}")
    m_display["MAE moy."]       = m_display["mae_mean"].apply(lambda x: f"{x:,.0f}")
    m_display["MAPE %"]         = m_display["mape_mean"].apply(lambda x: f"{x:.2f}%")
    m_display["Dir. Acc. %"]    = m_display["directional_accuracy_mean"].apply(lambda x: f"{x:.1f}%")
    m_display["Biais moy. MAD"] = m_display["bias_mean"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(m_display[["Modèle","RMSE moy.","MAE moy.","MAPE %","Dir. Acc. %","Biais moy. MAD"]].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">RMSE moyen par modèle</div>', unsafe_allow_html=True)
        fig_rmse = go.Figure(go.Bar(
            x=m_display["rmse_mean"], y=m_display["Modèle"],
            orientation="h",
            marker_color=[COLORS.get(r, "#999") for r in m_display["model"]],
            error_x=dict(type="data", array=m_display["rmse_std"].tolist(),
                         visible=True, color="#64748B"),
            text=m_display["rmse_mean"].apply(lambda x: f"{x:,.0f}"),
            textposition="outside",
            textfont=dict(color="#111827", size=12),
        ))
        apply_base_style(fig_rmse, height=300, margin=dict(l=0, r=70, t=5, b=0),
                         xaxis_title="RMSE (MAD)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Précision directionnelle</div>', unsafe_allow_html=True)
        fig_da = go.Figure(go.Bar(
            x=m_display["directional_accuracy_mean"], y=m_display["Modèle"],
            orientation="h",
            marker_color=[COLORS.get(r, "#999") for r in m_display["model"]],
            text=m_display["directional_accuracy_mean"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="#111827", size=12),
        ))
        fig_da.add_vline(x=50, line_dash="dash", line_color="#EF4444",
                         annotation_text="Aléatoire (50%)",
                         annotation_font_color="#111827",
                         annotation_bgcolor="rgba(255,255,255,0.85)")
        apply_base_style(fig_da, height=300, margin=dict(l=0, r=70, t=5, b=0),
                         xaxis_title="Dir. Acc. (%)",
                         xaxis_extra=dict(range=[0, 80]))
        st.plotly_chart(fig_da, use_container_width=True)

    st.markdown('<div class="section-title">Intervalles de confiance Bootstrap R² (95%) — OOS poolé</div>', unsafe_allow_html=True)
    r2_ci = bootstrap[bootstrap["metric"] == "r2"].copy()
    r2_ci["label"] = r2_ci["model"].map(lambda x: MODEL_LABELS.get(x, x))
    r2_ci = r2_ci.sort_values("point", ascending=False)
    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=r2_ci["point"], y=r2_ci["label"],
        mode="markers", marker=dict(size=10, color="#1E2761"),
        error_x=dict(type="data",
                     array=(r2_ci["ci_high"] - r2_ci["point"]).tolist(),
                     arrayminus=(r2_ci["point"] - r2_ci["ci_low"]).tolist(),
                     color="#C9A84C", thickness=2.5, width=6),
        name="R² [IC 95%]",
        hovertemplate="<b>%{y}</b><br>R² = %{x:.3f}<extra></extra>",
    ))
    apply_base_style(fig_ci, height=280, margin=dict(l=0, r=0, t=5, b=0),
                     xaxis_title="R² (OOS poolé)", xaxis_extra=dict(range=[0.6, 1.0]))
    st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown('<div class="section-title">Évolution du RMSE par fold (stabilité des modèles)</div>', unsafe_allow_html=True)
    has_baseline_roll = rolling["model"].str.startswith("Baseline").any()
    roll_clean = rolling[~rolling["model"].str.startswith("Baseline")] if has_baseline_roll else rolling
    default_fold_models = [m for m in ["SARIMAX", "XGBoost", "Ensemble_Weighted"]
                           if m in roll_clean["model"].unique()]
    if not default_fold_models:
        default_fold_models = list(roll_clean["model"].unique())[:3]
    models_fold = st.multiselect("Modèles à afficher (folds)",
                                 options=roll_clean["model"].unique().tolist(),
                                 default=default_fold_models,
                                 format_func=lambda x: MODEL_LABELS.get(x, x))
    fig_fold = go.Figure()
    for m in models_fold:
        sub = roll_clean[roll_clean["model"] == m].sort_values("test_start")
        fig_fold.add_trace(go.Scatter(
            x=sub["test_start"], y=sub["rmse"],
            name=MODEL_LABELS.get(m, m), mode="lines+markers",
            marker=dict(size=7), line=dict(color=COLORS.get(m, "#999"), width=2),
            hovertemplate=f"<b>{MODEL_LABELS.get(m,m)}</b><br>%{{x|%Y}}<br>RMSE = %{{y:,.0f}} MAD<extra></extra>",
        ))
    apply_base_style(fig_fold, height=320, margin=dict(l=0, r=0, t=5, b=0),
                     xaxis_title="Début du fold de test", yaxis_title="RMSE (MAD)",
                     hovermode="x unified",
                     legend_extra=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_fold, use_container_width=True)

    st.markdown("---")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button("⬇ Exporter métriques CSV",
                           data=metrics.to_csv(index=False).encode("utf-8"),
                           file_name="gold_metrics_backtest.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Analyse":
    tab1, tab2, tab3 = st.tabs(["📅 Événements culturels", "🏦 Indicateurs macro", "🔍 Coefficients SARIMAX"])

    # ── Tab 1 : Événements ────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Impact des événements socio-culturels marocains</div>', unsafe_allow_html=True)
        event_merged = hist.merge(events, on="date", how="inner")
        event_merged = event_merged[event_merged["date"].dt.year >= hist_range[0]]
        event_cols   = ["ramadan", "eid_alfitr", "eid_aladha", "wedding_season", "mre_season"]
        event_labels = {"ramadan":"Ramadan","eid_alfitr":"Aïd Al-Fitr","eid_aladha":"Aïd Al-Adha",
                        "wedding_season":"Saison Mariages","mre_season":"Saison MRE"}
        available_events = [e for e in event_cols if e in event_merged.columns]
        selected_event = st.selectbox("Sélectionner un événement", available_events,
                                      format_func=lambda x: event_labels.get(x, x))
        fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ev.add_trace(go.Scatter(x=event_merged["date"], y=event_merged["gold_price_mad"],
                                    name="Prix or (MAD)", line=dict(color=COLORS["Actual"], width=2)),
                         secondary_y=False)
        fig_ev.add_trace(go.Scatter(x=event_merged["date"], y=event_merged[selected_event],
                                    name=event_labels.get(selected_event, selected_event),
                                    line=dict(color="#C9A84C", width=1.5),
                                    fill="tozeroy", fillcolor="rgba(201,168,76,0.15)"),
                         secondary_y=True)
        fig_ev.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#111827", size=13),
            height=360, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05, font=dict(color="#111827")),
        )
        fig_ev.update_yaxes(title_text="MAD / once", secondary_y=False,
                            gridcolor="#F1F5F9",
                            tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
        fig_ev.update_yaxes(title_text="Intensité [0,1]", secondary_y=True,
                            tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
        fig_ev.update_xaxes(tickfont=dict(color="#111827"), gridcolor="#F1F5F9")
        st.plotly_chart(fig_ev, use_container_width=True)

        st.markdown('<div class="section-title">Corrélations avec le prix de l\'or (MAD)</div>', unsafe_allow_html=True)
        corrs = {event_labels.get(e, e): float(event_merged[e].corr(event_merged["gold_price_mad"]))
                 for e in available_events}
        fig_corr = go.Figure(go.Bar(
            x=list(corrs.values()), y=list(corrs.keys()), orientation="h",
            marker_color=["#22C55E" if v > 0 else "#EF4444" for v in corrs.values()],
            text=[f"{v:.3f}" for v in corrs.values()],
            textposition="outside",
            textfont=dict(color="#111827", size=12),
        ))
        apply_base_style(fig_corr, height=220, margin=dict(l=0, r=70, t=5, b=0),
                         xaxis_title="Corrélation de Pearson")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Tab 2 : Macro ─────────────────────────────────────────────────────────
    with tab2:
        if macro is None:
            st.info("Le fichier `macro_indicators.csv` n'est pas disponible.")
        else:
            st.markdown('<div class="section-title">Indicateurs macroéconomiques vs Prix de l\'or</div>', unsafe_allow_html=True)
            macro_merged = hist.merge(macro, on="date", how="inner")
            macro_merged = macro_merged[macro_merged["date"].dt.year >= hist_range[0]]
            macro_cols_all = ["oil_brent_usd","dxy_index","fed_funds_rate","inflation_morocco","policy_rate_bam"]
            macro_labels = {"oil_brent_usd":"Pétrole Brent (USD/baril)","dxy_index":"Indice DXY",
                            "fed_funds_rate":"Taux Fed Funds (%)","inflation_morocco":"Inflation Maroc (%)",
                            "policy_rate_bam":"Taux BAM (%)"}
            available_macro = [c for c in macro_cols_all if c in macro_merged.columns]
            selected_macro = st.selectbox("Indicateur macro", available_macro,
                                          format_func=lambda x: macro_labels.get(x, x))
            fig_mac = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mac.add_trace(go.Scatter(x=macro_merged["date"], y=macro_merged["gold_price_mad"],
                                         name="Prix or (MAD)", line=dict(color=COLORS["Actual"], width=2)),
                              secondary_y=False)
            fig_mac.add_trace(go.Scatter(x=macro_merged["date"], y=macro_merged[selected_macro],
                                         name=macro_labels.get(selected_macro, selected_macro),
                                         line=dict(color="#22C55E", width=1.5)),
                              secondary_y=True)
            fig_mac.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="#111827", size=13),
                height=360, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.05, font=dict(color="#111827")),
            )
            fig_mac.update_yaxes(title_text="MAD / once", secondary_y=False,
                                 gridcolor="#F1F5F9",
                                 tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
            fig_mac.update_yaxes(title_text=macro_labels.get(selected_macro, selected_macro),
                                 secondary_y=True,
                                 tickfont=dict(color="#111827"), title_font=dict(color="#111827"))
            fig_mac.update_xaxes(tickfont=dict(color="#111827"), gridcolor="#F1F5F9")
            st.plotly_chart(fig_mac, use_container_width=True)

            st.markdown('<div class="section-title">Matrice de corrélation</div>', unsafe_allow_html=True)
            corr_cols = ["gold_price_mad"] + available_macro
            corr_matrix = macro_merged[corr_cols].corr()
            corr_labels = ["Or MAD"] + [macro_labels.get(c, c) for c in available_macro]
            fig_heat = go.Figure(go.Heatmap(
                z=corr_matrix.values, x=corr_labels, y=corr_labels,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
                textfont=dict(color="#111827", size=12),
                hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.3f}<extra></extra>",
            ))
            fig_heat.update_layout(
                height=320, margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="white", font=dict(color="#111827"),
                xaxis=dict(tickfont=dict(color="#111827")),
                yaxis=dict(tickfont=dict(color="#111827")),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 3 : Coefficients SARIMAX ─────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Coefficients estimés du modèle SARIMAX</div>', unsafe_allow_html=True)
        if sarimax_coef is not None and "Estimation" in sarimax_coef.columns:
            coef_plot = sarimax_coef[sarimax_coef["Paramètre"] != "σ²"].copy()
            fig_coef = go.Figure(go.Bar(
                y=coef_plot["Paramètre"], x=coef_plot["Estimation"], orientation="h",
                marker_color=["#22C55E" if v > 0 else "#EF4444" for v in coef_plot["Estimation"]],
                text=[f"{v:.3f}" for v in coef_plot["Estimation"]],
                textposition="outside",
                textfont=dict(color="#111827", size=12),
            ))
            apply_base_style(fig_coef, height=max(300, len(coef_plot)*35),
                             margin=dict(l=0, r=90, t=5, b=0),
                             xaxis_title="Coefficient estimé")
            st.plotly_chart(fig_coef, use_container_width=True)
            st.dataframe(sarimax_coef, use_container_width=True, hide_index=True)
        else:
            st.info("Affichage des coefficients issus de l'entraînement final.")
            coef_data = {
                "Paramètre": ["Constante","usd_mad","ramadan","eid_aladha","wedding_season",
                              "eid_alfitr","mre_season","inflation_morocco","fed_funds_rate",
                              "policy_rate_bam","oil_brent_usd","dxy_index","MA(1)","σ²"],
                "Estimation": [62.90,1048.49,21.99,70.68,48.10,28.10,-37.41,
                               -2.09,-0.29,1.41,-17.65,-79.15,0.17,113332.51],
                "Interprétation": [
                    "Intercepte",
                    "↑ 1 MAD de dépréciation → +1 048 MAD/once (dominant)",
                    "Achats bijoux avant Aïd","Pic demande Aïd Al-Adha","Saison mariages",
                    "Cadeaux de fin Ramadan","Substitution MRE (négatif)",
                    "Pression coûts importation","Coût d'opportunité de l'or",
                    "Politique monétaire BAM","Rotation risk-on matières",
                    "Dollar fort → or moins cher en USD","Correction résiduelle MA(1)","Variance des résidus",
                ],
            }
            coef_df   = pd.DataFrame(coef_data)
            coef_plot = coef_df[coef_df["Paramètre"] != "σ²"]
            fig_coef  = go.Figure(go.Bar(
                y=coef_plot["Paramètre"], x=coef_plot["Estimation"], orientation="h",
                marker_color=["#22C55E" if v > 0 else "#EF4444" for v in coef_plot["Estimation"]],
                text=[f"{v:.2f}" for v in coef_plot["Estimation"]],
                textposition="outside",
                textfont=dict(color="#111827", size=12),
            ))
            apply_base_style(fig_coef, height=380, margin=dict(l=0, r=90, t=5, b=0),
                             xaxis_title="Coefficient estimé (MAD)")
            st.plotly_chart(fig_coef, use_container_width=True)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos":
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div style="background:#1E2761;border-radius:12px;padding:1.5rem;color:white;text-align:center">
            <div style="font-size:3rem">🥇</div>
            <div style="font-size:1.2rem;font-weight:700;margin-top:0.5rem">Djibril SALL</div>
            <div style="color:#CADCFC;font-size:0.9rem">Élève-Ingénieur</div>
            <div style="color:#C9A84C;font-size:0.85rem;margin-top:0.5rem">ENSAM Meknes</div>
            <hr style="border-color:#C9A84C;margin:0.8rem 0">
            <div style="color:#CADCFC;font-size:0.82rem">Encadrant :</div>
            <div style="font-size:0.95rem;font-weight:600">Pr. Tawfik Masrour</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### 🎯 Objectif du projet")
        st.write("""Pipeline académique complet de prévision mensuelle des prix de l'or en Dirham marocain (MAD). Combinaison de plusieurs approches :
- **Économétrie classique** : ARIMA, SARIMA, SARIMAX
- **Machine Learning** : XGBoost, modèle hybride ARIMA+XGBoost
- **Deep Learning** : LSTM univarié
- **Modèles probabilistes** : Prophet, Ensemble pondéré""")
        st.markdown("### 📁 Données utilisées")
        st.markdown("""| Fichier | Description |
|---------|-------------|
| `gold_prices.csv` | Cours XAU/USD (World Gold Council) |
| `usd_mad.csv` | Taux USD/MAD (Bank Al-Maghrib) |
| `moroccan_events.csv` | Événements culturels mensuels |
| `macro_indicators.csv` | BAM, DXY, Brent, Fed, CPI Maroc |""")
        st.markdown("### ⚙️ Pipeline technique")
        st.write("""- **307 mois** de données (août 2000 – février 2026)
- **Rolling CV** avec validation croisée temporelle (expanding window)
- **Anti-leakage strict** : prétraitement par fold, lags décalés ≥ 1
- **Graine reproductible** : 42 (NumPy, Python, TensorFlow)
- **Trend damping** : φ = 0.92, plafond ±4%/mois""")

    st.markdown("---")
    st.markdown("### 📊 Résultats clés — Classement des modèles")
    m_about = metrics[~metrics["model"].str.startswith("Baseline")].copy() \
        if metrics["model"].str.startswith("Baseline").any() else metrics.copy()
    m_about["Modèle"]      = m_about["model"].map(lambda x: MODEL_LABELS.get(x, x))
    m_about["RMSE moy."]   = m_about["rmse_mean"].apply(lambda x: f"{x:,.0f} MAD")
    m_about["MAPE %"]      = m_about["mape_mean"].apply(lambda x: f"{x:.2f}%")
    m_about["Dir. Acc. %"] = m_about["directional_accuracy_mean"].apply(lambda x: f"{x:.1f}%")
    m_about["Biais"]       = m_about["bias_mean"].apply(lambda x: f"{x:,.0f} MAD")
    st.dataframe(m_about[["Modèle","RMSE moy.","MAPE %","Dir. Acc. %","Biais"]],
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📂 Fichiers chargés avec succès")
    files_status = [
        ("data/gold_prices.csv",                              len(hist)),
        ("data/usd_mad.csv",                                  len(hist)),
        ("data/moroccan_events.csv",                          len(events)),
        ("data/macro_indicators.csv",                         len(macro) if macro is not None else "—"),
        ("results/forecasts/future_forecasts_to_2027_12.csv", len(forecasts)),
        ("results/forecasts/forecast_intervals_2027_12.csv",  len(intervals)),
        ("results/tables/model_ranking_backtest.csv",         len(metrics)),
        ("results/tables/rolling_cv_by_fold.csv",             len(rolling)),
        ("results/tables/metrics_bootstrap_ci.csv",           len(bootstrap)),
        ("results/tables/sarimax_coefficients.csv",
         len(sarimax_coef) if sarimax_coef is not None else "—"),
    ]
    for fname, nrows in files_status:
        icon = "✅" if nrows != "—" else "⚠️"
        st.markdown(f"{icon} `{fname}` — {nrows} lignes" if nrows != "—"
                    else f"{icon} `{fname}` — non trouvé")