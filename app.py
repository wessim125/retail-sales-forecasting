from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Palette (inspirée de l'identité Rossmann : rouge + neutres froids)
# ─────────────────────────────────────────────────────────────
INK = "#1B1F2A"        # texte principal
MUTED = "#6B7385"      # texte secondaire
LINE = "#E3E6EC"       # bordures / grilles
PAPER = "#F4F5F7"      # fond de page
CARD = "#FFFFFF"       # fond des cartes
RED = "#C8102E"        # accent principal (ventes)
RED_SOFT = "#FBE9EC"
TEAL = "#0F6B6B"       # accent secondaire (modèles / gains)
TEAL_SOFT = "#E3F1F0"
AMBER = "#D98E04"      # tendance / moyenne mobile

DAY_LABELS = {1: "Lun", 2: "Mar", 3: "Mer", 4: "Jeu", 5: "Ven", 6: "Sam", 7: "Dim"}


# ─────────────────────────────────────────────────────────────
# Données
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "processed" / "train_features.parquet"
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").copy()


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).resolve().parent / "reports" / "model_comparison.csv")


def currency(value: float) -> str:
    return f"€{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


# ─────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rossmann · Prévision des ventes", layout="wide")

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Manrope', system-ui, sans-serif;
            color: {INK};
        }}
        .stApp {{ background: {PAPER}; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
        header[data-testid="stHeader"] {{ background: transparent; }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {CARD};
            border-right: 1px solid {LINE};
        }}

        /* En-tête */
        .hero {{
            border-left: 6px solid {RED};
            padding: 0.2rem 0 0.2rem 1.1rem;
            margin-bottom: 1.6rem;
        }}
        .hero h1 {{
            font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em;
            margin: 0; color: {INK}; line-height: 1.15;
        }}
        .hero p {{ color: {MUTED}; margin: 0.35rem 0 0; font-size: 0.98rem; max-width: 62ch; }}

        /* Cartes KPI */
        .kpi {{
            background: {CARD};
            border: 1px solid {LINE};
            border-radius: 12px;
            padding: 1.05rem 1.2rem;
            height: 100%;
        }}
        .kpi-label {{ color: {MUTED}; font-size: 0.85rem; font-weight: 600; }}
        .kpi-value {{
            color: {INK}; font-size: 1.85rem; font-weight: 800;
            letter-spacing: -0.02em; margin-top: 0.2rem; line-height: 1.2;
        }}
        .kpi-note {{ color: {MUTED}; font-size: 0.82rem; margin-top: 0.4rem; }}
        .kpi.primary {{ border-top: 4px solid {RED}; }}
        .kpi.good    {{ border-top: 4px solid {TEAL}; }}
        .chip {{
            display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
            font-size: 0.78rem; font-weight: 700;
        }}
        .chip.red  {{ background: {RED_SOFT};  color: {RED}; }}
        .chip.teal {{ background: {TEAL_SOFT}; color: {TEAL}; }}

        /* Panneaux (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {CARD};
            border: 1px solid {LINE} !important;
            border-radius: 12px;
            padding: 0.4rem 0.6rem;
        }}
        .panel-title {{ font-size: 1.05rem; font-weight: 700; color: {INK}; margin-bottom: 0.1rem; }}
        .panel-sub   {{ font-size: 0.85rem; color: {MUTED}; margin-bottom: 0.6rem; }}

        /* Onglets */
        button[data-baseweb="tab"] {{ font-weight: 600; color: {MUTED}; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: {RED}; }}
        div[data-baseweb="tab-highlight"] {{ background-color: {RED}; }}

        /* Métriques natives */
        div[data-testid="stMetric"] {{
            background: {PAPER}; border-radius: 10px; padding: 0.7rem 0.9rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(chart: alt.Chart, height: int = 300) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=LINE,
            domainColor=LINE, tickColor=LINE, labelFont="Manrope", titleFont="Manrope",
        )
        .configure_legend(labelColor=INK, titleColor=MUTED, orient="top", labelFont="Manrope")
    )


def kpi(label: str, value: str, note: str, variant: str = "") -> str:
    return f"""
    <div class="kpi {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """


def panel_header(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="panel-sub">{sub}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Chargement & calculs
# ─────────────────────────────────────────────────────────────
feature_df = load_data()
metrics = load_model_metrics()

best_model = metrics.sort_values("RMSE").iloc[0]
baseline = metrics[metrics["Model"] == "Baseline"].iloc[0]
rmse_gain = 100 * (baseline["RMSE"] - best_model["RMSE"]) / baseline["RMSE"]
mae_gain = 100 * (baseline["MAE"] - best_model["MAE"]) / baseline["MAE"]

with st.sidebar:
    st.header("Filtres")
    selected_store = st.selectbox("Magasin", sorted(feature_df["Store"].unique().tolist()))
    selected_year = st.selectbox("Année", sorted(feature_df["Date"].dt.year.unique().tolist()))
    st.divider()
    st.caption(
        "Les indicateurs portent sur le magasin et l'année choisis, "
        "jours d'ouverture uniquement."
    )

filtered = feature_df[
    (feature_df["Store"] == selected_store) & (feature_df["Date"].dt.year == selected_year)
]
filtered = filtered[filtered["Open"] == 1].copy()

if filtered.empty:
    st.warning("Aucune donnée d'ouverture pour ce magasin et cette année. Choisissez une autre sélection.")
    st.stop()

monthly = (
    filtered.groupby(filtered["Date"].dt.to_period("M"))["Sales"]
    .agg(total="sum", moyenne="mean")
    .reset_index()
)
monthly["Mois"] = monthly["Date"].dt.to_timestamp()

promo_sales = filtered.loc[filtered["Promo"] == 1, "Sales"]
nonpromo_sales = filtered.loc[filtered["Promo"] == 0, "Sales"]
if not promo_sales.empty and not nonpromo_sales.empty:
    promo_lift = promo_sales.mean() - nonpromo_sales.mean()
    promo_lift_pct = 100 * promo_lift / nonpromo_sales.mean()
else:
    promo_lift, promo_lift_pct = 0.0, 0.0

weekday = filtered.groupby("DayOfWeek")["Sales"].mean().reset_index()
weekday["Jour"] = weekday["DayOfWeek"].map(DAY_LABELS)

summary_total = filtered["Sales"].sum()
summary_mean = filtered["Sales"].mean()
summary_promo_days = int(filtered["Promo"].sum())
peak_month = monthly.sort_values("total", ascending=False).iloc[0]
best_day = weekday.sort_values("Sales", ascending=False).iloc[0]

# ─────────────────────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Prévision des ventes Rossmann</h1>
        <p>Suivez les ventes d'un magasin, mesurez l'effet des promotions
        et comparez la précision des modèles de prévision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.markdown(
    kpi("Meilleur modèle", str(best_model["Model"]),
        f'<span class="chip teal">RMSE {best_model["RMSE"]:.2f}</span>', "good"),
    unsafe_allow_html=True,
)
k2.markdown(
    kpi("Ventes totales", currency(summary_total), f"Magasin {selected_store} · {selected_year}", "primary"),
    unsafe_allow_html=True,
)
k3.markdown(
    kpi("Ventes moyennes par jour", currency(summary_mean), f"{summary_promo_days} jours de promotion"),
    unsafe_allow_html=True,
)
k4.markdown(
    kpi("Effet des promotions", currency(promo_lift),
        f'<span class="chip red">{promo_lift_pct:+.1f}%</span> vs jours sans promotion'),
    unsafe_allow_html=True,
)

st.write("")

with st.container(border=True):
    panel_header("Impact business", "Interprétation métier des performances observées.")
    i1, i2, i3 = st.columns(3)
    i1.metric("Effet promo", f"{promo_lift_pct:+.1f}%")
    i2.metric("Jour le plus fort", str(best_day["Jour"]))
    i3.metric("Mois clé", peak_month["Date"].strftime("%b %Y"))

    st.markdown(
        """
        <div style="margin-top: 0.5rem; color: #6B7385; font-size: 0.92rem; line-height: 1.6;">
            <strong>Insight métier :</strong> les jours de promotion augmentent en moyenne les ventes de
            <strong>{:.1f}%</strong> par rapport aux jours sans promotion. Le magasin présente ses meilleures performances
            autour de <strong>{}</strong>, ce qui confirme un potentiel d'optimisation commercial en ciblant les périodes les plus propices.
        </div>
        """.format(promo_lift_pct, str(best_day["Jour"])) ,
        unsafe_allow_html=True,
    )

st.write("")

# ─────────────────────────────────────────────────────────────
# Benchmark des modèles
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1.7, 1.1])

with left:
    with st.container(border=True):
        panel_header("Comparaison des modèles", "Erreur moyenne en euros : plus la barre est basse, meilleur est le modèle.")
        long_df = metrics.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Métrique", value_name="Erreur")
        bench = (
            alt.Chart(long_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="Métrique:N",
                y=alt.Y("Erreur:Q", title="Erreur (€)"),
                color=alt.Color("Métrique:N", title=None,
                                scale=alt.Scale(domain=["MAE", "RMSE"], range=[TEAL, RED])),
                tooltip=["Model", "Métrique", alt.Tooltip("Erreur:Q", format=",.2f")],
            )
        )
        st.altair_chart(style_chart(bench), use_container_width=True)

with right:
    with st.container(border=True):
        panel_header("Gain par rapport à la référence")
        st.markdown(
            f"Avec **{best_model['Model']}**, l'erreur RMSE baisse de "
            f'<span class="chip teal">{pct(rmse_gain)}</span> et l\'erreur MAE de '
            f'<span class="chip teal">{pct(mae_gain)}</span>.',
            unsafe_allow_html=True,
        )
        st.write("")
        st.dataframe(
            metrics.sort_values("RMSE").round(4),
            hide_index=True,
            use_container_width=True,
        )

st.write("")

# ─────────────────────────────────────────────────────────────
# Onglets
# ─────────────────────────────────────────────────────────────
tabs = st.tabs(["Vue du magasin", "Habitudes de vente", "Analyse des modèles", "Aperçu des données"])

with tabs[0]:
    with st.container(border=True):
        panel_header("Ventes mensuelles", f"Total des ventes par mois, magasin {selected_store}, {selected_year}.")
        month_chart = (
            alt.Chart(monthly)
            .mark_bar(color=RED, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("yearmonth(Mois):T", title=None, axis=alt.Axis(format="%b", labelAngle=0)),
                y=alt.Y("total:Q", title="Ventes (€)"),
                tooltip=[alt.Tooltip("yearmonth(Mois):T", title="Mois", format="%B %Y"),
                         alt.Tooltip("total:Q", title="Total", format=",.0f"),
                         alt.Tooltip("moyenne:Q", title="Moyenne/jour", format=",.0f")],
            )
        )
        st.altair_chart(style_chart(month_chart), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Ventes totales", currency(summary_total))
    c2.metric("Vente moyenne par jour", currency(summary_mean))
    c3.metric("Jours de promotion", f"{summary_promo_days}")

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            panel_header("Ventes moyennes par jour de la semaine")
            wk = (
                alt.Chart(weekday)
                .mark_bar(color=TEAL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Jour:N", sort=list(DAY_LABELS.values()), title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Sales:Q", title="Ventes moyennes (€)"),
                    tooltip=["Jour", alt.Tooltip("Sales:Q", title="Moyenne", format=",.0f")],
                )
            )
            st.altair_chart(style_chart(wk), use_container_width=True)
    with col2:
        with st.container(border=True):
            panel_header("Évolution quotidienne", "Ventes du jour et moyenne sur 7 jours.")
            trend = filtered[["Date", "Sales"]].sort_values("Date").copy()
            trend["Moyenne 7 jours"] = trend["Sales"].rolling(7, min_periods=1).mean()
            base = alt.Chart(trend).encode(x=alt.X("Date:T", title=None))
            daily = base.mark_line(color=RED, opacity=0.35, strokeWidth=1.2).encode(
                y=alt.Y("Sales:Q", title="Ventes (€)"),
                tooltip=["Date:T", alt.Tooltip("Sales:Q", format=",.0f")],
            )
            smooth = base.mark_line(color=AMBER, strokeWidth=2.5).encode(y="Moyenne 7 jours:Q")
            st.altair_chart(style_chart(daily + smooth), use_container_width=True)

with tabs[2]:
    with st.container(border=True):
        panel_header("Qualité des prévisions par modèle", "Classement du plus précis au moins précis (RMSE).")
        st.dataframe(
            metrics.sort_values("RMSE").reset_index(drop=True).round(4),
            hide_index=True,
            use_container_width=True,
        )

    st.write("")
    with st.container(border=True):
        panel_header("Insights métier et recommandations", "Interprétation commerciale du benchmark.")
        if best_model["Model"] == "LightGBM":
            st.success(
                "LightGBM est le modèle le plus précis sur cette tâche et reste le modèle recommandé pour un usage opérationnel. "
                "Sa meilleure précision permet une meilleure allocation des stocks, des ressources promotionnelles et une anticipation des ventes."
            )
        else:
            st.info(f"{best_model['Model']} est le plus précis ici ; le choix final peut dépendre des contraintes métier.")

        st.markdown(
            f"- <strong>Impact promo :</strong> {promo_lift_pct:+.1f}% de différence moyenne entre les jours avec et sans promotions.\n"
            f"- <strong>Jour le plus fort :</strong> {best_day['Jour']} avec une moyenne de ventes de {currency(best_day['Sales'])} par jour.\n"
            f"- <strong>Période clé :</strong> {peak_month['Date'].strftime('%b %Y')} est le mois le plus rentable de la sélection.\n"
            f"- <strong>Recommandation :</strong> prioriser les campagnes promotionnelles autour de cette fenêtre et ajuster les stocks sur les jours de pointe.",
            unsafe_allow_html=True,
        )

with tabs[3]:
    with st.container(border=True):
        panel_header("Aperçu des données", "15 premières lignes de la sélection.")
        preview_cols = [
            "Date", "Store", "Sales", "Promo", "StateHoliday", "SchoolHoliday",
            "DayOfWeek", "sales_lag_7", "sales_rolling_mean_7",
        ]
        st.dataframe(filtered[preview_cols].head(15), use_container_width=True)

st.caption("Données : jeu Rossmann prétraité et comparaison finale des modèles.")