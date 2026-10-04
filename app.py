"""
Application Streamlit — Classification de graines de blé (CAH, 2 groupes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

FEATURE_LABELS_FR = {
    "area A": "Aire (A)",
    "perimeter": "Périmètre",
    "compactness": "Compacité",
    "length of kernel": "Longueur du grain",
    "width of kernel": "Largeur du grain",
    "asymmetry coefficient": "Coefficient d'asymétrie",
    "length of kernel groove": "Longueur du sillon",
}

PALETTE = {
    "0": "#2D6A4F",
    "1": "#D4A373",
    "bg": "#0F1419",
    "card": "rgba(255, 255, 255, 0.06)",
    "accent": "#E9C46A",
    "text": "#F4F1DE",
    "muted": "#A8B2B8",
}


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;700&family=Outfit:wght@300;400;500;600&display=swap');

        .stApp {{
            background: radial-gradient(ellipse 120% 80% at 10% -20%, #1a3d2e 0%, transparent 50%),
                        radial-gradient(ellipse 80% 60% at 100% 0%, #3d2c1e 0%, transparent 45%),
                        linear-gradient(165deg, #0a0e12 0%, #121820 40%, #0f1419 100%);
            color: {PALETTE["text"]};
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(15, 20, 25, 0.98) 0%, rgba(26, 35, 45, 0.95) 100%);
            border-right: 1px solid rgba(233, 196, 106, 0.15);
        }}

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown p {{
            font-family: 'Outfit', sans-serif !important;
        }}

        h1, h2, h3 {{
            font-family: 'Fraunces', serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}

        p, label, .stMarkdown, span {{
            font-family: 'Outfit', sans-serif !important;
        }}

        .hero-wrap {{
            padding: 2rem 0 1.5rem 0;
            animation: fadeUp 0.8s ease-out;
        }}

        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(16px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .hero-badge {{
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(233, 196, 106, 0.12);
            border: 1px solid rgba(233, 196, 106, 0.35);
            color: {PALETTE["accent"]};
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }}

        .hero-title {{
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.15;
            background: linear-gradient(135deg, #F4F1DE 0%, #E9C46A 50%, #D4A373 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 0.5rem 0;
        }}

        .hero-sub {{
            color: {PALETTE["muted"]};
            font-size: 1.05rem;
            max-width: 42rem;
            line-height: 1.6;
        }}

        .glass-card {{
            background: {PALETTE["card"]};
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        }}

        .result-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }}

        div[data-testid="stMetric"] label {{
            color: {PALETTE["muted"]} !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {PALETTE["accent"]} !important;
            font-family: 'Fraunces', serif !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(255, 255, 255, 0.04);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: {PALETTE["muted"]};
            font-family: 'Outfit', sans-serif;
            padding: 0.5rem 1rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: rgba(233, 196, 106, 0.15) !important;
            border-color: rgba(233, 196, 106, 0.4) !important;
            color: {PALETTE["accent"]} !important;
        }}

        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_resources() -> dict:
    features = json.loads((BASE_DIR / "features.json").read_text(encoding="utf-8"))
    defaults = json.loads((BASE_DIR / "valeurs_defaut.json").read_text(encoding="utf-8"))
    class_names = json.loads((BASE_DIR / "noms_classes.json").read_text(encoding="utf-8"))
    examples = json.loads((BASE_DIR / "exemples.json").read_text(encoding="utf-8"))
    centres = pd.read_csv(BASE_DIR / "centres_cah.csv")
    dataset = pd.read_csv(BASE_DIR / "wheat_seeds_dataset.csv")
    return {
        "features": features,
        "defaults": defaults,
        "class_names": class_names,
        "examples": examples,
        "centres": centres,
        "dataset": dataset,
    }


def vector_from_inputs(features: list[str], values: dict[str, float]) -> np.ndarray:
    return np.array([values[f] for f in features], dtype=float)


def classify_nearest_centroid(
    x: np.ndarray, centres: pd.DataFrame, features: list[str]
) -> tuple[int, dict[int, float]]:
    distances: dict[int, float] = {}
    for _, row in centres.iterrows():
        cluster_id = int(row["Classe_CAH"])
        center = row[features].values.astype(float)
        distances[cluster_id] = float(np.linalg.norm(x - center))
    predicted = min(distances, key=distances.get)
    return predicted, distances


def confidence_from_distances(distances: dict[int, float]) -> float:
    d_vals = sorted(distances.values())
    if len(d_vals) < 2 or d_vals[0] + d_vals[1] == 0:
        return 1.0
    margin = (d_vals[1] - d_vals[0]) / (d_vals[0] + d_vals[1])
    return float(np.clip(margin * 2, 0.05, 0.99))


def radar_figure(
    features: list[str],
    values: np.ndarray,
    centres: pd.DataFrame,
    predicted: int,
) -> go.Figure:
    labels = [FEATURE_LABELS_FR.get(f, f) for f in features]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    colors = [PALETTE["0"], PALETTE["1"]]

    vmin = np.minimum(values, centres[features].min().values)
    vmax = np.maximum(values, centres[features].max().values)
    span = np.where(vmax - vmin == 0, 1.0, vmax - vmin)

    def norm(v: np.ndarray) -> list[float]:
        n = (v - vmin) / span
        return np.clip(n, 0, 1).tolist() + [float(np.clip((v[0] - vmin[0]) / span[0], 0, 1))]

    sample_norm = norm(values)
    fig.add_trace(
        go.Scatterpolar(
            r=sample_norm,
            theta=labels_closed,
            fill="toself",
            name="Votre échantillon",
            line=dict(color=PALETTE["accent"], width=2),
            fillcolor="rgba(233, 196, 106, 0.25)",
        )
    )

    for i, row in centres.iterrows():
        cid = int(row["Classe_CAH"])
        center = row[features].values.astype(float)
        fig.add_trace(
            go.Scatterpolar(
                r=norm(center),
                theta=labels_closed,
                fill="toself",
                name=f"Centre {cid}",
                line=dict(color=colors[cid % len(colors)], width=1.5, dash="dot"),
                fillcolor=f"rgba({int(colors[cid][1:3], 16)}, {int(colors[cid][3:5], 16)}, {int(colors[cid][5:7], 16)}, 0.12)",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color=PALETTE["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=40, b=80, l=60, r=60),
        height=420,
    )
    return fig


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">🌾 Clustering · CAH · Wheat Seeds</div>
            <h1 class="hero-title">Graines de blé — Classification intelligente</h1>
            <p class="hero-sub">
                Saisissez les mesures morphologiques d'un grain ou choisissez un exemple.
                L'application l'assigne au groupe le plus proche (classification par centroïdes CAH).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls(res: dict) -> tuple[str, int | None]:
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Section",
        ["🏠 Accueil & Classifier", "📊 Explorer le jeu de données", "ℹ️ Méthode"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Exemples rapides")
    example_idx = st.sidebar.selectbox(
        "Charger un profil",
        options=list(range(len(res["examples"]))),
        format_func=lambda i: f"Exemple {i + 1}",
        index=None,
        placeholder="— Choisir —",
    )
    return page, example_idx


def build_input_form(res: dict) -> dict[str, float]:
    features = res["features"]
    defaults = res["defaults"]
    dataset = res["dataset"]
    values: dict[str, float] = {}

    st.markdown("#### Mesures du grain")
    cols = st.columns(2)
    for idx, feat in enumerate(features):
        col = cols[idx % 2]
        lo = float(dataset[feat].min())
        hi = float(dataset[feat].max())
        step = 0.001 if feat == "compactness" else 0.01
        label = FEATURE_LABELS_FR.get(feat, feat)
        with col:
            values[feat] = st.number_input(
                label,
                min_value=lo,
                max_value=hi,
                value=float(defaults[feat]),
                step=step,
                format="%.4f" if feat == "compactness" else "%.3f",
                key=f"input_{feat}",
            )
    return values


def page_classifier(res: dict, example_idx: int | None) -> None:
    features = res["features"]
    centres = res["centres"]
    class_names = res["class_names"]

    if example_idx is not None:
        for i, feat in enumerate(features):
            st.session_state[f"input_{feat}"] = float(res["examples"][example_idx][i])

    col_form, col_viz = st.columns([1.05, 0.95], gap="large")

    with col_form:
        values = build_input_form(res)
        classify = st.button("✨ Classifier ce grain", type="primary", use_container_width=True)

    x = vector_from_inputs(features, values)

    with col_viz:
        st.markdown("#### Profil comparé aux centres")
        if classify:
            pred, dists = classify_nearest_centroid(x, centres, features)
            conf = confidence_from_distances(dists)
            st.session_state["last_prediction"] = {
                "pred": pred,
                "dists": dists,
                "conf": conf,
                "values": values,
            }

        if "last_prediction" in st.session_state:
            state = st.session_state["last_prediction"]
            pred = state["pred"]
            dists = state["dists"]
            conf = state["conf"]
            name = class_names.get(str(pred), f"Groupe {pred}")
            color = PALETTE[str(pred % 2)]

            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="result-pill" style="background: {color}22; border: 1px solid {color}55; color: {PALETTE['text']};">
                        Résultat : <strong>{name}</strong>
                    </div>
                    <p style="color: {PALETTE['muted']}; margin-top: 0.75rem; font-size: 0.95rem;">
                        Confiance relative (écart aux centroïdes) : <strong style="color: {PALETTE['accent']};">{conf:.0%}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Distance au groupe assigné", f"{dists[pred]:.3f}")
            other_key = 1 if pred == 0 else 0
            m2.metric("Distance à l'autre groupe", f"{dists[other_key]:.3f}")
            m3.metric("Features", len(features))

            fig = radar_figure(features, x, centres, pred)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cliquez sur **Classifier ce grain** pour voir le résultat et le graphique radar.")


def page_explore(res: dict) -> None:
    df = res["dataset"]
    features = res["features"]

    st.markdown("#### Aperçu statistique")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Échantillons", len(df))
    c2.metric("Variables", len(features))
    c3.metric("Aire moyenne", f"{df['area A'].mean():.2f}")
    c4.metric("Compacité moy.", f"{df['compactness'].mean():.4f}")

    tab1, tab2, tab3 = st.tabs(["Corrélations", "Distributions", "Nuage 3D"])

    with tab1:
        corr = df[features].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=[[0, "#1a3d2e"], [0.5, "#0f1419"], [1, "#E9C46A"]],
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text"]),
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        feat = st.selectbox(
            "Variable",
            features,
            format_func=lambda f: FEATURE_LABELS_FR.get(f, f),
        )
        fig = px.histogram(
            df,
            x=feat,
            nbins=28,
            color_discrete_sequence=[PALETTE["accent"]],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["text"]),
            xaxis_title=FEATURE_LABELS_FR.get(feat, feat),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        x_ax = st.selectbox("Axe X", features, index=0, key="3d_x")
        y_ax = st.selectbox("Axe Y", features, index=3, key="3d_y")
        z_ax = st.selectbox("Axe Z", features, index=4, key="3d_z")
        fig = px.scatter_3d(
            df,
            x=x_ax,
            y=y_ax,
            z=z_ax,
            opacity=0.75,
            color_discrete_sequence=[PALETTE["0"]],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title=FEATURE_LABELS_FR.get(x_ax, x_ax),
                yaxis_title=FEATURE_LABELS_FR.get(y_ax, y_ax),
                zaxis_title=FEATURE_LABELS_FR.get(z_ax, z_ax),
                bgcolor="rgba(0,0,0,0)",
            ),
            font=dict(color=PALETTE["text"]),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Voir les données brutes"):
        st.dataframe(df, use_container_width=True, height=320)


def page_method(res: dict) -> None:
    centres = res["centres"]
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Comment ça marche ?</h3>
            <p style="color: #A8B2B8; line-height: 1.7;">
                Le notebook entraîne un <strong>clustering hiérarchique agglomératif (CAH)</strong>
                avec <code>linkage='average'</code> et <strong>2 groupes</strong> sur le jeu
                <em>Wheat Seeds</em> (199 grains, 7 descripteurs géométriques).
            </p>
            <p style="color: #A8B2B8; line-height: 1.7;">
                Cette interface calcule la <strong>distance euclidienne</strong> entre votre saisie
                et les <strong>centroïdes</strong> exportés (<code>centres_cah.csv</code>), puis
                assigne le groupe le plus proche — même logique qu'une règle du plus proche voisin
                sur les centres de clusters.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Centroïdes des groupes")
    display = centres.copy()
    display["Classe_CAH"] = display["Classe_CAH"].map(
        lambda c: res["class_names"].get(str(int(c)), f"Groupe {c}")
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Graines de blé — Clustering",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    res = load_resources()
    page, example_idx = sidebar_controls(res)
    render_hero()

    if page.startswith("🏠"):
        page_classifier(res, example_idx)
    elif page.startswith("📊"):
        page_explore(res)
    else:
        page_method(res)


if __name__ == "__main__":
    main()
