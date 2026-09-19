"""
Mental Health in Tech Survey — Interactive Analytics & Prediction App
Author: P Suman Sangeet
Source notebook: Mental_Health_in_Tech_Survey_EDA_and_ML.ipynb
Dataset: OSMI Mental Health in Tech Survey (2014, n=1259) + 2016 wave (n=1433)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------------------------------
# Page config & palette
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental Health in Tech Survey | Analytics & Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Futuristic dark palette
# --------------------------------------------------------------------------------------
VOID = "#080B18"          # page background
PANEL = "#111629"          # card / panel background
PANEL_2 = "#161C33"        # slightly lighter panel (hover / secondary)
GRID = "#232A47"           # gridlines, borders
CYAN = "#00E5FF"           # primary neon accent
PURPLE = "#B26BFF"         # secondary neon accent
PINK = "#FF4FC3"           # tertiary neon accent
GREEN = "#39FFB0"          # positive / success neon
AMBER = "#FFC857"          # warning / highlight neon
CORAL = "#FF6B6B"          # negative / alert neon
TEXT = "#E7ECFB"           # primary text on dark
MUTED = "#8B96C0"          # secondary / muted text

# Kept for backward compatibility with earlier variable names used below
TEAL = CYAN
NAVY = VOID
SLATE = MUTED
BLUE = "#4FC3F7"
BG = PANEL
PALETTE = [CYAN, PURPLE, PINK, GREEN, AMBER, CORAL]

# --------------------------------------------------------------------------------------
# Register a matching dark Plotly template so every chart shares the futuristic theme
# --------------------------------------------------------------------------------------
pio.templates["futuristic_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="sans serif"),
        title=dict(font=dict(color=TEXT, size=17)),
        colorway=PALETTE,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=MUTED),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        polar=dict(
            bgcolor=PANEL,
            radialaxis=dict(gridcolor=GRID, linecolor=GRID, color=MUTED),
            angularaxis=dict(gridcolor=GRID, linecolor=GRID, color=MUTED),
        ),
        coloraxis=dict(colorbar=dict(tickfont=dict(color=MUTED))),
        hoverlabel=dict(bgcolor=PANEL_2, font=dict(color=TEXT)),
    )
)
pio.templates.default = "futuristic_dark"

CUSTOM_CSS = f"""
<style>
    @keyframes auroraShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 12px rgba(0,229,255,0.18), 0 0 2px rgba(0,229,255,0.35) inset; }}
        50%      {{ box-shadow: 0 0 22px rgba(0,229,255,0.32), 0 0 4px rgba(0,229,255,0.5) inset; }}
    }}

    .stApp {{
        background:
            radial-gradient(circle at 15% 8%, rgba(178,107,255,0.10), transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(0,229,255,0.10), transparent 40%),
            {VOID};
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0B0F20 0%, {VOID} 100%);
        border-right: 1px solid {GRID};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
    section[data-testid="stSidebar"] .stRadio label span {{ font-size: 0.95rem; }}
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {{
        border-color: {CYAN} !important;
    }}

        /* --------------------------------------------------
       MHT Corporate Logo
    -------------------------------------------------- */

    .mht-brand {{
        text-align: center;
        padding: 10px 5px 18px 5px;
        margin-bottom: 15px;
        border-bottom: 1px solid #232A47;
    }}

    .mht-brand img {{
        width: 100%;
        max-width: 220px;
        height: auto;
        margin-bottom: 10px;
    }}

    .mht-brand-title {{
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 3px;
        color: #00E5FF;
    }}

    .mht-brand-subtitle {{
        font-size: 10px;
        letter-spacing: 1.5px;
        color: #8B96C0;
        margin-top: 5px;
    }}

    .hero {{
        background: linear-gradient(120deg, {VOID} 0%, #1B1440 40%, #0E3A4A 75%, {VOID} 100%);
        background-size: 260% 260%;
        animation: auroraShift 14s ease infinite;
        padding: 2.6rem 2.3rem;
        border-radius: 16px;
        border: 1px solid {GRID};
        color: {TEXT};
        margin-bottom: 1.4rem;
        box-shadow: 0 0 40px rgba(0,229,255,0.08);
    }}
    .hero h1 {{
        margin: 0 0 0.4rem 0; font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(90deg, {CYAN}, {PURPLE} 60%, {PINK});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero p {{ margin: 0; font-size: 1.02rem; color: {MUTED}; max-width: 780px; line-height: 1.6; }}
    .hero .tag {{
        display: inline-block; background: rgba(0,229,255,0.12); border: 1px solid rgba(0,229,255,0.35);
        color: {CYAN}; padding: 0.22rem 0.7rem;
        border-radius: 20px; font-size: 0.76rem; margin-bottom: 0.9rem; letter-spacing: 0.03em;
    }}

    .kpi-card {{
        background: linear-gradient(160deg, {PANEL} 0%, {PANEL_2} 100%);
        backdrop-filter: blur(6px);
        border-radius: 14px; padding: 1.1rem 1.2rem;
        border: 1px solid {GRID}; height: 100%;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(0,229,255,0.45);
        box-shadow: 0 0 22px rgba(0,229,255,0.18);
    }}
    .kpi-card .kpi-label {{
        font-size: 0.78rem; color: {MUTED}; margin-bottom: 0.35rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .kpi-card .kpi-value {{
        font-size: 1.7rem; font-weight: 800; color: {TEXT};
        text-shadow: 0 0 14px rgba(0,229,255,0.35);
    }}
    .kpi-card .kpi-sub {{ font-size: 0.78rem; color: {CYAN}; margin-top: 0.2rem; }}

    .insight-box {{
        background: linear-gradient(90deg, rgba(255,200,87,0.08), rgba(17,22,41,0.4));
        border-left: 3px solid {AMBER}; padding: 0.85rem 1.05rem;
        border-radius: 8px; font-size: 0.92rem; color: {TEXT}; margin: 0.6rem 0 1.1rem 0;
    }}
    .impact-box {{
        background: linear-gradient(90deg, rgba(0,229,255,0.08), rgba(17,22,41,0.4));
        border-left: 3px solid {CYAN}; padding: 0.85rem 1.05rem;
        border-radius: 8px; font-size: 0.92rem; color: {TEXT}; margin: 0.6rem 0 1.1rem 0;
    }}

    h1, h2, h3, h4, p, span, label, .stMarkdown {{ color: {TEXT}; }}
    h2, h3 {{ color: {TEXT}; }}
    .section-note {{ color: {MUTED}; font-size: 0.88rem; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {PANEL}; border-radius: 8px 8px 0 0; color: {MUTED};
        border: 1px solid {GRID}; border-bottom: none;
    }}
    .stTabs [aria-selected="true"] {{ color: {CYAN} !important; }}

    div[data-testid="stForm"] {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 14px; padding: 1.2rem;
    }}
    .stButton>button, .stFormSubmitButton>button {{
        background: linear-gradient(90deg, {CYAN}, {PURPLE});
        color: {VOID}; font-weight: 700; border: none; border-radius: 8px;
        animation: glowPulse 3s ease-in-out infinite;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{ filter: brightness(1.1); }}

    div[data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 10px; padding: 0.6rem 0.8rem;
    }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {VOID}; }}
    ::-webkit-scrollbar-thumb {{ background: {GRID}; border-radius: 6px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {CYAN}; }}

    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Data loading & cleaning (mirrors the source notebook's pipeline exactly)
# --------------------------------------------------------------------------------------
ENCODE_COLS = [
    "Gender", "self_employed", "family_history", "treatment", "work_interfere",
    "no_employees", "remote_work", "tech_company", "benefits", "care_options",
    "wellness_program", "seek_help", "anonymity", "leave", "mental_health_consequence",
    "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview",
    "phys_health_interview", "mental_vs_physical", "obs_consequence",
]

FRIENDLY_NAMES = {
    "self_employed": "Self-employed", "family_history": "Family history of mental illness",
    "work_interfere": "Condition interferes with work", "no_employees": "Company size",
    "remote_work": "Works remotely", "tech_company": "Tech company",
    "benefits": "Employer provides mental-health benefits", "care_options": "Aware of care options",
    "wellness_program": "Wellness program offered", "seek_help": "Employer resources to seek help",
    "anonymity": "Anonymity protected", "leave": "Ease of taking mental-health leave",
    "mental_health_consequence": "Fears mental-health disclosure consequence",
    "phys_health_consequence": "Fears physical-health disclosure consequence",
    "coworkers": "Comfortable with coworkers", "supervisor": "Comfortable with supervisor",
    "mental_health_interview": "Would raise in interview (mental)",
    "phys_health_interview": "Would raise in interview (physical)",
    "mental_vs_physical": "Employer treats mental = physical health",
    "obs_consequence": "Observed negative consequences for others", "Gender": "Gender", "Age": "Age",
}


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv("data/survey.csv")


def _normalize_gender(g) -> str:
    g = str(g).strip().lower()
    male_set = {"male", "m", "man", "cis male", "cis man", "male (cis)", "mail", "malr",
                "make", "msle", "maile", "mal", "male-ish", "guy (-ish) ^_^", "male leaning androgynous"}
    female_set = {"female", "f", "woman", "cis female", "cis-female/femme", "female (cis)",
                  "femake", "femail", "female "}
    if g in male_set:
        return "Male"
    if g in female_set:
        return "Female"
    return "Other"


@st.cache_data
def clean_2014(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.drop(columns=["comments", "Timestamp", "state"], inplace=True)

    d.loc[(d["Age"] < 15) | (d["Age"] > 75), "Age"] = np.nan
    d["Age"] = d["Age"].fillna(d["Age"].median()).astype(int)

    d["Gender"] = d["Gender"].apply(_normalize_gender)
    d["self_employed"] = d["self_employed"].fillna(d["self_employed"].mode()[0])
    d["work_interfere"] = d["work_interfere"].fillna("Not applicable")

    top_countries = d["Country"].value_counts().nlargest(8).index
    d["Country_grouped"] = np.where(d["Country"].isin(top_countries), d["Country"], "Other")

    d["Age_group"] = pd.cut(d["Age"], bins=[14, 24, 34, 44, 54, 75],
                             labels=["15-24", "25-34", "35-44", "45-54", "55-75"])
    return d


@st.cache_resource
def encode_features(df_clean: pd.DataFrame):
    df_encoded = df_clean.copy()
    label_encoders = {}
    for c in ENCODE_COLS:
        le = LabelEncoder()
        df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))
        label_encoders[c] = le
    corr_cols = ["Age"] + ENCODE_COLS
    feature_cols = [c for c in corr_cols if c != "treatment"]
    return df_encoded, label_encoders, feature_cols, corr_cols


# Models that need standardized inputs (distance/gradient based); tree ensembles don't.
SCALED_MODELS = {"Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors", "Neural Network (MLP)"}


@st.cache_resource
def train_models(df_encoded: pd.DataFrame, feature_cols: list):
    X = df_encoded[feature_cols]
    y = df_encoded["treatment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
        # --- add-on models ---
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", max_iter=800,
            early_stopping=True, random_state=42,
        ),
    }

    rows, fitted, curves, preds, probas = [], {}, {}, {}, {}
    for name, model in models.items():
        if name in SCALED_MODELS:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        fitted[name] = model
        preds[name] = y_pred
        probas[name] = y_proba
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        curves[name] = (fpr, tpr)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1),
            "Recall": recall_score(y_test, y_pred, pos_label=1),
            "F1 Score": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })

    results_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]
    best_pred = preds[best_name]
    cm = confusion_matrix(y_test, best_pred)
    report = classification_report(y_test, best_pred, target_names=["No", "Yes"], output_dict=True)

    return {
        "fitted": fitted, "results_df": results_df, "scaler": scaler, "curves": curves,
        "best_name": best_name, "confusion_matrix": cm, "report": report,
        "X_test": X_test, "y_test": y_test, "preds": preds, "probas": probas,
    }


@st.cache_data
def load_2016_combined(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_2016_raw = pd.read_csv("data/mental-heath-in-tech-2016_20161114.csv")
    col_map_2016 = {
        "Are you self-employed?": "self_employed",
        "How many employees does your company or organization have?": "no_employees",
        "Is your employer primarily a tech company/organization?": "tech_company",
        "Does your employer provide mental health benefits as part of healthcare coverage?": "benefits",
        "Do you know the options for mental health care available under your employer-provided coverage?": "care_options",
        "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "anonymity",
        "Do you think that discussing a mental health disorder with your employer would have negative consequences?": "mental_health_consequence",
        "Would you feel comfortable discussing a mental health disorder with your coworkers?": "coworkers",
        "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "supervisor",
        "Do you feel that your employer takes mental health as seriously as physical health?": "mental_vs_physical",
        "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "obs_consequence",
        "Do you have a family history of mental illness?": "family_history",
        "Have you ever sought treatment for a mental health issue from a mental health professional?": "treatment",
        "What is your age?": "Age",
        "What is your gender?": "Gender",
        "What country do you live in?": "Country",
        "Do you work remotely?": "remote_work",
    }
    df_2016 = df_2016_raw.rename(columns=col_map_2016)[list(col_map_2016.values())].copy()
    df_2016["treatment"] = df_2016["treatment"].map({1: "Yes", 0: "No"})
    df_2016["self_employed"] = df_2016["self_employed"].map({1: "Yes", 0: "No"})
    df_2016.loc[(df_2016["Age"] < 15) | (df_2016["Age"] > 75), "Age"] = np.nan
    df_2016["Age"] = df_2016["Age"].fillna(df_2016["Age"].median()).astype(int)
    df_2016["Gender"] = df_2016["Gender"].apply(_normalize_gender)
    df_2016["Country"] = df_2016["Country"].replace({"United States of America": "United States"})
    df_2016["remote_work"] = df_2016["remote_work"].map({"Always": "Yes", "Sometimes": "Yes", "Never": "No"})
    df_2016["Year"] = 2016

    df_2014_slim = df_clean[[
        "self_employed", "no_employees", "tech_company", "benefits", "care_options",
        "anonymity", "mental_health_consequence", "coworkers", "supervisor",
        "mental_vs_physical", "obs_consequence", "family_history", "treatment",
        "Age", "Gender", "Country", "remote_work",
    ]].copy()
    df_2014_slim["Year"] = 2014

    return pd.concat([df_2014_slim, df_2016], ignore_index=True)


# --------------------------------------------------------------------------------------
# Load everything once
# --------------------------------------------------------------------------------------
raw_df = load_raw()
df_clean = clean_2014(raw_df)
df_encoded, label_encoders, feature_cols, corr_cols = encode_features(df_clean)
model_bundle = train_models(df_encoded, feature_cols)
df_combined = load_2016_combined(df_clean)

TREATMENT_RATE = (df_clean["treatment"] == "Yes").mean() * 100
best_model = model_bundle["fitted"][model_bundle["best_name"]]
best_auc = model_bundle["results_df"].iloc[0]["ROC-AUC"]

importance_df = pd.DataFrame({
    "Feature": [FRIENDLY_NAMES.get(f, f) for f in feature_cols],
    "raw": feature_cols,
    "Importance": model_bundle["fitted"]["Random Forest"].feature_importances_,
}).sort_values("Importance", ascending=False).reset_index(drop=True)
TOP_DRIVER = importance_df.iloc[0]["Feature"]


def kpi_card(col, label, value, sub=""):
    col.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div></div>""",
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f'<div class="insight-box">🔍 <b>Insight —</b> {text}</div>', unsafe_allow_html=True)


def impact(text):
    st.markdown(f'<div class="impact-box">💼 <b>Business impact —</b> {text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------

st.sidebar.image(
    "assets/mht_logo.png",
    use_container_width=True,
)

st.sidebar.markdown(
    """
    <div style="
        text-align: center;
        margin-top: -12px;
        margin-bottom: 18px;
    ">
        <div style="
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 3px;
            color: #00E5FF;
        ">
            MHT
        </div>

        <div style=
            font-size: 10px;
            letter-spacing: 1.5px;
            color: #8B96C0;
        >
            MENTAL HEALTH IN TECH-SUVEY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption("OSMI Survey · EDA + ML Capstone")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🔎 Explore & Segment",
        "🧩 Correlations & Drivers",
        "🧪 Model Performance",
        "🎯 Try the Prediction Tool",
        "📈 2014 vs 2016 Trends",
        "📌 Recommendations & About",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""<span style='font-size:0.78rem; color:#B9C7C4;'>
    Dataset: 2014 OSMI Mental Health in Tech Survey ({len(df_clean):,} respondents),
    with a 2016 wave ({(df_combined['Year']==2016).sum():,} respondents) for trend comparison.<br><br>
    Built by <b>P Suman Sangeet</b> — LABMENTIX Data Analytics and AI.
    </span>""",
    unsafe_allow_html=True,
)

# ======================================================================================
# PAGE: OVERVIEW
# ======================================================================================
if page == "🏠 Overview":
    st.markdown(
        f"""<div class="hero">
        <div class="tag">EDA + Predictive Analytics · Recruiter-Facing Demo</div>
        <h1>Mental Health in Tech Survey — Analytics & Prediction</h1>
        <p>What drives an employee in tech to seek treatment for a mental-health condition —
        and how can HR / people-ops teams target limited wellness budget where it will actually
        move the needle? This app turns a 1,259-respondent OSMI survey into an interactive
        dashboard, a validated ML model, and a live screening-style prediction tool.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Respondents analyzed", f"{len(df_clean):,}", "2014 OSMI survey, 26 cleaned features")
    kpi_card(c2, "Sought treatment", f"{TREATMENT_RATE:.1f}%", "of respondents overall")
    kpi_card(c3, "Best model ROC-AUC", f"{best_auc:.3f}", f"{model_bundle['best_name']}")
    kpi_card(c4, "Top predictor", TOP_DRIVER, "highest feature importance")

    st.markdown("### Project story")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(
            """
The raw survey arrives with real-world scars: 49 inconsistent spellings of `Gender`, ages
ranging from **-1,726 to 99,999,999,999**, and null-heavy columns like `state` (41% missing)
and `comments` (87% missing). A deliberate cleaning pass — capping age to a realistic 15–75
range, collapsing gender into 3 consistent buckets, and treating missing `work_interfere` as
its own meaningful category — brings the data to **zero remaining nulls** before any chart or
model is trusted.

From there the analysis moves through three layers: **who responded** (demographics),
**what workplace factors move with treatment-seeking** (benefits, anonymity, leave, care
options), and **can we predict it** (six models compared side by side — Logistic Regression,
Random Forest, Gradient Boosting, Support Vector Machine, K-Nearest Neighbors, and a small
Neural Network — evaluated on accuracy through ROC-AUC).
            """
        )
    with right:
        st.markdown("#### Key findings")
        st.markdown(
            f"""
- **Family history** and **work interference** are, by a wide margin, the two strongest
  predictors of treatment-seeking — well ahead of country, gender, or company size.
- Structural workplace factors — **care options awareness**, **anonymity**, **benefits** —
  show a smaller but consistent positive association with getting help.
- Treatment-seeking rose from the 2014 to the 2016 OSMI wave, though survey-composition
  changes mean that shift is directional, not strictly causal.
- The best model (**{model_bundle['best_name']}**) separates treatment-seekers from
  non-seekers with **{best_auc:.1%} ROC-AUC** on held-out data.
            """
        )

    impact(
        "HR and people-ops teams get more leverage from making existing mental-health "
        "benefits visible and easy to find than from adding new benefits nobody knows about — "
        "a classic 'the friction is in discovery, not provision' finding."
    )
    st.caption("Use the sidebar to explore demographics, workplace drivers, model performance, "
               "and a live prediction tool built on the same features as the notebook.")

# ======================================================================================
# PAGE: EXPLORE & SEGMENT
# ======================================================================================
elif page == "🔎 Explore & Segment":
    st.markdown("## 🔎 Explore & Segment")
    st.caption("Filter the cleaned 2014 survey and see how treatment-seeking shifts across segments.")

    f1, f2, f3 = st.columns(3)
    genders = f1.multiselect("Gender", sorted(df_clean["Gender"].unique()), default=list(df_clean["Gender"].unique()))
    countries = f2.multiselect("Country (top 8 + Other)", sorted(df_clean["Country_grouped"].unique()),
                                default=list(df_clean["Country_grouped"].unique()))
    sizes = f3.multiselect("Company size", list(df_clean["no_employees"].unique()),
                            default=list(df_clean["no_employees"].unique()))

    fdf = df_clean[
        df_clean["Gender"].isin(genders)
        & df_clean["Country_grouped"].isin(countries)
        & df_clean["no_employees"].isin(sizes)
    ]
    st.markdown(f"**{len(fdf):,} of {len(df_clean):,} respondents** match the current filters.")

    if len(fdf) == 0:
        st.warning("No respondents match this combination — widen a filter.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            g = fdf.groupby("Gender")["treatment"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="pct")
            fig = px.bar(g, x="Gender", y="pct", color="Gender", color_discrete_sequence=PALETTE,
                         text_auto=".1f", title="Treatment-seeking rate by gender (%)")
            fig.update_layout(template="futuristic_dark", showlegend=False, yaxis_title="% sought treatment")
            st.plotly_chart(fig, use_container_width=True)

            age_g = fdf.groupby("Age_group", observed=True)["treatment"].apply(
                lambda s: (s == "Yes").mean() * 100).reset_index(name="pct")
            fig2 = px.bar(age_g, x="Age_group", y="pct", color_discrete_sequence=[TEAL],
                          text_auto=".1f", title="Treatment-seeking rate by age group (%)")
            fig2.update_layout(template="futuristic_dark", yaxis_title="% sought treatment")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.histogram(fdf, x="Age", nbins=25, color_discrete_sequence=[BLUE],
                                 title="Age distribution of respondents")
            fig3.update_layout(template="futuristic_dark")
            st.plotly_chart(fig3, use_container_width=True)

            size_order = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
            size_g = fdf.groupby("no_employees")["treatment"].apply(
                lambda s: (s == "Yes").mean() * 100).reindex(size_order).dropna().reset_index(name="pct")
            fig4 = px.bar(size_g, x="no_employees", y="pct", color_discrete_sequence=[AMBER],
                          text_auto=".1f", title="Treatment-seeking rate by company size (%)")
            fig4.update_layout(template="futuristic_dark", xaxis_title="Company size (employees)",
                               yaxis_title="% sought treatment")
            st.plotly_chart(fig4, use_container_width=True)

        insight(
            "Treatment-seeking is fairly consistent across company sizes and skews higher for "
            "respondents identifying outside the Male/Female binary — but the gap narrows once "
            "family history and work interference are controlled for (see Correlations & Drivers)."
        )

        top_countries = fdf["Country_grouped"].value_counts().reset_index()
        top_countries.columns = ["Country", "Respondents"]
        fig5 = px.bar(top_countries, x="Respondents", y="Country", orientation="h",
                      color_discrete_sequence=[SLATE], title="Respondents by country")
        fig5.update_layout(template="futuristic_dark", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("The dataset is heavily US/UK-skewed, typical of an OSMI-community-shared online survey — "
                   "treat country-level cuts as directional rather than globally representative.")

# ======================================================================================
# PAGE: CORRELATIONS & DRIVERS
# ======================================================================================
elif page == "🧩 Correlations & Drivers":
    st.markdown("## 🧩 Correlations & Drivers")
    st.caption("Which workplace and personal factors move together with treatment-seeking?")

    corr_matrix = df_encoded[corr_cols].corr()
    heat_labels = [FRIENDLY_NAMES.get(c, c) for c in corr_cols]
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=heat_labels, y=heat_labels,
        colorscale="RdBu", zmid=0, colorbar=dict(title="corr"),
    ))
    fig.update_layout(template="futuristic_dark", height=650, title="Correlation heatmap — all encoded features",
                       margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    treat_corr = corr_matrix["treatment"].drop("treatment").abs().sort_values(ascending=False).head(10)
    treat_corr_df = pd.DataFrame({
        "Feature": [FRIENDLY_NAMES.get(i, i) for i in treat_corr.index],
        "Absolute correlation with treatment": treat_corr.values,
    })
    fig2 = px.bar(treat_corr_df, x="Absolute correlation with treatment", y="Feature", orientation="h",
                  color="Absolute correlation with treatment", color_continuous_scale=[SLATE, TEAL],
                  title="Top 10 features correlated with treatment-seeking")
    fig2.update_layout(template="futuristic_dark", yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    insight(
        f"<b>{FRIENDLY_NAMES['family_history']}</b> and <b>{FRIENDLY_NAMES['work_interfere']}</b> "
        "dominate the correlation ranking — far ahead of demographic or company-size variables."
    )

    st.markdown("### Workplace policy vs. treatment-seeking")
    policy_cols = ["benefits", "care_options", "anonymity", "leave"]
    tabs = st.tabs([FRIENDLY_NAMES[c] for c in policy_cols])
    for tab, col in zip(tabs, policy_cols):
        with tab:
            ct = pd.crosstab(df_clean[col], df_clean["treatment"], normalize="index") * 100
            ct = ct.reset_index().melt(id_vars=col, var_name="treatment", value_name="pct")
            fig3 = px.bar(ct, x=col, y="pct", color="treatment", barmode="group",
                          color_discrete_sequence=[SLATE, TEAL], text_auto=".1f",
                          title=f"Treatment-seeking rate by {FRIENDLY_NAMES[col].lower()} (%)")
            fig3.update_layout(template="futuristic_dark", xaxis_title=FRIENDLY_NAMES[col], yaxis_title="% of group")
            st.plotly_chart(fig3, use_container_width=True)

    impact(
        "Awareness and trust levers (knowing care options exist, believing anonymity is protected, "
        "clear benefits) show a real but secondary lift compared to family history and existing "
        "symptoms — so awareness campaigns are worthwhile, but won't fully close the treatment gap "
        "on their own."
    )

# ======================================================================================
# PAGE: MODEL PERFORMANCE
# ======================================================================================
elif page == "🧪 Model Performance":
    st.markdown("## 🧪 Model Performance")
    st.caption("Six classifiers trained to predict `treatment` from 22 cleaned features, "
               "evaluated on a held-out 20% test split — pick any model below to inspect it.")

    res = model_bundle["results_df"].copy()
    res_display = res.set_index("Model").style.format("{:.3f}").highlight_max(
        axis=0, color="rgba(0,229,255,0.25)"
    )
    st.dataframe(res_display, use_container_width=True)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    radar = go.Figure()
    for _, r in res.iterrows():
        radar.add_trace(go.Scatterpolar(
            r=[r[m] for m in metric_cols] + [r[metric_cols[0]]],
            theta=metric_cols + [metric_cols[0]],
            name=r["Model"], fill="toself", opacity=0.55,
        ))
    radar.update_layout(
        template="futuristic_dark", title="All six models, every metric at a glance",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=460, legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    st.plotly_chart(radar, use_container_width=True)
    st.caption(
        "Interactive — click a model name in the legend to isolate or hide it and compare "
        "shapes directly."
    )

    st.markdown("### Inspect a single model")
    model_names = list(model_bundle["fitted"].keys())
    default_ix = model_names.index(model_bundle["best_name"])
    chosen_model = st.selectbox(
        "Model to inspect (confusion matrix, classification report, ROC placement)",
        model_names, index=default_ix,
    )
    is_best = chosen_model == model_bundle["best_name"]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for name, (fpr, tpr) in model_bundle["curves"].items():
            auc_val = res.loc[res["Model"] == name, "ROC-AUC"].values[0]
            is_chosen = name == chosen_model
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})",
                line=dict(width=4 if is_chosen else 1.5, dash=None if is_chosen else "dot"),
                opacity=1.0 if is_chosen else 0.45,
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED),
                                  name="Chance"))
        fig.update_layout(template="futuristic_dark", title="ROC curves — selected model highlighted",
                           xaxis_title="False positive rate",
                           yaxis_title="True positive rate", height=430)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cm = (model_bundle["confusion_matrix"] if is_best
              else confusion_matrix(model_bundle["y_test"], model_bundle["preds"][chosen_model]))
        fig2 = px.imshow(cm, text_auto=True, color_continuous_scale=[PANEL, CYAN],
                          x=["Predicted: No", "Predicted: Yes"], y=["Actual: No", "Actual: Yes"],
                          title=f"Confusion matrix — {chosen_model}"
                                f"{' (best by ROC-AUC)' if is_best else ''}")
        fig2.update_layout(template="futuristic_dark", height=430, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(importance_df.head(12).sort_values("Importance"), x="Importance", y="Feature",
                  orientation="h", color="Importance", color_continuous_scale=[GRID, CYAN],
                  title="Feature importance — Random Forest (top 12)")
    fig3.update_layout(template="futuristic_dark", height=480, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    rep = (model_bundle["report"] if is_best
           else classification_report(model_bundle["y_test"], model_bundle["preds"][chosen_model],
                                       target_names=["No", "Yes"], output_dict=True))
    rep_df = pd.DataFrame(rep).T.loc[["No", "Yes"], ["precision", "recall", "f1-score", "support"]]
    st.markdown(f"#### Classification report — {chosen_model}")
    st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                       "f1-score": "{:.3f}", "support": "{:.0f}"}),
                 use_container_width=True)

    impact(
        f"{model_bundle['best_name']} correctly separates treatment-seekers from non-seekers "
        f"{best_auc:.1%} of the time (ROC-AUC) using only pre-existing survey questions — strong "
        "enough to sit behind a lightweight, explainable HR screening or awareness tool, not a "
        "clinical diagnostic. Switch the selector above to see how the add-on models "
        "(SVM, KNN, Neural Network) trade off against it."
    )

# ======================================================================================
# PAGE: LIVE PREDICTION
# ======================================================================================
elif page == "🎯 Try the Prediction Tool":
    st.markdown("## 🎯 Try the Prediction Tool")
    st.caption(
        "Enter a hypothetical respondent's profile and see what the model predicts. "
        "This mirrors the exact feature set and encoding used to train the models — "
        "for demonstration purposes, not a clinical or diagnostic tool."
    )

    model_names = list(model_bundle["fitted"].keys())
    verdict_model = st.selectbox(
        "Model driving the verdict below (all six still run and are shown in the comparison chart)",
        model_names, index=model_names.index(model_bundle["best_name"]),
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 15, 75, 30)
            gender = st.selectbox("Gender", sorted(df_clean["Gender"].unique()))
            self_employed = st.selectbox("Self-employed?", sorted(df_clean["self_employed"].unique()))
            family_history = st.selectbox("Family history of mental illness?",
                                           sorted(df_clean["family_history"].unique()))
            work_interfere = st.selectbox("If applicable, condition interferes with work",
                                           ["Never", "Rarely", "Sometimes", "Often", "Not applicable"])
            no_employees = st.selectbox("Company size",
                                         ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
            remote_work = st.selectbox("Works remotely ≥50% of the time?", sorted(df_clean["remote_work"].unique()))
        with c2:
            tech_company = st.selectbox("Employer is a tech company?", sorted(df_clean["tech_company"].unique()))
            benefits = st.selectbox("Employer provides mental-health benefits?",
                                     ["Yes", "No", "Don't know"])
            care_options = st.selectbox("Aware of employer's care options?", ["Yes", "No", "Not sure"])
            wellness_program = st.selectbox("Wellness program discussed?", ["Yes", "No", "Don't know"])
            seek_help = st.selectbox("Employer provides resources to seek help?", ["Yes", "No", "Don't know"])
            anonymity = st.selectbox("Anonymity protected if seeking treatment?", ["Yes", "No", "Don't know"])
            leave = st.selectbox("Ease of taking mental-health leave",
                                  ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])
        with c3:
            mental_health_consequence = st.selectbox("Fears consequence of disclosing (mental)?",
                                                       ["Yes", "No", "Maybe"])
            phys_health_consequence = st.selectbox("Fears consequence of disclosing (physical)?",
                                                    ["Yes", "No", "Maybe"])
            coworkers = st.selectbox("Comfortable discussing with coworkers?", ["Yes", "No", "Some of them"])
            supervisor = st.selectbox("Comfortable discussing with supervisor?", ["Yes", "No", "Some of them"])
            mental_health_interview = st.selectbox("Would raise mental health in an interview?",
                                                     ["Yes", "No", "Maybe"])
            phys_health_interview = st.selectbox("Would raise physical health in an interview?",
                                                  ["Yes", "No", "Maybe"])
            mental_vs_physical = st.selectbox("Employer treats mental health = physical health?",
                                               ["Yes", "No", "Don't know"])
            obs_consequence = st.selectbox("Observed negative consequences for open coworkers?",
                                            sorted(df_clean["obs_consequence"].unique()))

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        raw_inputs = {
            "Age": age, "Gender": gender, "self_employed": self_employed, "family_history": family_history,
            "work_interfere": work_interfere, "no_employees": no_employees, "remote_work": remote_work,
            "tech_company": tech_company, "benefits": benefits, "care_options": care_options,
            "wellness_program": wellness_program, "seek_help": seek_help, "anonymity": anonymity,
            "leave": leave, "mental_health_consequence": mental_health_consequence,
            "phys_health_consequence": phys_health_consequence, "coworkers": coworkers,
            "supervisor": supervisor, "mental_health_interview": mental_health_interview,
            "phys_health_interview": phys_health_interview, "mental_vs_physical": mental_vs_physical,
            "obs_consequence": obs_consequence,
        }
        row = {}
        for col in feature_cols:
            if col == "Age":
                row[col] = age
            else:
                row[col] = label_encoders[col].transform([str(raw_inputs[col])])[0]
        X_new = pd.DataFrame([row])[feature_cols]

        probs = {}
        for name, model in model_bundle["fitted"].items():
            if name in SCALED_MODELS:
                X_use = model_bundle["scaler"].transform(X_new)
            else:
                X_use = X_new
            probs[name] = model.predict_proba(X_use)[0][1]

        chosen_prob = probs[verdict_model]
        verdict = "Likely to seek treatment" if chosen_prob >= 0.5 else "Less likely to seek treatment"

        r1, r2 = st.columns([1, 1.4])
        with r1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=chosen_prob * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": CYAN},
                       "bgcolor": PANEL,
                       "steps": [{"range": [0, 50], "color": PANEL_2},
                                 {"range": [50, 100], "color": "rgba(0,229,255,0.22)"}]},
                title={"text": f"Predicted probability ({verdict_model})"},
            ))
            fig.update_layout(template="futuristic_dark", height=300, margin=dict(t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"### {verdict}")

        with r2:
            prob_df = pd.DataFrame({"Model": list(probs.keys()), "Probability": list(probs.values())})
            color_map = {m: (CYAN if m == verdict_model else PURPLE) for m in prob_df["Model"]}
            fig2 = px.bar(prob_df, x="Model", y="Probability", color="Model",
                          color_discrete_map=color_map,
                          text_auto=".1%", title="Prediction across all six models")
            fig2.update_layout(template="futuristic_dark", showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"Verdict above uses **{verdict_model}** — the other five are shown for comparison.")

        drivers = importance_df.head(5)
        driver_notes = []
        for _, r in drivers.iterrows():
            driver_notes.append(f"**{r['Feature']}**: entered as *{raw_inputs.get(r['raw'], age if r['raw']=='Age' else '')}*")
        st.markdown("#### Top model drivers for this prediction")
        st.markdown(
            "The model's five most influential features overall — and what this respondent entered for each:\n\n"
            + "\n".join(f"- {n}" for n in driver_notes)
        )
        st.caption(
            "This is a demonstration of the same predictive pipeline built in the notebook — useful to "
            "illustrate an explainable HR screening/awareness concept, not a medical assessment."
        )

# ======================================================================================
# PAGE: TRENDS 2014 vs 2016
# ======================================================================================
elif page == "📈 2014 vs 2016 Trends":
    st.markdown("## 📈 2014 vs 2016 Trends")
    st.caption("The 2014 OSMI wave harmonized with the 2016 wave on shared questions, to check "
               "whether attitudes and treatment-seeking moved over two years.")

    n14 = (df_combined["Year"] == 2014).sum()
    n16 = (df_combined["Year"] == 2016).sum()
    st.markdown(f"**2014 respondents:** {n14:,}  |  **2016 respondents:** {n16:,}  |  "
                f"**Combined:** {len(df_combined):,}")

    yoy_treat = pd.crosstab(df_combined["Year"], df_combined["treatment"], normalize="index") * 100
    yoy_treat = yoy_treat.reset_index().melt(id_vars="Year", var_name="treatment", value_name="pct")
    fig = px.bar(yoy_treat, x="Year", y="pct", color="treatment", barmode="group",
                 color_discrete_sequence=[CORAL, TEAL], text_auto=".1f",
                 title="Treatment-seeking rate: 2014 vs 2016 (%)")
    fig.update_layout(template="futuristic_dark", xaxis=dict(tickmode="array", tickvals=[2014, 2016]))
    st.plotly_chart(fig, use_container_width=True)
    insight("Treatment-seeking rose from ~51% in 2014 to ~59% in 2016 — an ~8-point increase in "
            "just two years.")
    impact("Positive if it reflects declining stigma — but survey composition (which communities "
           "shared the link each year) could also drive part of the shift, so treat this as a "
           "directional signal rather than a precise causal estimate.")

    yoy_care = pd.crosstab(df_combined["Year"], df_combined["care_options"].fillna("Unknown"),
                            normalize="index") * 100
    yoy_care = yoy_care.reset_index().melt(id_vars="Year", var_name="care_options", value_name="pct")
    fig2 = px.bar(yoy_care, x="Year", y="pct", color="care_options", barmode="group",
                  color_discrete_sequence=PALETTE, text_auto=".1f",
                  title="Awareness of employer care options: 2014 vs 2016 (%)")
    fig2.update_layout(template="futuristic_dark", xaxis=dict(tickmode="array", tickvals=[2014, 2016]))
    st.plotly_chart(fig2, use_container_width=True)

    yoy_benefits = pd.crosstab(df_combined["Year"], df_combined["benefits"].fillna("Unknown"),
                                normalize="index") * 100
    yoy_benefits = yoy_benefits.reset_index().melt(id_vars="Year", var_name="benefits", value_name="pct")
    fig3 = px.bar(yoy_benefits, x="Year", y="pct", color="benefits", barmode="group",
                  color_discrete_sequence=PALETTE, text_auto=".1f",
                  title="Employer-provided mental-health benefits: 2014 vs 2016 (%)")
    fig3.update_layout(template="futuristic_dark", xaxis=dict(tickmode="array", tickvals=[2014, 2016]))
    st.plotly_chart(fig3, use_container_width=True)

# ======================================================================================
# PAGE: RECOMMENDATIONS & ABOUT
# ======================================================================================
elif page == "📌 Recommendations & About":
    st.markdown("## 📌 Recommendations & About")

    st.markdown("### Business objective")
    st.markdown(
        "Give tech-company HR / people-ops teams a data-backed answer to: **which workplace "
        "policies and demographic segments are associated with higher treatment-seeking, so that "
        "limited wellness budget and communication effort can be targeted where it will have the "
        "most impact** — rather than spread evenly (and thinly) across every possible intervention."
    )

    st.markdown("### Recommendations for HR / people-ops")
    st.markdown(
        """
1. **Make existing benefits and care options impossible to miss.** The awareness gap
   (`care_options` = "No"/"Not sure") is a bigger lever than adding new benefits.
2. **Protect and communicate anonymity explicitly.** Perceived anonymity is a consistent,
   independent driver of help-seeking — restate it in onboarding and every wellness comms push.
3. **Train managers on `leave` friction.** "Somewhat/very difficult" mental-health leave
   correlates with lower treatment-seeking; clarifying the process removes a real barrier.
4. **Target messaging using the model's top drivers**, not generic company-wide campaigns —
   family history and existing symptom interference are the strongest signals, so screening-style
   nudges (opt-in, anonymous) are more efficient than blanket outreach.
5. **Re-run this pipeline on your own engagement survey.** The cleaning and modeling steps
   generalize to any employer's internal mental-health pulse survey with similar questions.
        """
    )

    st.markdown("### Tech stack")
    st.markdown(
        "Python · pandas / numpy for cleaning · scikit-learn (Logistic Regression, Random Forest, "
        "Gradient Boosting, SVM, KNN, MLP Neural Network) · Plotly for interactive visuals · "
        "**Streamlit** for this deployment."
    )

    st.markdown("### Attribution")
    st.markdown(
        """
- **Author:** P Suman Sangeet
- **Program:** LABMENTIX Data Analytics and AI
- **Source notebook:** `Mental_Health_in_Tech_Survey_EDA_and_ML.ipynb`
- **Dataset:** [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
  (2014) and its 2016 companion wave
        """
    )
    st.caption(
        "This survey is self-selected and community-shared, so results should be read as "
        "directional signal about the respondent population, not a representative population estimate."
    )
