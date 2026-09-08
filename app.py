"""
Streamlit web application for the published XGBoost prediction model.

Run locally:
    streamlit run app.py

The app accepts the 13 model predictors from the user and returns:
  - Predicted probability of sleep disturbance
  - Three-tier risk stratification (Low <0.25 / Intermediate 0.25-0.40 / High >0.40)
    with recommended community healthcare actions
  - SHAP feature contributions for the individual prediction

The model artifact in /models was trained on the de-identified Wuhan community
health survey cohort (n = 15,755) described in the source article.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Disorder Risk Screening Tool",
    page_icon=":bed:",
    layout="wide",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "xgboost_sleep_model.pkl"
FEATURES_PATH = ROOT / "models" / "feature_list.json"


@st.cache_resource(show_spinner="Loading model ...")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_info = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return model, feature_info


# ---------------------------------------------------------------------------
# Three-tier risk stratification (Table 4 of the source article)
# ---------------------------------------------------------------------------
RISK_TIERS = [
    {
        "name": "Low Risk",
        "range": "Predicted probability < 0.25",
        "action": "Village health station: sleep hygiene posters; online educational materials",
        "burden": "Low (population-level)",
        "color": "#2ca02c",
    },
    {
        "name": "Intermediate Risk",
        "range": "Predicted probability 0.25 - 0.40",
        "action": "Family physician monthly follow-up; brief CBT-I self-help materials",
        "burden": "Moderate (targeted)",
        "color": "#ff7f0e",
    },
    {
        "name": "High Risk",
        "range": "Predicted probability > 0.40",
        "action": "Referral to county-level sleep clinic; comprehensive evaluation",
        "burden": "High (specialized)",
        "color": "#d62728",
    },
]


def assign_tier(prob: float):
    if prob < 0.25:
        return RISK_TIERS[0]
    if prob <= 0.40:
        return RISK_TIERS[1]
    return RISK_TIERS[2]


# ---------------------------------------------------------------------------
# Demo shortcut for eFigure 1 screenshots
# ---------------------------------------------------------------------------
# Preset inputs so the sidebar form and the main-panel output stay in sync
# when the app is opened with ?demo=high or ?demo=low.
DEMO_LIST = st.query_params.get_all("demo")
DEMO = DEMO_LIST[0] if DEMO_LIST else None

if DEMO == "high":
    # 70-year-old male with depression, fatigue, hypertension, diabetes
    _demo_defaults = {
        "Age": 70,
        "Sex": 1,              # Male
        "Smoke": 1,            # Yes
        "Depression": 1,       # Yes
        "Fatigue": 1,          # Yes
        "Terrified": 0,        # No
        "Afraid": 0,           # No
        "Hypertension": 1,     # Yes
        "Diabetes": 1,         # Yes
        "Dyslipidemia": 0,     # No
        "Cardiopathy": 0,      # No
        "COPD": 0,             # No
        "Stroke": 0,           # No
    }
elif DEMO == "low":
    # 65-year-old female without any risk factors
    _demo_defaults = {
        "Age": 65,
        "Sex": 0,              # Female
        "Smoke": 0,            # No
        "Depression": 0,       # No
        "Fatigue": 0,          # No
        "Terrified": 0,        # No
        "Afraid": 0,           # No
        "Hypertension": 0,     # No
        "Diabetes": 0,         # No
        "Dyslipidemia": 0,     # No
        "Cardiopathy": 0,      # No
        "COPD": 0,             # No
        "Stroke": 0,           # No
    }
else:
    _demo_defaults = None


def _default(name: str):
    """Return the default value/index for a given predictor.

    If a demo preset exists, use it; otherwise use the ordinary default.
    """
    ordinary = {
        "Age": 65,
        "Sex": 0,          # Female
        "Smoke": 0,        # No
        "Depression": 0,   # No
        "Fatigue": 0,      # No
        "Terrified": 0,    # No
        "Afraid": 0,       # No
        "Hypertension": 0, # No
        "Diabetes": 0,     # No
        "Dyslipidemia": 0, # No
        "Cardiopathy": 0,  # No
        "COPD": 0,         # No
        "Stroke": 0,       # No
    }
    if _demo_defaults is not None:
        return int(_demo_defaults[name])
    return ordinary[name]


# ---------------------------------------------------------------------------
# Sidebar - individual input (13 predictors)
# ---------------------------------------------------------------------------
st.sidebar.header(":clipboard: Individual input (13 predictors)")

with st.sidebar.form("input_form"):
    age = st.number_input("Age (years)", min_value=45, max_value=100, value=_default("Age"), step=1)
    male = st.selectbox("Sex", ["Female", "Male"], index=_default("Sex")) == "Male"
    current_smoking = st.selectbox("Current smoking", ["No", "Yes"], index=_default("Smoke")) == "Yes"
    depression = st.selectbox("Depression", ["No", "Yes"], index=_default("Depression")) == "Yes"
    fatigue = st.selectbox("Fatigue", ["No", "Yes"], index=_default("Fatigue")) == "Yes"
    feeling_terrified = st.selectbox("Feeling terrified", ["No", "Yes"], index=_default("Terrified")) == "Yes"
    feeling_afraid = st.selectbox("Feeling afraid", ["No", "Yes"], index=_default("Afraid")) == "Yes"
    hypertension = st.selectbox("Hypertension", ["No", "Yes"], index=_default("Hypertension")) == "Yes"
    diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=_default("Diabetes")) == "Yes"
    dyslipidemia = st.selectbox("Dyslipidemia", ["No", "Yes"], index=_default("Dyslipidemia")) == "Yes"
    heart_disease = st.selectbox("Heart disease", ["No", "Yes"], index=_default("Cardiopathy")) == "Yes"
    copd = st.selectbox("Chronic obstructive pulmonary disease (COPD)", ["No", "Yes"], index=_default("COPD")) == "Yes"
    stroke = st.selectbox("Stroke", ["No", "Yes"], index=_default("Stroke")) == "Yes"
    submitted = st.form_submit_button(":bar_chart: Predict sleep disorder risk")

USER_INPUT: Dict[str, float] = {
    "Age": age,
    "Sex": 1 if male else 0,
    "Smoke": 1 if current_smoking else 0,
    "Depression": 1 if depression else 0,
    "Fatigue": 1 if fatigue else 0,
    "Terrified": 1 if feeling_terrified else 0,
    "Afraid": 1 if feeling_afraid else 0,
    "Hypertension": 1 if hypertension else 0,
    "Diabetes": 1 if diabetes else 0,
    "Dyslipidemia": 1 if dyslipidemia else 0,
    "Cardiopathy": 1 if heart_disease else 0,
    "COPD": 1 if copd else 0,
    "Stroke": 1 if stroke else 0,
}

# Demo mode bypasses the form submit button and shows the prediction immediately
if DEMO in ("high", "low"):
    submitted = True

# ---------------------------------------------------------------------------
# Main panel - header
# ---------------------------------------------------------------------------
st.title(":bed: Sleep Disorder Risk Screening Tool")
st.markdown(
    """
    This tool implements the **XGBoost** prediction model for sleep disturbance
    risk among community-dwelling middle-aged and older adults, externally
    validated in the China Health and Retirement Longitudinal Study (CHARLS).
    """
)

model, feature_info = load_artifacts()
features = feature_info["final_model_features"]

if not submitted:
    st.info(
        ":information_source: Adjust individual values on the left (13 predictors), "
        "then press **Predict sleep disorder risk** to see the predicted probability, "
        "the three-tier risk stratification with recommended community healthcare "
        "actions, and SHAP feature contributions."
    )
    st.stop()

# Build input frame in the exact order the model expects
X_new = pd.DataFrame([{f: USER_INPUT[f] for f in features}])

prob = float(model.predict_proba(X_new)[0, 1])
tier = assign_tier(prob)

# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------
st.subheader(":bar_chart: Predicted risk")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted probability of sleep disturbance", f"{prob * 100:.1f}%")
c2.metric("Risk category", tier["name"])
c3.metric(
    "Model performance (validation)",
    "AUC 0.895 / 0.802",
    help="Internal validation AUC = 0.895 (5-fold cross-validation within the "
    "development cohort); external validation AUC = 0.802 in CHARLS 2018, "
    "as reported in the source article.",
)

# ---------------------------------------------------------------------------
# Three-tier risk stratification display
# ---------------------------------------------------------------------------
st.subheader(":triangular_ruler: Three-tier risk stratification and recommended community action")

for t in RISK_TIERS:
    highlight = " (this individual)" if t["name"] == tier["name"] else ""
    border = f"2px solid {t['color']}" if t["name"] == tier["name"] else "1px solid #cccccc"
    st.markdown(
        f"""
        <div style="padding:0.8em 1em;margin:0.4em 0;border-radius:0.5em;
                    border:{border};background:#f9f9f9">
          <span style="color:{t['color']};font-weight:bold;font-size:1.05em">
            {t['name']}{highlight}</span>
          &nbsp;|&nbsp; {t['range']} &nbsp;|&nbsp; Resource burden: {t['burden']}
          <br><strong>Recommended community action:</strong> {t['action']}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------
st.subheader(":mag: Why did the model predict this? (SHAP)")

with st.spinner("Computing SHAP values ..."):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_new)
    if isinstance(shap_values, list):  # binary classifier
        shap_values = shap_values[1]
    base_value = float(np.array(explainer.expected_value).ravel()[0])
    sv = shap_values[0]
    sv_series = pd.Series(sv, index=features).sort_values(key=np.abs, ascending=False)

    contrib_df = pd.DataFrame(
        {
            "feature": sv_series.index,
            "value": [USER_INPUT[f] for f in sv_series.index],
            "shap_value": sv_series.values,
        }
    )

col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown("**Top feature contributions (sorted by |SHAP|)**")
    st.dataframe(
        contrib_df.style.format({"shap_value": "{:.3f}", "value": "{:.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with col_right:
    st.markdown("**SHAP waterfall (text view)**")
    text_rows = [f"base value = {base_value:.3f}"]
    running = base_value
    for _, row in contrib_df.iterrows():
        running = running + row["shap_value"]
        sign = "+" if row["shap_value"] >= 0 else "−"
        text_rows.append(
            f"{sign} {row['feature']} = {row['shap_value']:+.3f} "
            f"→ running = {running:.3f}"
        )
    text_rows.append(f"final log-odds = {running:.3f}")
    st.code("\n".join(text_rows), language="text")

st.caption(
    "SHAP values are in log-odds units; positive values push the prediction "
    "towards higher sleep-disturbance risk. Feature coding: Sex 1 = male, 0 = "
    "female; all binary predictors 1 = present, 0 = absent."
)

# ---------------------------------------------------------------------------
# About / methods
# ---------------------------------------------------------------------------
with st.expander(":book: About this model and the source article"):
    st.markdown(
        """
        **Cohorts**
        - Development: community health survey of middle-aged and older adults
          in Wuhan, China (n = 15,755; sleep disturbance prevalence 35.9%).
        - External validation: China Health and Retirement Longitudinal Study
          (CHARLS 2018, n = 13,472; prevalence 36.0%).

        **Pipeline**
        1. 13 candidate predictors selected a priori from the geriatric sleep
           epidemiology literature; all 13 retained by penalized logistic
           regression, multivariable logistic regression, and the Boruta
           algorithm.
        2. 5-fold cross-validated hyperparameter tuning across 8 algorithms
           (RF, XGBoost, SVM, KNN, MLP, logistic regression, AdaBoost, GBM).
        3. Best model: XGBoost (training AUC = 0.922; internal validation
           AUC = 0.895; external validation AUC = 0.802).
        4. SHAP for global and local interpretability; E-value sensitivity
           analysis for unmeasured confounding.

        **Reporting**: TRIPOD+AI 27-item checklist, provided in the
        Supplementary Materials of the source article.
        """
    )

st.sidebar.markdown(
    """
    ---
    :warning: **Research use only.** This screening tool is an academic
    prototype derived from the published study. It must not be used as the
    sole basis for clinical decision-making.
    """
)
