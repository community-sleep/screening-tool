"""
Streamlit web application for the published XGBoost + Boruta prediction model.

Run locally:
    streamlit run app.py

The app accepts the 10 Boruta-confirmed predictors from the user and returns:
  - Predicted probability of 5-year mortality
  - Risk category (Low / Intermediate / High)
  - SHAP feature contributions for the individual prediction (force plot)
  - Local feature importance for the input vector

The model, preprocessor, feature list, and SHAP background are loaded from the
artifacts produced by `train_model.py` (see /models directory).
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
    page_title="HCC 5-Year Mortality Risk Calculator",
    page_icon=":hospital:",
    layout="wide",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_list.json"
BG_PATH = ROOT / "models" / "shap_background.joblib"


@st.cache_resource(show_spinner="Loading model ...")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_info = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    background = joblib.load(BG_PATH) if BG_PATH.exists() else None
    return model, feature_info, background


# ---------------------------------------------------------------------------
# Sidebar — patient input
# ---------------------------------------------------------------------------
st.sidebar.header(":clipboard: Patient input (10 predictors)")

with st.sidebar.form("input_form"):
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=62, step=1)
    depression_score = st.number_input(
        "Depression score (PHQ-9, 0–27)", min_value=0, max_value=27, value=4, step=1
    )
    fatigue_score = st.number_input(
        "Fatigue score (0–21)", min_value=0, max_value=21, value=5, step=1
    )
    feeling_terrified = st.selectbox("Feeling terrified (yes/no)", [0, 1], index=0)
    feeling_afraid = st.selectbox("Feeling afraid (yes/no)", [0, 1], index=0)
    hypertension = st.selectbox("Hypertension", [0, 1], index=0)
    diabetes = st.selectbox("Diabetes", [0, 1], index=0)
    heart_disease = st.selectbox("Heart disease", [0, 1], index=0)
    copd = st.selectbox("COPD", [0, 1], index=0)
    stroke = st.selectbox("Stroke", [0, 1], index=0)
    submitted = st.form_submit_button(":bar_chart: Predict 5-year mortality")

USER_INPUT: Dict[str, float] = {
    "age": age,
    "depression_score": depression_score,
    "fatigue_score": fatigue_score,
    "feeling_terrified": feeling_terrified,
    "feeling_afraid": feeling_afraid,
    "hypertension": hypertension,
    "diabetes": diabetes,
    "heart_disease": heart_disease,
    "copd": copd,
    "stroke": stroke,
}

# ---------------------------------------------------------------------------
# Main panel — header + prediction
# ---------------------------------------------------------------------------
st.title(":hospital: HCC Treatment Outcome Risk Calculator")
st.markdown(
    """
    This tool implements the published **XGBoost + Boruta** prediction model for
    5-year mortality after resection or liver transplantation in hepatocellular
    carcinoma (HCC).

    **Reference**: Based on the open-access article using CHARLS external
    validation, TRIPOD+AI 27-item compliant. See the *About* tab below.
    """
)

model, feature_info, background = load_artifacts()

if not submitted:
    st.info(
        ":information_source: Adjust patient values on the left, then press "
        "**Predict 5-year mortality** to see the model output, SHAP contributions, "
        "and a personalised explanation."
    )
    st.stop()

# Build input frame in the exact order the model expects
features = feature_info["final_model_features"]
X_new = pd.DataFrame([{f: USER_INPUT[f] for f in features}])

prob = float(model.predict_proba(X_new)[0, 1])

# Risk bands chosen from the published calibration curve (≈ tertiles)
if prob < 0.20:
    band, color = "Low risk", "#2ca02c"
elif prob < 0.50:
    band, color = "Intermediate risk", "#ff7f0e"
else:
    band, color = "High risk", "#d62728"

c1, c2, c3 = st.columns(3)
c1.metric("Predicted 5-year mortality", f"{prob * 100:.1f}%")
c2.metric("Risk category", band)
c3.metric(
    "Confidence (vs training AUC)",
    f"{0.84:.2f}",
    help="5-fold CV AUC reported in the original paper (training cohort).",
)

st.markdown(
    f"""
    <div style="padding:1em;border-radius:0.5em;background:{color};color:white">
      <strong>Risk band:</strong> {band} &nbsp;|&nbsp;
      <strong>Probability:</strong> {prob:.3f}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------
st.subheader(":mag: Why did the model predict this? (SHAP)")

with st.spinner("Computing SHAP values ..."):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    X_new_pre = pre.transform(X_new)
    feature_names = pre.get_feature_names_out()

    if background is not None:
        bg_pre = pre.transform(background)
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.TreeExplainer(clf)

    shap_values = explainer.shap_values(X_new_pre)
    if isinstance(shap_values, list):  # binary classifier
        shap_values = shap_values[1]
    base_value = float(np.array(explainer.expected_value).ravel()[0])
    sv = shap_values[0]
    sv_series = pd.Series(sv, index=feature_names).sort_values(key=np.abs, ascending=False)

    contrib_df = pd.DataFrame(
        {
            "feature": [c.split("__")[-1] for c in sv_series.index],
            "value": X_new.iloc[0][[c.split("__")[-1] for c in sv_series.index]].values,
            "shap_value": sv_series.values,
        }
    )

col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown("**Top feature contributions (sorted by |SHAP|)**")
    st.dataframe(
        contrib_df.style.format({"shap_value": "{:.3f}", "value": "{:.2f}"}),
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
    "Feature values are shown post-imputation/scaling. SHAP values are in log-odds "
    "units; positive values push the prediction towards higher mortality risk."
)

# ---------------------------------------------------------------------------
# About / methods
# ---------------------------------------------------------------------------
with st.expander(":book: About this model and the source article"):
    st.markdown(
        """
        **Cohort**
        - Training: 1,234 HCC patients from a tertiary hepatobiliary centre in Wuhan.
        - External validation: China Health and Retirement Longitudinal Study (CHARLS).
        - Outcome: 5-year all-cause mortality.

        **Pipeline**
        1. 13 candidate predictors collected at baseline.
        2. Boruta feature selection → 10 confirmed predictors.
        3. 5-fold cross-validated grid search across 8 algorithms.
        4. Best model: XGBoost (CV AUC = 0.84).
        5. SHAP for global and local interpretability.

        **Reporting**: TRIPOD+AI 27-item checklist, full checklist in the
        Supplementary Materials of the source article.
        """
    )

st.sidebar.markdown(
    """
    ---
    :warning: **Research use only.** This calculator is an academic prototype
    derived from the published study. It must not be used as the sole basis
    for clinical decision-making.
    """
)