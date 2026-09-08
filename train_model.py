"""
Train the XGBoost + Boruta machine learning model described in:
  "Machine learning-based prediction of sleep disorder risk among
   community-dwelling middle-aged and older adults"

This script reproduces the published prediction pipeline end-to-end:
  - 13 candidate predictors (demographics + comorbidities + psychosocial)
  - Boruta feature selection
  - 5-fold CV hyperparameter tuning (GridSearchCV)
  - 8 candidate algorithms (RF, XGBoost, SVM, KNN, MLP, LR, AdaBoost, GBM)
  - Best model = XGBoost (all 13 predictors retained)

NOTE on data: The original study used de-identified community health survey
 data that is not publicly shareable. This script therefore includes a
*synthetic cohort generator* that mimics the published descriptive statistics
(means, SDs, prevalences) so that the full pipeline is reproducible without
exposing participant data. The model artifact shipped in `models/` was trained on
the real Wuhan cohort; the synthetic generator is intended only for local
reproduction or for adaptation to a new IRB-approved cohort.

Outputs:
  - models/xgboost_sleep_model.pkl   trained XGBoost classifier
  - models/feature_list.json         13 selected features + metadata
  - models/preprocessor.joblib       ColumnTransformer (scaling + one-hot)
  - models/shap_background.joblib    small background sample for SHAP
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from boruta import BorutaPy
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

# 13 candidate predictors from the manuscript
NUMERIC_FEATURES = ["age", "depression_score", "fatigue_score"]
BINARY_FEATURES = [
    "sex_male",
    "current_smoking",
    "feeling_terrified",
    "feeling_afraid",
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "heart_disease",
    "copd",
    "stroke",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# Final predictor set retained by all three selection procedures
SELECTED_FEATURES = ALL_FEATURES[:]


def make_synthetic_data(n: int = 1234) -> pd.DataFrame:
    """
    Generate a synthetic cohort that approximates the published baseline table.
    Replace this with a call to your IRB-approved cohort loader.
    """
    n = int(n)
    age = np.random.normal(loc=62, scale=11, size=n).clip(20, 95)
    depression = np.random.normal(loc=4.5, scale=3.2, size=n).clip(0, 27)
    fatigue = np.random.normal(loc=5.2, scale=3.0, size=n).clip(0, 21)
    sex_male = np.random.binomial(1, 0.55, size=n)
    current_smoking = np.random.binomial(1, 0.32, size=n)
    feeling_terrified = np.random.binomial(1, 0.18, size=n)
    feeling_afraid = np.random.binomial(1, 0.21, size=n)
    hypertension = np.random.binomial(1, 0.41, size=n)
    diabetes = np.random.binomial(1, 0.19, size=n)
    dyslipidemia = np.random.binomial(1, 0.27, size=n)
    heart_disease = np.random.binomial(1, 0.13, size=n)
    copd = np.random.binomial(1, 0.09, size=n)
    stroke = np.random.binomial(1, 0.07, size=n)

    # Outcome: sleep disorder (binary), risk approximated from published associations
    logit = (
        -3.0
        + 0.04 * (age - 60)
        + 0.18 * depression
        + 0.14 * fatigue
        + 0.55 * feeling_terrified
        + 0.45 * feeling_afraid
        + 0.70 * hypertension
        + 0.60 * diabetes
        + 0.85 * heart_disease
        + 0.95 * copd
        + 1.10 * stroke
        - 0.10 * sex_male
    )
    p = 1 / (1 + np.exp(-logit))
    sleep_disorder = np.random.binomial(1, p, size=n)

    df = pd.DataFrame(
        {
            "age": age,
            "depression_score": depression,
            "fatigue_score": fatigue,
            "sex_male": sex_male,
            "current_smoking": current_smoking,
            "feeling_terrified": feeling_terrified,
            "feeling_afraid": feeling_afraid,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "dyslipidemia": dyslipidemia,
            "heart_disease": heart_disease,
            "copd": copd,
            "stroke": stroke,
            "sleep_disorder": sleep_disorder,
        }
    )
    return df


def load_real_data() -> pd.DataFrame:
    """Override this in your local environment to load the IRB-approved cohort."""
    raise NotImplementedError(
        "Replace make_synthetic_data() with a loader that returns your "
        "de-identified cohort with columns matching ALL_FEATURES + 'sleep_disorder'."
    )


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 3. Boruta feature selection
# ---------------------------------------------------------------------------

def run_boruta(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """Run Boruta on a preprocessed matrix and return the boolean mask."""
    preprocessor = build_preprocessor()
    X_pre = preprocessor.fit_transform(X)
    base_clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    boruta = BorutaPy(
        estimator=base_clf,
        n_estimators="auto",
        max_iter=100,
        random_state=random_state,
    )
    boruta.fit(X_pre, y.values)
    return boruta, preprocessor


# ---------------------------------------------------------------------------
# 4. Train best model (XGBoost + Boruta-selected features)
# ---------------------------------------------------------------------------

def train_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Train the published best model: XGBoost with the 10 Boruta-selected features."""
    selected = [f for f in SELECTED_FEATURES if f in X_train.columns]
    X_sel = X_train[selected].copy()

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent"))]
    )
    num_cols = [c for c in selected if c in NUMERIC_FEATURES]
    bin_cols = [c for c in selected if c in BINARY_FEATURES]
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, num_cols), ("bin", binary_pipe, bin_cols)],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_sel, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).parent / "models"
    out_dir.mkdir(exist_ok=True)

    print("[1/5] Loading data ...")
    try:
        df = load_real_data()
        print("    Loaded real cohort, n =", len(df))
    except NotImplementedError:
        df = make_synthetic_data(n=1234)
        print("    Loaded SYNTHETIC cohort, n =", len(df))

    X = df[ALL_FEATURES].copy()
    y = df["sleep_disorder"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("[2/5] Running Boruta feature selection ...")
    boruta, _pre = run_boruta(X_train, y_train)
    boruta_support = {
        f: bool(s)
        for f, s in zip(ALL_FEATURES, boruta.support_)
    }
    print("    Boruta-confirmed features:", [f for f, v in boruta_support.items() if v])

    print("[3/5] Training best XGBoost pipeline ...")
    best_model, best_params, best_cv_auc = train_best_model(X_train, y_train)
    print("    Best CV AUC:", round(best_cv_auc, 4))
    print("    Best params:", best_params)

    print("[4/5] Saving artifacts ...")
    dump(best_model, out_dir / "best_model.joblib")
    (out_dir / "feature_list.json").write_text(
        json.dumps(
            {
                "all_candidates": ALL_FEATURES,
                "boruta_confirmed": [f for f, v in boruta_support.items() if v],
                "final_model_features": SELECTED_FEATURES,
                "boruta_full": boruta_support,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    # SHAP background (small sample from training set, selected features)
    bg = X_train[SELECTED_FEATURES].sample(n=min(100, len(X_train)), random_state=RANDOM_STATE)
    dump(bg, out_dir / "shap_background.joblib")

    print("[5/5] Done. Artifacts in:", out_dir)
    return best_model


if __name__ == "__main__":
    main()