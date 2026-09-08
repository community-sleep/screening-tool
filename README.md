# Sleep Disorder Risk Screening Tool

A Streamlit web application implementing the **XGBoost** prediction model from our
study on machine-learning-based prediction of sleep-disturbance risk among
community-dwelling middle-aged and older adults.

---

## ✨ Features

- **Individual risk prediction** — enter 13 routine community-health-survey
  predictors and obtain the predicted probability of sleep disturbance.
- **Three-tier risk stratification** — Low (<0.25), Intermediate (0.25–0.40), and
  High (>0.40) risk bands, each mapped to recommended community healthcare
  actions from the published analysis.
- **SHAP explanation** — ranked feature contributions and a text waterfall so the
  user can see *why* the model produced that prediction.
- **TRIPOD+AI compliant** — the source article reports the full 27-item
  TRIPOD+AI checklist, model card, and open code.

---

## 📁 Repository layout

```
.
├── app.py                         # Streamlit application (entry point)
├── train_model.py                 # End-to-end training pipeline
├── requirements.txt               # Python dependencies
├── models/
│   ├── xgboost_sleep_model.pkl    # Trained XGBoost model (real cohort)
│   ├── feature_list.json          # Final 13 predictors and model metadata
│   └── shap_background.joblib     # Background sample for SHAP (optional)
├── README.md
└── LICENSE
```

## 🚀 Quick start

```bash
# 1. Clone the repository
git clone https://github.com/18108666570/screening-tool.git
cd screening-tool

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

---

## 📊 Model summary

| Item | Value |
|------|-------|
| Training cohort | Community health survey, Wuhan (n = 15,755) |
| External validation | CHARLS 2018 (n = 13,472) |
| Candidate predictors | 13 (demographics, comorbidities, psychosocial) |
| Final model features | 13 (all candidate predictors retained by LASSO, multivariable logistic regression, and Boruta) |
| Algorithms compared | RF, XGBoost, SVM, KNN, MLP, LR, AdaBoost, GBM |
| Best model | XGBoost |
| Training AUC | 0.922 (95% CI 0.917–0.927) |
| Internal validation AUC | 0.895 (95% CI 0.886–0.905) |
| External validation AUC | 0.802 (95% CI 0.794–0.810) |
| External Brier score | 0.181 (logistic recalibration 0.168, improvement 6.8%) |
| Reporting standard | TRIPOD+AI (27 items) |

---

## 🔒 Data sharing & ethics

The original study used de-identified community health survey data. Because
individual-level data cannot be shared publicly, this repository ships the
**trained XGBoost model artifact** (rather than raw participant data) and a
reproducible training pipeline. `train_model.py` demonstrates the full pipeline
with a synthetic cohort generator; replace `make_synthetic_data()` with your own
`load_real_data()` function if you are running the pipeline on an
IRB-approved cohort.

For research deployment, you may replace `models/xgboost_sleep_model.pkl` with an
artifact trained on your own ethically approved cohort; the app input schema and
feature order are documented in `models/feature_list.json`.

---

## 📜 Citation

If you use this code or app in academic work, please cite the source article.
(See the published manuscript for the full bibliographic entry.)

---

## 📄 License

Released for academic research use. See `LICENSE` for the full text.
