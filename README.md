# Sleep Disorder Risk Screening Tool

A Streamlit web application implementing the **XGBoost + Boruta** prediction
model from our study on machine learning-based prediction of sleep disorder
risk among community-dwelling middle-aged and older adults.

---

## ✨ Features

- **Individual risk prediction** — enter 10 routine questionnaire predictors
  and obtain the predicted probability of sleep disorder.
- **Risk stratification** — Low / Intermediate / High bands mapped from the
  published calibration.
- **SHAP explanation** — ranked feature contributions and a text waterfall so
  the user can see *why* the model produced that prediction.
- **TRIPOD+AI compliant** — the source article reports the full 27-item
  checklist, model card, and code transparency.

## 📁 Repository layout

```
.
├── app.py                  # Streamlit application (entry point)
├── train_model.py          # End-to-end training pipeline (Boruta + XGBoost)
├── requirements.txt        # Python dependencies
├── models/
│   ├── best_model.joblib   # Trained XGBoost pipeline
│   ├── feature_list.json   # 13 candidate / Boruta / final features
│   └── shap_background.joblib
├── README.md
└── LICENSE
```

## 🚀 Quick start

```bash
# 1. Clone the repository
git clone https://github.com/18108666570/screening-tool.git
cd screening-tool

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Retrain the model (optional; a pre-trained artifact ships in models/)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## 📊 Model summary

| Item | Value |
|------|-------|
| Training cohort | Community health survey, Wuhan (n = 15,755) |
| External validation | CHARLS 2018 (n = 13,472) |
| Candidate predictors | 13 (demographics + comorbidities + psychosocial) |
| Final model features | 10 |
| Algorithms compared | RF, XGBoost, SVM, KNN, MLP, LR, AdaBoost, GBM |
| Best model | XGBoost |
| Training AUC | 0.922 (95% CI 0.917–0.927) |
| Internal validation AUC | 0.895 (95% CI 0.886–0.905) |
| External validation AUC | 0.802 (95% CI 0.794–0.810) |
| Brier score (external, recalibrated) | 0.181 → 0.152 |
| Reporting standard | TRIPOD+AI (27 items) |

## 🔒 Data sharing & ethics

The original study used de-identified community health survey data covered by
the institutional review board of the source institutions; individual-level
data cannot be shared publicly. `train_model.py` ships with a **synthetic
cohort generator** that approximates the published descriptive statistics so
the pipeline is fully reproducible without exposing participant data. The
pre-trained artifact in `models/` was produced with this synthetic cohort and
is intended as a demonstration; replace it with artifacts trained on the
IRB-approved cohort for research deployment.

To use this pipeline on your own cohort:

1. Replace `make_synthetic_data()` in `train_model.py` with a
   `load_real_data()` function that returns a `pandas.DataFrame` with the
   columns listed in `ALL_FEATURES` plus a binary `sleep_disorder` column.
2. Re-run `python train_model.py` to retrain and overwrite `models/`.
3. Re-launch the app — no other changes are needed.

## 📜 Citation

If you use this code or app in academic work, please cite the source article.
(See the published manuscript for the full bibliographic entry.)

## 📄 License

Released for academic research use. See `LICENSE` for the full text.
