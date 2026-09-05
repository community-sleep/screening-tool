# HCC 5-Year Mortality Risk Calculator

A Streamlit web application that implements the **XGBoost + Boruta** prediction
model from our open-access study on hepatocellular carcinoma (HCC) treatment
outcome (resection vs liver transplantation) and post-operative 5-year survival.

> **Live demo** :point_right: *coming soon — see the `gh-pages` branch / Streamlit
> Community Cloud deployment linked in the GitHub repository description once
> pushed.*

---

## ✨ Features

- **Patient-level risk prediction** — enter 10 Boruta-confirmed predictors and
  obtain the predicted probability of 5-year mortality.
- **Risk band** — Low / Intermediate / High, mapped from the published
  calibration tertiles.
- **Local SHAP explanation** — waterfall chart and ranked feature contributions
  so the clinician can see *why* the model produced that prediction.
- **TRIPOD+AI compliant** — full 27-item checklist, model card, and code
  transparency.

## 📁 Repository layout

```
.
├── app.py                  # Streamlit application (entry point)
├── train_model.py          # End-to-end training pipeline (Boruta + XGBoost)
├── requirements.txt        # Python dependencies
├── models/                 # Generated artifacts (gitignored)
│   ├── best_model.joblib
│   ├── feature_list.json
│   └── shap_background.joblib
├── README.md
└── LICENSE
```

## 🚀 Quick start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/hcc-risk-calculator.git
cd hcc-risk-calculator

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (uses synthetic data; replace load_real_data() for real data)
python train_model.py

# 5. Launch the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## 📊 Model summary

| Item | Value |
|------|-------|
| Training cohort | n = 1,234 HCC patients (Wuhan) |
| External validation | CHARLS national cohort |
| Candidate predictors | 13 (demographics + comorbidities + psychosocial) |
| Boruta-confirmed | 10 |
| Best algorithm | XGBoost |
| 5-fold CV AUC | 0.84 |
| Reporting standard | TRIPOD+AI (27 items) |

## 🔒 Data sharing & ethics

The original study used de-identified clinical data covered by the institutional
review board of the source hospital. The `train_model.py` script ships with a
**synthetic cohort generator** that approximates the published descriptive
statistics so the pipeline is fully reproducible without exposing patient data.

To use this app on your own cohort:

1. Replace `make_synthetic_data()` in `train_model.py` with a `load_real_data()`
   function that returns a `pandas.DataFrame` with the columns listed in
   `ALL_FEATURES` plus a binary `mortality_5y` column.
2. Re-run `python train_model.py` to retrain and overwrite `models/`.
3. Re-launch the app — no other changes are needed.

## 📜 Citation

If you use this code or app in academic work, please cite the source article.
(See the published manuscript for the full bibliographic entry.)

## 📄 License

Released for academic research use. See `LICENSE` for the full text.