# Uber Trip Analysis — Demand Forecasting (Jan–Feb 2015)

**Project:** Short-term trip forecasting for Uber (sample dataset)  
**Author:** Tabish Deshmukh  
**Repository:** `Uber_Trip_Analysis` — Jupyter + Streamlit demonstration of EDA, feature engineering, ML modeling and model explainability (SHAP).

---

## Summary (Recruiter-friendly)

This project builds a robust, production-oriented pipeline to forecast daily Uber trip counts using a sample dataset (Jan–Feb 2015). It contains:
- Exploratory Data Analysis (EDA)
- Feature Engineering (rolling means, lags, weekend flag)
- Model training and evaluation (Random Forest, XGBoost, Gradient Boosting)
- Feature explainability using SHAP
- A small interactive Streamlit app to make single-day predictions and show explanations

**Key result:** Gradient Boosting delivered the best performance on the test split with:
- **MAPE:** 7.67%  
- **RMSE:** 1,505.93 
- **R²:** 0.982

This demonstrates accurate day-level demand forecasting — suitable as a baseline for capacity planning and driver allocation.

---

## Demo / Streamlit app

To run locally (recommended):
1. Ensure the repository root has:
   - `app_streamlit.py`
   - `Data/Uber-Jan-Feb-FOIL.csv`
   - `models/best_model_gradient_boosting.pkl`

2. Activate your environment and install requirements (I include a `requirements.txt` later):
   ```bash
   conda activate uber_env
   pip install -r requirements.txt
   cd Uber_Trip_Analysis
Run the Streamlit app:

bash
streamlit run app_streamlit.py
Open http://localhost:8501 in your browser (or the port Streamlit shows).

The app allows uploading a CSV (date,trips,active_vehicles), selecting a target date, overriding active_vehicles, and generating predictions + SHAP explanations.

Repository structure
Copy code
Uber_Trip_Analysis/
├─ Data/
│  └─ Uber-Jan-Feb-FOIL.csv
├─ models/
│  └─ best_model_gradient_boosting.pkl
├─ Reports/
│  └─ Executive_Summary.md
├─ 01_data_load_and_EDA.ipynb
├─ 02_feature_engineering.ipynb
├─ 03_model_building.ipynb
├─ app_streamlit.py
├─ README.md
└─ requirements.txt
How it works — quick technical steps
Load & clean: parse date, sort by date, drop NA.

Feature engineering (in 02_feature_engineering.ipynb):

day_of_week, is_weekend, month, day

rolling averages: trips_rolling_mean_3, trips_rolling_mean_7

lag features: lag_1, lag_2, lag_3

Train/test split: time-ordered split (no shuffle) — test_size=0.2.

Models trained (in 03_model_building.ipynb):

RandomForestRegressor

XGBRegressor

GradientBoostingRegressor (best)
Models evaluated using MAPE, RMSE, R².

Explainability: SHAP values computed for the best model and exported as images + interactive HTML file for local inspection.

Deployment: app_streamlit.py provides a single-day prediction UI and saves explanation outputs to Reports/.

Model performance (test set)
Model	              MAPE (%)	  RMSE	          R²
Random Forest	      9.05	      2,048.68	      0.967
XGBoost	            8.72	      1,798.10	      0.975
Gradient Boosting	  7.67	      1,505.94	      0.982

Visuals & Artifacts
shap_summary_bar.png — global feature importance (SHAP)

shap_beeswarm.png — SHAP beeswarm

shap_force_index_5.html — interactive single-sample explanation

Reports/ contains one-page summary & presentation slides used for showcasing.

Reproducibility & notes
Use the provided conda env (uber_env) or install from requirements.txt.

Model is saved via joblib into models/best_model_gradient_boosting.pkl.

Rolling means and lags require at least 7 days of history; the app falls back to last known values if needed.

Next steps (how this makes you stand out)
Expand horizon forecasting: multi-day / sequence models (Prophet or LSTM).

Evaluate per-base or geo-partitioned forecasting (spatial models).

Add automated retraining and a deployment pipeline (CI/CD).

Build dashboards for monitoring actual vs predicted and model drift alerts.

License & contact
License: MIT (add LICENSE if you want to publish)

Author: Tabish Deshmukh — deshmukhtabish4@gmail.com


