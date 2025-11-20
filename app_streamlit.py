import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import datetime as dt

# ---------------------------------------------
# Logging setup
# ---------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------
# Defensive joblib import
# ---------------------------------------------
try:
    import joblib
except Exception as e:
    logger.exception("joblib import failed")
    st.error(
        "❌ Critical: `joblib` is missing.\n\n"
        "Fix: Ensure `joblib==1.4.2` is listed in requirements.txt and redeploy."
    )
    st.stop()

# ---------------------------------------------
# Page config
# ---------------------------------------------
st.set_page_config(
    page_title="Uber Trip Demand — Enterprise Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------
# Header
# ---------------------------------------------
st.markdown(
    """
    <h1 style="font-size:38px; font-weight:700;">🚖 Uber Trip Demand Forecast — Enterprise Dashboard</h1>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------
# Load model safely
# ---------------------------------------------
MODEL_PATH = "models/best_model_gradient_boosting.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file missing at: `models/best_model_gradient_boosting.pkl`.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.success("Model loaded ✔")

# ---------------------------------------------
# Sidebar controls
# ---------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload historical CSV (optional)",
    type=["csv"],
    help="CSV must contain: date, trips, active_vehicles",
)

# ---------------------------------------------
# Load data
# ---------------------------------------------
@st.cache_data
def load_default():
    df = pd.read_csv("Data/Uber-Jan-Feb-FOIL.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])
    st.sidebar.info("Uploaded dataset loaded ✔")
else:
    df = load_default()
    st.sidebar.info("Loaded default historical dataset.")

# ---------------------------------------------
# Feature Engineering Function
# ---------------------------------------------
def engineer_features(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")
    data["day_of_week"] = data["date"].dt.day_name()
    data["is_weekend"] = data["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["trips_rolling_mean_3"] = data["trips"].rolling(3).mean()
    data["trips_rolling_mean_7"] = data["trips"].rolling(7).mean()
    data["lag_1"] = data["trips"].shift(1)
    data["lag_2"] = data["trips"].shift(2)
    data["lag_3"] = data["trips"].shift(3)
    return data.dropna()

df_feat = engineer_features(df)

# ---------------------------------------------
# Prediction Controls
# ---------------------------------------------
st.subheader("Run Prediction")

col1, col2 = st.columns(2)

with col1:
    pred_date = st.date_input(
        "Prediction date",
        value=dt.date(2015, 3, 1)
    )

with col2:
    active_override = st.number_input(
        "Active vehicles (0 = use last known)",
        value=0,
        min_value=0,
        step=1,
    )

# ---------------------------------------------
# Predict Button
# ---------------------------------------------
if st.button("Predict", use_container_width=True):

    st.success("Features ready — results shown on the right.")

    last_row = df_feat.iloc[-1].copy()
    new_row = last_row.copy()

    new_row["date"] = pd.to_datetime(pred_date)
    new_row["day_of_week"] = new_row["date"].day_name()
    new_row["is_weekend"] = int(new_row["day_of_week"] in ["Saturday", "Sunday"])
    new_row["month"] = new_row["date"].month
    new_row["day"] = new_row["date"].day

    if active_override > 0:
        new_row["active_vehicles"] = active_override

    feature_cols = [
        "is_weekend", "month", "day",
        "trips_rolling_mean_3", "trips_rolling_mean_7",
        "lag_1", "lag_2", "lag_3"
    ]

    X_new = pd.DataFrame([new_row[feature_cols]])
    pred = model.predict(X_new)[0]

    st.metric("Predicted Trips", f"{pred:,.0f}")

    # ---------------------------------------------
    # SHAP Explanation
    # ---------------------------------------------
    with st.expander("🔍 SHAP Explanation"):

        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_new)
            shap_fig = shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value,
                shap_vals[0],
                feature_names=X_new.columns
            )
            st.pyplot(shap_fig)
        except:
            st.warning("SHAP unavailable — fallback shown.")
            st.image("shap_summary_bar.png")

# Footer
st.markdown("---")
st.caption("Enterprise Forecasting Dashboard — Built by Tabish Deshmukh")