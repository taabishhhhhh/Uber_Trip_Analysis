# Save this file as app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap
from io import StringIO

st.set_page_config(page_title="Uber Trip Demand Forecast", layout="centered")

# ----- Header -----
st.markdown("<h1 style='text-align:center;color:#0A66C2'>Uber Trip Demand Forecasting — Demo</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Interactive demo that predicts daily Uber trips using the saved Gradient Boosting model. Includes SHAP explainability (inline or fallback images).</p>", unsafe_allow_html=True)
st.write("---")

# ----- Paths (adjust if needed) -----
DATA_PATH = os.path.join("Data", "Uber-Jan-Feb-FOIL.csv")
MODEL_PATH = os.path.join("models", "best_model_gradient_boosting.pkl")
SHAP_BAR = "shap_summary_bar.png"
SHAP_BEES = "shap_beeswarm.png"

# ----- Load model -----
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded from: " + MODEL_PATH)
    except Exception as e:
        st.error("Failed to load model: " + str(e))
else:
    st.warning(f"Model not found at '{MODEL_PATH}'. Save your trained model there and refresh.")

# ----- Load history data (optional) -----
raw = None
if os.path.exists(DATA_PATH):
    try:
        raw = pd.read_csv(DATA_PATH)
        raw['date'] = pd.to_datetime(raw['date'])
        raw = raw.sort_values('date').reset_index(drop=True)
        st.sidebar.info("Loaded sample historical data from Data/Uber-Jan-Feb-FOIL.csv")
    except Exception as e:
        st.sidebar.error("Failed to load sample CSV: " + str(e))
else:
    st.sidebar.warning("No sample CSV found at Data/Uber-Jan-Feb-FOIL.csv. You can upload history below.")

uploaded = st.sidebar.file_uploader("Upload historical CSV (optional). Must have columns: date,trips,active_vehicles", type=["csv"])
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        uploaded_df['date'] = pd.to_datetime(uploaded_df['date'])
        raw = uploaded_df.sort_values('date').reset_index(drop=True)
        st.sidebar.success("Uploaded history loaded and will be used for feature computation.")
    except Exception as e:
        st.sidebar.error("Unable to read uploaded CSV: " + str(e))

# ----- helper: create features for a target date -----
def make_features_for_date(history_df, target_date, active_vehicles_override=None):
    df = history_df.copy().sort_values('date').reset_index(drop=True)
    df['trips'] = df['trips'].astype(float)
    df['trips_rolling_mean_3'] = df['trips'].rolling(window=3).mean()
    df['trips_rolling_mean_7'] = df['trips'].rolling(window=7).mean()
    df['lag_1'] = df['trips'].shift(1)
    df['lag_2'] = df['trips'].shift(2)
    df['lag_3'] = df['trips'].shift(3)
    last = df.iloc[-1]

    d = pd.to_datetime(target_date)
    feat = {}
    feat['date'] = d
    feat['day_of_week'] = d.day_name()
    feat['is_weekend'] = 1 if d.day_name() in ['Saturday','Sunday'] else 0
    feat['month'] = d.month
    feat['day'] = d.day

    # rolling/lag from history; fallback to last known
    feat['trips_rolling_mean_3'] = float(df['trips_rolling_mean_3'].dropna().iloc[-1]) if df['trips_rolling_mean_3'].dropna().shape[0] > 0 else float(last['trips'])
    feat['trips_rolling_mean_7'] = float(df['trips_rolling_mean_7'].dropna().iloc[-1]) if df['trips_rolling_mean_7'].dropna().shape[0] > 0 else float(last['trips'])
    feat['lag_1'] = float(last['trips']) if not np.isnan(last['trips']) else 0.0
    feat['lag_2'] = float(df['lag_2'].dropna().iloc[-1]) if df['lag_2'].dropna().shape[0] > 0 else feat['lag_1']
    feat['lag_3'] = float(df['lag_3'].dropna().iloc[-1]) if df['lag_3'].dropna().shape[0] > 0 else feat['lag_1']
    feat['active_vehicles'] = active_vehicles_override if active_vehicles_override is not None else (float(last['active_vehicles']) if 'active_vehicles' in last else 0.0)
    return pd.DataFrame([feat])

# ----- Sidebar controls -----
st.sidebar.header("Prediction options")
if raw is not None:
    default_date = (raw['date'].max() + pd.Timedelta(days=1)).date()
else:
    default_date = pd.Timestamp.now().date()

target_date = st.sidebar.date_input("Target date to predict", value=default_date)
active_override = st.sidebar.number_input("Active vehicles override (0 = use history's last value)", min_value=0.0, step=1.0, value=0.0)
show_shap_images = st.sidebar.checkbox("Show SHAP images if inline SHAP fails", value=True)

# ----- Main layout: columns -----
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    st.write("Select target date and (optionally) override active vehicles. Click Predict to compute features and run the model.")
    if st.button("Create features & Predict"):
        if raw is None:
            st.error("No historical data available. Upload history or place Data/Uber-Jan-Feb-FOIL.csv in the Data folder.")
        elif model is None:
            st.error("No trained model found. Save your model to models/best_model_gradient_boosting.pkl.")
        else:
            # build features and predict
            feat_df = make_features_for_date(raw, target_date, active_vehicles_override=(active_override if active_override>0 else None))
            st.session_state['last_features'] = feat_df.copy()
            st.session_state['last_target'] = str(target_date)
            st.success("Features created. See results in the right column.")
            st.rerun()


with col2:
    st.subheader("Results")
    if 'last_features' in st.session_state:
        feat_df = st.session_state['last_features']
        st.write("### Features used for prediction")
        st.table(feat_df.T)

        model_features = ['active_vehicles','is_weekend','month','day','trips_rolling_mean_3','trips_rolling_mean_7','lag_1','lag_2','lag_3']
        X_pred = feat_df[model_features]
        try:
            pred = model.predict(X_pred)[0]
            pred_int = int(round(pred))
            st.success(f"Predicted trips for {st.session_state['last_target']}: **{pred_int:,}** trips")
        except Exception as e:
            st.error("Model prediction failed: " + str(e))
            pred = None
            pred_int = None

        # historical + predicted plot
        try:
            hist_plot = raw[['date','trips']].copy()
            predicted_row = pd.DataFrame([{'date': pd.to_datetime(st.session_state['last_target']), 'trips': pred}])
            hist_plot = pd.concat([hist_plot, predicted_row], ignore_index=True)
            hist_plot = hist_plot.sort_values('date')
            fig, ax = plt.subplots(figsize=(8,3.5))
            ax.plot(hist_plot['date'], hist_plot['trips'], marker='o')
            ax.axvline(pd.to_datetime(st.session_state['last_target']), color='gray', linestyle='--')
            ax.set_title("Historical trips & predicted point")
            ax.set_xlabel("Date")
            ax.set_ylabel("Trips")
            st.pyplot(fig)
        except Exception as e:
            st.warning("Plot failed: " + str(e))

        # download the prediction row as CSV
        try:
            out_df = feat_df.copy()
            out_df['predicted_trips'] = pred_int
            csv = out_df.to_csv(index=False)
            st.download_button("Download prediction CSV", data=csv, file_name=f"prediction_{st.session_state['last_target']}.csv", mime="text/csv")
        except Exception as e:
            st.info("Download not available: " + str(e))

        # SHAP explanation (try inline, otherwise show images)
        try:
            explainer = shap.TreeExplainer(model, X_pred)
            # if history large, show global on last N days; else show local force plot
            if raw.shape[0] >= 10:
                sample_feats = []
                for i in range(max(0, raw.shape[0]-10), raw.shape[0]):
                    tmp = raw.copy().iloc[:i+1]
                    tmp['trips'] = tmp['trips'].astype(float)
                    tmp['trips_rolling_mean_3'] = tmp['trips'].rolling(window=3).mean()
                    tmp['trips_rolling_mean_7'] = tmp['trips'].rolling(window=7).mean()
                    tmp['lag_1'] = tmp['trips'].shift(1)
                    tmp['lag_2'] = tmp['trips'].shift(2)
                    tmp['lag_3'] = tmp['trips'].shift(3)
                    lastrow = tmp.iloc[-1]
                    feat = {
                        'active_vehicles': float(lastrow['active_vehicles']),
                        'is_weekend': 1 if pd.to_datetime(lastrow['date']).day_name() in ['Saturday', 'Sunday'] else 0,
                        'month': pd.to_datetime(lastrow['date']).month,
                        'day': pd.to_datetime(lastrow['date']).day,
                        'trips_rolling_mean_3': float(tmp['trips_rolling_mean_3'].dropna().iloc[-1]) if tmp['trips_rolling_mean_3'].dropna().shape[0] > 0 else float(lastrow['trips']),
                        'trips_rolling_mean_7': float(tmp['trips_rolling_mean_7'].dropna().iloc[-1]) if tmp['trips_rolling_mean_7'].dropna().shape[0] > 0 else float(lastrow['trips']),
                        'lag_1': float(lastrow['trips']) if not np.isnan(lastrow['trips']) else 0.0,
                        'lag_2': float(tmp['lag_2'].dropna().iloc[-1]) if tmp['lag_2'].dropna().shape[0] > 0 else float(lastrow['trips']),
                        'lag_3': float(tmp['lag_3'].dropna().iloc[-1]) if tmp['lag_3'].dropna().shape[0] > 0 else float(lastrow['trips'])
                    }
                    sample_feats.append(feat)
                sample_df = pd.DataFrame(sample_feats)
                sample_df = sample_df[['active_vehicles','is_weekend','month','day','trips_rolling_mean_3','trips_rolling_mean_7','lag_1','lag_2','lag_3']]
                shap_values = explainer.shap_values(sample_df)
                st.subheader("SHAP: Global importance (recent sample)")
                plt.figure(figsize=(6,2.5))
                shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False)
                st.pyplot(plt.gcf())
            else:
                shap_values = explainer.shap_values(X_pred)
                st.subheader("SHAP: Local contributions")
                plt.figure(figsize=(6,2.5))
                shap.force_plot(explainer.expected_value, shap_values[0], X_pred.iloc[0], matplotlib=True, show=False)
                st.pyplot(plt.gcf())
        except Exception as e:
            st.warning("Inline SHAP failed: " + str(e))
            if show_shap_images:
                if os.path.exists(SHAP_BAR):
                    st.image(SHAP_BAR, caption="SHAP summary (bar)", use_column_width=True)
                if os.path.exists(SHAP_BEES):
                    st.image(SHAP_BEES, caption="SHAP beeswarm", use_column_width=True)
            else:
                st.info("Enable 'Show SHAP images' in the sidebar to display saved SHAP images as fallback.")
    else:
        st.info("No prediction run yet. Use the controls on the left and click 'Create features & Predict'.")
