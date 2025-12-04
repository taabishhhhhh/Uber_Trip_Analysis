# app_streamlit.py
"""
Enterprise Streamlit app — production version
Place in project root (next to Data/ and models/), then run:
    streamlit run app_streamlit.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import importlib
import pandas as pd
import numpy as np
import streamlit as st

# Matplotlib - use Agg for headless safety; import with try/except
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

# SHAP is optional but preferred
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ------------- CONFIG -------------
st.set_page_config(
    page_title="Uber Trip Demand Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths (adjust if required)
DATA_PATH = os.path.join("Data", "Uber-Jan-Feb-FOIL.csv")
MODEL_PATH = os.path.join("models", "best_model_gradient_boosting.pkl")
SHAP_BAR = "shap_summary_bar.png"
SHAP_BEES = "shap_beeswarm.png"

MODEL_FEATURES = [
    'active_vehicles','is_weekend','month','day',
    'trips_rolling_mean_3','trips_rolling_mean_7',
    'lag_1','lag_2','lag_3'
]

# ------------- HELPERS -------------
@st.cache_data(ttl=600)
def load_history(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def load_model(path=MODEL_PATH):
    # Lazy import joblib to avoid top-level import issues
    try:
        import joblib
    except Exception as e:
        return None, f"Failed to import joblib: {type(e).__name__}: {e}. Ensure requirements.txt contains joblib."
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except AttributeError as ae:
        return None, (
            "Unpickle AttributeError — likely scikit-learn / Cython binary mismatch.\n"
            "Pin scikit-learn in requirements.txt to the version used when the model was saved.\n"
            f"Original error: {type(ae).__name__}: {ae}"
        )
    except Exception as e:
        return None, f"Failed to load model: {type(e).__name__}: {e}"

def safe_make_features(history_df: pd.DataFrame, target_date, active_override=None):
    df = history_df.copy().sort_values("date").reset_index(drop=True)
    df['trips'] = df['trips'].astype(float)
    df['trips_rolling_mean_3'] = df['trips'].rolling(3).mean()
    df['trips_rolling_mean_7'] = df['trips'].rolling(7).mean()
    df['lag_1'] = df['trips'].shift(1)
    df['lag_2'] = df['trips'].shift(2)
    df['lag_3'] = df['trips'].shift(3)

    last = df.iloc[-1]
    d = pd.to_datetime(target_date)

    feat = {
        "date": d,
        "day_of_week": d.day_name(),
        "is_weekend": 1 if d.day_name() in ['Saturday','Sunday'] else 0,
        "month": d.month,
        "day": d.day,
    }

    def last_nonnull(s, fallback):
        s2 = s.dropna()
        return float(s2.iloc[-1]) if s2.shape[0] > 0 else float(fallback)

    feat['trips_rolling_mean_3'] = last_nonnull(df['trips_rolling_mean_3'], last['trips'])
    feat['trips_rolling_mean_7'] = last_nonnull(df['trips_rolling_mean_7'], last['trips'])
    feat['lag_1'] = float(last['trips']) if not np.isnan(last['trips']) else 0.0
    feat['lag_2'] = last_nonnull(df['lag_2'], feat['lag_1'])
    feat['lag_3'] = last_nonnull(df['lag_3'], feat['lag_1'])
    feat['active_vehicles'] = float(active_override) if (active_override is not None and active_override > 0) else float(last.get('active_vehicles', 0.0))
    return pd.DataFrame([feat])

def plot_history_prediction(history_df, target_date, predicted_trips):
    df = history_df[['date','trips']].copy().sort_values('date').reset_index(drop=True)
    if predicted_trips is not None:
        df = pd.concat([df, pd.DataFrame([{'date': pd.to_datetime(target_date), 'trips': predicted_trips}])], ignore_index=True)
    df = df.sort_values('date').reset_index(drop=True)

    if not MATPLOTLIB_OK:
        return None

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(df['date'], df['trips'], marker='o', linewidth=1.4, label='Trips')
    if predicted_trips is not None:
        ax.scatter([pd.to_datetime(target_date)], [predicted_trips], s=80, zorder=6, label='Predicted')
        ax.axvline(pd.to_datetime(target_date), color='gray', linestyle='--', alpha=0.6)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Date")
    ax.set_ylabel("Trips")
    ax.set_title("Historical Trips & Predicted Point")
    ax.grid(alpha=0.25)
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig

# ------------- LOAD DATA & MODEL -------------
history_df = load_history()
model, model_err = load_model()

# ------------- SIDEBAR -------------
st.sidebar.markdown("## Controls")
st.sidebar.write("Upload a historical CSV (optional). Required columns: date,trips,active_vehicles")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded, parse_dates=['date'])
        uploaded_df = uploaded_df.sort_values('date').reset_index(drop=True)
        history_df = uploaded_df
        st.sidebar.success("Uploaded CSV loaded.")
    except Exception as e:
        st.sidebar.error("Failed to parse uploaded CSV: " + str(e))

st.sidebar.markdown("---")
if history_df is not None:
    default_date = (history_df['date'].max() + pd.Timedelta(days=1)).date()
else:
    default_date = pd.Timestamp.now().date()

target_date = st.sidebar.date_input("Target date to predict", value=default_date)
active_override = st.sidebar.number_input("Active vehicles override (0 = use last known)", min_value=0.0, step=1.0, value=0.0)
show_shap_images = st.sidebar.checkbox("Show SHAP fallback images", value=True)
shap_sample_days = st.sidebar.slider("SHAP sample size (recent days)", min_value=3, max_value=30, value=10)
st.sidebar.markdown("---")
if model is None:
    st.sidebar.warning("Model not loaded. Save trained model to models/best_model_gradient_boosting.pkl")
    if model_err:
        st.sidebar.caption(model_err)
else:
    st.sidebar.success("Model loaded ✓")

# ------------- HEADER (Header C) -------------
st.markdown("""<div style="padding:6px 0;">
<h1 style="margin:0; font-size:28px;">Uber Trip Demand Forecast Dashboard</h1>
<p style="margin:2px 0 12px 0; color:#555;">Powered by Gradient Boosting + SHAP explainability — polished for portfolio and interviews.</p>
</div>""", unsafe_allow_html=True)

# ------------- MAIN LAYOUT -------------
left, right = st.columns([1, 2])

with left:
    st.header("Run prediction")
    st.write("Choose a date (left), optionally override active vehicles, and click Predict.")
    if st.button("Predict"):
        if history_df is None:
            st.error("No historical data available. Upload CSV or place Data/Uber-Jan-Feb-FOIL.csv")
        elif model is None:
            st.error("Model not available. Save your trained model to models/best_model_gradient_boosting.pkl")
        else:
            feat = safe_make_features(history_df, target_date, active_override if active_override > 0 else None)
            st.session_state['last_features'] = feat
            st.session_state['last_target'] = str(target_date)
            try:
                Xp = feat[MODEL_FEATURES]
                pred_value = model.predict(Xp)[0]
                st.session_state['last_pred'] = float(pred_value)
                st.success("Prediction completed — see results on the right.")
            except Exception as e:
                st.session_state['last_pred'] = None
                st.error("Prediction failed: " + str(e))

with right:
    st.header("Results")
    if 'last_features' in st.session_state:
        feat_df = st.session_state['last_features']
        st.subheader("Model features (clean numeric table)")
        try:
            clean_features = feat_df[MODEL_FEATURES].astype(float)
            st.dataframe(clean_features, width="stretch")
        except Exception:
            st.dataframe(feat_df[MODEL_FEATURES].astype(str), width="stretch")

        st.markdown("**Meta**")
        meta = feat_df[['date','day_of_week']].copy()
        meta['date'] = meta['date'].dt.strftime("%Y-%m-%d")
        st.table(meta.T.astype(str))

        pred = st.session_state.get('last_pred', None)
        if pred is not None:
            st.success(f"Predicted trips for **{st.session_state['last_target']}**: **{int(round(pred)):,}** trips")
        else:
            st.warning("Prediction not available (see errors).")

        try:
            fig = plot_history_prediction(history_df, st.session_state['last_target'], st.session_state.get('last_pred', None))
            if fig is not None:
                st.pyplot(fig)
            else:
                st.table(history_df[['date','trips']].tail(15).astype(str))
        except Exception as e:
            st.warning("Plotting failed: " + str(e))

        try:
            out = feat_df.copy()
            out['predicted_trips'] = int(round(st.session_state['last_pred'])) if st.session_state.get('last_pred') is not None else None
            csv = out.to_csv(index=False)
            st.download_button("Download prediction (CSV)", data=csv, file_name=f"prediction_{st.session_state['last_target']}.csv", mime="text/csv")
        except Exception as e:
            st.info("Download not available: " + str(e))

        st.subheader("Model explainability (SHAP)")
        shap_failed = False
        try:
            if not SHAP_OK:
                raise RuntimeError("shap library not installed")

            explainer = shap.TreeExplainer(model)

            if history_df.shape[0] >= 5:
                n = min(shap_sample_days, history_df.shape[0]-1)
                sample_feats = []
                for i in range(history_df.shape[0]-n, history_df.shape[0]):
                    part = history_df.iloc[:i+1].copy()
                    part['trips'] = part['trips'].astype(float)
                    part['trips_rolling_mean_3'] = part['trips'].rolling(3).mean()
                    part['trips_rolling_mean_7'] = part['trips'].rolling(7).mean()
                    part['lag_1'] = part['trips'].shift(1)
                    part['lag_2'] = part['trips'].shift(2)
                    part['lag_3'] = part['trips'].shift(3)
                    lastrow = part.iloc[-1]
                    feat = {
                        'active_vehicles': float(lastrow['active_vehicles']),
                        'is_weekend': 1 if pd.to_datetime(lastrow['date']).day_name() in ['Saturday','Sunday'] else 0,
                        'month': pd.to_datetime(lastrow['date']).month,
                        'day': pd.to_datetime(lastrow['date']).day,
                        'trips_rolling_mean_3': float(part['trips_rolling_mean_3'].dropna().iloc[-1]) if part['trips_rolling_mean_3'].dropna().shape[0]>0 else float(lastrow['trips']),
                        'trips_rolling_mean_7': float(part['trips_rolling_mean_7'].dropna().iloc[-1]) if part['trips_rolling_mean_7'].dropna().shape[0]>0 else float(lastrow['trips']),
                        'lag_1': float(lastrow['trips']) if not np.isnan(lastrow['trips']) else 0.0,
                        'lag_2': float(part['lag_2'].dropna().iloc[-1]) if part['lag_2'].dropna().shape[0]>0 else float(lastrow['trips']),
                        'lag_3': float(part['lag_3'].dropna().iloc[-1]) if part['lag_3'].dropna().shape[0]>0 else float(lastrow['trips'])
                    }
                    sample_feats.append(feat)
                sample_df = pd.DataFrame(sample_feats)[MODEL_FEATURES]
                shap_vals = explainer.shap_values(sample_df)

                plt.figure(figsize=(6,2.6))
                shap.summary_plot(shap_vals, sample_df, plot_type="bar", show=False)
                st.pyplot(plt.gcf())
                plt.clf()

                plt.figure(figsize=(6,3.2))
                shap.summary_plot(shap_vals, sample_df, plot_type="dot", show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                st.info("Not enough history for global SHAP — showing local contribution instead.")

            try:
                X_local = feat_df[MODEL_FEATURES]
                shap_local = explainer.shap_values(X_local)
                plt.figure(figsize=(6,2.6))
                shap.force_plot(explainer.expected_value, shap_local[0], X_local.iloc[0], matplotlib=True, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e_local:
                st.warning("Local SHAP force plot failed (fallback to bar): " + str(e_local))
                plt.figure(figsize=(6,2.6))
                shap.summary_plot(shap_local, X_local, plot_type="bar", show=False)
                st.pyplot(plt.gcf())
                plt.clf()

        except Exception as e_shap:
            shap_failed = True
            st.warning("Inline SHAP failed: " + str(e_shap))

        if shap_failed and show_shap_images:
            st.info("Showing prepared SHAP images as fallback.")
            if os.path.exists(SHAP_BAR):
                st.image(SHAP_BAR, caption="SHAP summary (bar)", use_column_width=True)
            if os.path.exists(SHAP_BEES):
                st.image(SHAP_BEES, caption="SHAP beeswarm", use_column_width=True)
            if not os.path.exists(SHAP_BAR) and not os.path.exists(SHAP_BEES):
                st.info("No fallback SHAP images found in repo root; generate them from notebook if needed.")

    else:
        st.info("No prediction run yet. Use the controls (left) and click Predict.")
