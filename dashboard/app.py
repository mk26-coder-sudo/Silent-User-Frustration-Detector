import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Page Configuration

st.set_page_config(
    page_title="Silent User Frustration Detector",
    layout="wide"
)

# Title

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B;'>
    🚨 Silent User Frustration Detector
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Explainable ML System using Random Forest + SHAP
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Load Model + Feature Names

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "frustration_model.pkl"
)

model = joblib.load(model_path)

feature_names = [
    "session_duration",
    "num_clicks",
    "error_count",
    "rapid_clicks",
    "page_revisits",
    "scroll_depth",
    "idle_time"
]

explainer = shap.TreeExplainer(model)

# Sidebar Inputs

st.sidebar.header("🧠 User Behavior Inputs")

error_count = st.sidebar.slider("Error Count", 0, 10, 1)
page_revisits = st.sidebar.slider("Page Revisits", 0, 15, 2)
idle_time = st.sidebar.slider("Idle Time (seconds)", 0, 300, 50)
rapid_clicks = st.sidebar.slider("Rapid Clicks", 0, 10, 1)
session_duration = st.sidebar.slider("Session Duration (seconds)", 0, 1000, 200)
num_clicks = st.sidebar.slider("Total Clicks", 0, 100, 20)
scroll_depth = st.sidebar.slider("Scroll Depth (%)", 0, 100, 40)


# ----------------------------
# Threshold Slider (NEW)
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Decision Threshold")
threshold = st.sidebar.slider(
    "Frustration Threshold",
    min_value=0.3,
    max_value=0.7,
    value=0.5,
    step=0.05,
    help="Lower = catch more frustrated users (higher recall). Higher = only flag confident cases (higher precision)."
)

# 
# ----------------------------

# Prediction

st.header("📊 Prediction Result")

if st.button("Predict Frustration"):

    input_values = [
        session_duration,
        num_clicks,
        error_count,
        rapid_clicks,
        page_revisits,
        scroll_depth,
        idle_time
    ]

    input_data = pd.DataFrame([input_values], columns=feature_names)
    probability = model.predict_proba(input_data)[0][1]


    # Use threshold instead of default 0.5
    prediction = int(probability >= threshold)

    # ----------------------------
    # Risk Profile (NEW)
    # ----------------------------
    if probability >= 0.7:
        risk_label = "🔴 High Risk"
        risk_color = "#5c1a1b"
    elif probability >= 0.4:
        risk_label = "🟡 Medium Risk"
        risk_color = "#7a5c00"
    else:
        risk_label = "🟢 Low Risk"
        risk_color = "#1e5631"

    # ----------------------------
    # Plain English Reason (NEW)
    # ----------------------------
    shap_values = explainer(input_data)
    shap_vals = shap_values[0, :, 1].values
    feature_shap = dict(zip(feature_names, shap_vals))
    top_feature = max(feature_shap, key=feature_shap.get)

    reason_map = {
        "error_count":        "repeated errors on the page",
        "rapid_clicks":       "rapid clicking behavior (rage-click pattern)",
        "page_revisits":      "repeatedly revisiting the same pages",
        "idle_time":          "long idle periods suggesting confusion",
        "session_duration":   "unusually long session duration",
        "scroll_depth":       "low scroll depth suggesting early drop-off",
        "num_clicks":         "abnormally high number of clicks"
    }

    reason = reason_map.get(top_feature, top_feature)

    # ----------------------------
    # Result UI
    # ----------------------------
    st.markdown(
        f"""
        <div style="padding:25px;
                    border-radius:15px;
                    background-color:{risk_color};
                    color:white;
                    text-align:center;">
            <h2>{risk_label}</h2>
            <h3>Frustration Probability: {probability:.2f}</h3>
            <p style="font-size:16px;">Threshold set at: {threshold}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Plain English Reason (NEW)
    st.markdown("---")
    st.subheader("💬 What's driving this?")


    if prediction == 1:
        st.info(f"⚠️ This user's frustration is mainly driven by **{reason}**.")
    else:
        st.success(f"✅ No strong frustration signals detected. Closest factor was **{reason}**, but below threshold.")

    st.markdown("---")


    # ----------------------------
    # SHAP Explanation (existing)
    # ----------------------------
    st.subheader("🔍 Why This Prediction? (SHAP)")

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)


    st.markdown("---")

    # ----------------------------
    # Model Comparison Table (NEW)
    # ----------------------------
    st.subheader("📈 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Accuracy": [0.94, 0.98],
        "F1 (Frustrated Class)": [0.90, 0.97],
        "CV Mean Accuracy": [0.94, 0.97],
        "Explainability": ["SHAP (per-user)", "Coefficients (global)"],
        "Why Chosen": ["✅ Primary — richer explanations", "📊 Baseline comparison"]
    })

    st.dataframe(comparison_df, use_container_width=True)
    st.caption(
        "LR outperforms on synthetic data due to linear label generation. "
        "RF chosen as primary for its per-user SHAP explainability — "
        "more suitable for real-world non-linear behavioral patterns."
    )

