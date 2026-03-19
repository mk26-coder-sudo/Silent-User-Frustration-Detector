import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Silent User Frustration Detector",
    layout="wide"
)

# ----------------------------
# Title
# ----------------------------
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

# ----------------------------
# Load Model + Feature Names
# ----------------------------
model = joblib.load("models/frustration_model.pkl")
feature_names = [
    "session_duration",
    "num_clicks",
    "error_count",
    "rapid_clicks",
    "page_revisits",
    "scroll_depth",
    "idle_time"
]


# SHAP Explainer
explainer = shap.TreeExplainer(model)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🧠 User Behavior Inputs")

error_count = st.sidebar.slider("Error Count", 0, 10, 1)
page_revisits = st.sidebar.slider("Page Revisits", 0, 15, 2)
idle_time = st.sidebar.slider("Idle Time (seconds)", 0, 300, 50)
rapid_clicks = st.sidebar.slider("Rapid Clicks", 0, 10, 1)
session_duration = st.sidebar.slider("Session Duration (seconds)", 0, 1000, 200)
num_clicks = st.sidebar.slider("Total Clicks", 0, 100, 20)
scroll_depth = st.sidebar.slider("Scroll Depth (%)", 0, 100, 40)

# ----------------------------
# Prediction
# ----------------------------
st.header("📊 Prediction Result")

if st.button("Predict Frustration"):

    # Create input in correct format
    input_values = [
        error_count,
        page_revisits,
        idle_time,
        rapid_clicks,
        session_duration,
        num_clicks,
        scroll_depth
    ]

    input_data = pd.DataFrame([input_values], columns=feature_names)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # ----------------------------
    # Result UI
    # ----------------------------
    if prediction == 1:
        st.markdown(
            f"""
            <div style="padding:25px;
                        border-radius:15px;
                        background-color:#5c1a1b;
                        color:white;
                        text-align:center;">
                <h2>😠 User is Frustrated</h2>
                <h3>Confidence: {probability:.2f}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="padding:25px;
                        border-radius:15px;
                        background-color:#1e5631;
                        color:white;
                        text-align:center;">
                <h2>🙂 User is Not Frustrated</h2>
                <h3>Confidence: {1 - probability:.2f}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ----------------------------
    # SHAP Explanation
    # ----------------------------
    st.subheader("🔍 Why This Prediction?")

    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)