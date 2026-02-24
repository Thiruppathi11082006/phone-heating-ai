import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Smart Phone Heating AI",
    page_icon="📱",
    layout="wide"
)

model = joblib.load("rf_model.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;color:#00d4ff;'>
📱 Smart Phone Heating AI Dashboard
</h1>
<hr>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("⚙ Device Controls")

cpu = st.sidebar.slider("CPU Usage (%)", 0, 100, 50)
temp = st.sidebar.slider("Temperature (°C)", 20, 50, 35)
charging = st.sidebar.toggle("Charging Mode")
brightness = st.sidebar.slider("Brightness (%)", 0, 100, 60)
apps = st.sidebar.slider("Running Apps", 0, 15, 5)
network = st.sidebar.slider("Network Usage (%)", 0, 100, 40)

charging_val = 1 if charging else 0

input_data = np.array([[cpu, temp, charging_val,
                        brightness, apps, network]])

# ---------------- PREDICT BUTTON ----------------
st.markdown("## 🚀 Run AI Prediction")

predict = st.button(
    "🔥 Predict Heating Risk",
    use_container_width=True
)

# session storage
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = 0

# ---------------- PREDICTION ----------------
if predict:

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    # -------- SMART SAFETY OVERRIDE --------
    if brightness > 85 and apps > 10:
        pred = 1

    if cpu > 85 and charging and brightness > 80:
        pred = 2

    if temp >= 42:
        pred = 2

    st.session_state.prediction = pred
    st.session_state.confidence = round(max(prob) * 100, 2)

prediction = st.session_state.prediction
confidence = st.session_state.confidence

# ---------------- RESULT DISPLAY ----------------
st.markdown("---")
st.subheader("🤖 AI Heating Risk Analysis")

if prediction == 0:
    st.success("✅ SAFE — Device operating normally")

elif prediction == 1:
    st.warning("⚠ WARNING — Device heating risk detected")
    st.info("👉 Reduce brightness or close background apps.")

elif prediction == 2:
    st.error("🔥 CRITICAL HEATING RISK!")
    st.markdown("""
    ### 🚨 Immediate Actions
    - Stop heavy applications
    - Disconnect charger
    - Allow device cooling
    """)

# ---------------- CONFIDENCE ----------------
if prediction is not None:
    st.markdown("### 🧠 AI Confidence Level")
    st.progress(confidence / 100)
    st.write(f"Prediction Confidence: **{confidence}%**")

# ---------------- MINI DATA GRAPH ----------------
data = pd.read_csv("phone_heating_dataset.csv")

st.markdown("---")
st.subheader("📊 Training Dataset Temperature Trend")

st.line_chart(data["Temp"])

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>
Machine Learning Fundamentals Project by Thiru
</p>
""", unsafe_allow_html=True)