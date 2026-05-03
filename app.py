import json
import time
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Phone Heating Risk Prediction",
    page_icon="phone",
    layout="wide",
    initial_sidebar_state="expanded",
)


FEATURE_COLUMNS = ["cpu", "temp", "charging", "brightness", "apps", "network"]

SCENARIO_DATA = {
    "Normal usage": [25, 32.0, 0, 40, 3, 20],
    "Gaming": [85, 45.0, 0, 90, 8, 70],
    "Video streaming": [60, 40.0, 0, 75, 5, 85],
    "Charging with heavy use": [70, 43.0, 1, 65, 6, 40],
    "Video recording": [75, 45.0, 0, 80, 4, 30],
    "5G navigation in sunlight": [68, 41.5, 0, 95, 7, 88],
    "Fast charging with hotspot": [72, 44.5, 1, 82, 6, 92],
}

RISK_MAP = {
    0: {
        "label": "Safe",
        "color": "#5BD6A2",
        "description": "Temperature is stable and the device is operating within a low-risk zone.",
        "actions": "Continue usage normally and monitor only during sustained workloads.",
    },
    1: {
        "label": "Warning",
        "color": "#F2B35D",
        "description": "Heat is increasing and performance may drop if the workload continues.",
        "actions": "Lower brightness, close background apps, and reduce charging or gaming load.",
    },
    2: {
        "label": "Critical",
        "color": "#FF6B6B",
        "description": "The device is at high heating risk and should be cooled immediately.",
        "actions": "Stop intensive usage, unplug charging if possible, and allow the phone to cool down.",
    },
}


@st.cache_resource
def load_model():
    return joblib.load("heating_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv("phone_heating_dataset.csv")


@st.cache_data
def load_training_summary():
    summary_path = Path("training_summary.json")
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


@st.cache_data
def evaluate_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df["risk"],
        test_size=0.25,
        random_state=42,
        stratify=df["risk"],
    )
    model = load_model()
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Safe", "Warning", "Critical"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    return accuracy_score(y_test, predictions), pd.DataFrame(report).transpose(), pd.DataFrame(
        matrix,
        index=["Actual Safe", "Actual Warning", "Actual Critical"],
        columns=["Pred Safe", "Pred Warning", "Pred Critical"],
    )


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 119, 61, 0.20), transparent 26%),
                radial-gradient(circle at top right, rgba(90, 209, 255, 0.18), transparent 30%),
                linear-gradient(135deg, #08111f 0%, #101c2f 50%, #08111f 100%);
            color: #edf3ff;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1728 0%, #0a1422 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] .stRadio label {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            margin-bottom: 10px;
            padding: 12px 14px;
        }

        [data-testid="stSidebar"] .stRadio label p {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            color: #eef4ff !important;
        }

        [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
            gap: 0.3rem;
        }

        .hero-card,
        .glass-card,
        .result-card,
        .mini-card,
        .tip-card {
            background: rgba(10, 18, 30, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 22px;
            backdrop-filter: blur(10px);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        }

        .hero-card {
            padding: 28px 30px;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #f5f7ff;
            margin-bottom: 0.3rem;
            line-height: 1.1;
        }

        .hero-text {
            color: #c9d5e9;
            font-size: 1rem;
            max-width: 760px;
        }

        .glass-card,
        .result-card {
            padding: 22px;
            margin-top: 0.8rem;
        }

        .mini-card {
            padding: 18px;
            text-align: center;
            min-height: 120px;
        }

        .mini-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #ffffff;
        }

        .mini-label {
            color: #9ab0d0;
            font-size: 0.95rem;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f0f4ff;
            margin-bottom: 0.6rem;
        }

        .helper {
            color: #99abc6;
            font-size: 0.95rem;
        }

        .sidebar-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.35rem;
        }

        .sidebar-subtitle {
            color: #9db2d3;
            font-size: 1rem;
            margin-bottom: 0.85rem;
        }

        .tip-card {
            padding: 16px;
            min-height: 105px;
        }

        .tip-title {
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .tip-body {
            color: #b5c5de;
            font-size: 0.94rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 14px;
            border: 0;
            background: linear-gradient(90deg, #ff8c42 0%, #ff5f6d 100%);
            color: white;
            font-weight: 700;
            padding: 0.8rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_prediction(model, values):
    features = np.array([values], dtype=float)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities)) * 100
    risk_score = round(
        min(
            100.0,
            probabilities[1] * 55 + probabilities[2] * 100 + (values[1] - 28) * 1.8 + values[0] * 0.08,
        ),
        1,
    )
    return prediction, probabilities, confidence, risk_score


def infer_contributors(values):
    labels = {
        "cpu": "CPU load",
        "temp": "Device temperature",
        "charging": "Charging state",
        "brightness": "Screen brightness",
        "apps": "Running apps",
        "network": "Network activity",
    }
    weights = {
        "cpu": values[0] / 100,
        "temp": max(0.0, (values[1] - 28) / 20),
        "charging": 0.7 if values[2] else 0.15,
        "brightness": values[3] / 100,
        "apps": values[4] / 10,
        "network": values[5] / 100,
    }
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    return [labels[key] for key, _ in ranked[:3]]


def build_recommendations(values, prediction):
    tips = []
    if values[1] >= 42:
        tips.append(("Cool device", "Pause heavy tasks and move the phone away from heat or direct sunlight."))
    if values[2]:
        tips.append(("Reduce charging heat", "Switch to normal charging or unplug temporarily during heavy usage."))
    if values[0] >= 70:
        tips.append(("Lower processor load", "Stop gaming, recording, or long foreground tasks for a few minutes."))
    if values[3] >= 80:
        tips.append(("Dim the display", "Reduce brightness to cut display heat and battery drain."))
    if values[4] >= 7:
        tips.append(("Close background apps", "Shut down unused apps that keep the CPU and RAM active."))
    if values[5] >= 75:
        tips.append(("Reduce network strain", "Pause hotspot, large downloads, or streaming to cool the radio stack."))
    if prediction == 0 and not tips:
        tips.append(("Healthy state", "Current conditions are stable. Periodic monitoring is enough."))
    return tips[:4]


def risk_guide():
    st.markdown('<div class="section-title">Risk guide</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, risk_key in zip(cols, [0, 1, 2]):
        risk = RISK_MAP[risk_key]
        col.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-value" style="color:{risk['color']}">{risk['label']}</div>
                <div class="mini-label">{risk['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def dataset_insights(df):
    st.markdown('<div class="section-title">Dataset insights</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Samples", f"{len(df)}")
    metric_cols[1].metric("Average temp", f"{df['temp'].mean():.1f} C")
    metric_cols[2].metric("Average CPU", f"{df['cpu'].mean():.0f}%")
    metric_cols[3].metric("Critical cases", f"{int((df['risk'] == 2).sum())}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.caption("Temperature trend across records")
        st.line_chart(df["temp"], height=260)
    with chart_right:
        st.caption("Average temperature by risk level")
        avg_temp_by_risk = (
            df.groupby("risk", as_index=False)["temp"]
            .mean()
            .replace({"risk": {0: "Safe", 1: "Warning", 2: "Critical"}})
            .set_index("risk")
        )
        st.bar_chart(avg_temp_by_risk, height=260)

    scatter_df = df.rename(columns={"temp": "Temperature", "cpu": "CPU"})
    st.caption("CPU vs temperature relationship")
    st.scatter_chart(scatter_df, x="CPU", y="Temperature", color="risk", height=320)
    st.dataframe(df.head(10), use_container_width=True)


def batch_prediction_table(model, uploaded_file):
    batch_df = pd.read_csv(uploaded_file)
    missing = [column for column in FEATURE_COLUMNS if column not in batch_df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(missing)}")
        return

    features = batch_df[FEATURE_COLUMNS].astype(float)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    result_df = batch_df.copy()
    result_df["predicted_risk"] = [RISK_MAP[int(pred)]["label"] for pred in predictions]
    result_df["confidence"] = probabilities.max(axis=1).round(4)
    result_df["risk_score"] = (
        probabilities[:, 1] * 55
        + probabilities[:, 2] * 100
        + (result_df["temp"] - 28) * 1.8
        + result_df["cpu"] * 0.08
    ).clip(0, 100).round(1)

    st.dataframe(result_df, use_container_width=True)
    st.download_button(
        "Download batch results",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name="phone_heating_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def feature_importance_section(model):
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is unavailable for this model.")
        return

    importance_df = (
        pd.DataFrame(
            {
                "Feature": ["CPU", "Temperature", "Charging", "Brightness", "Apps", "Network"],
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .set_index("Feature")
    )
    st.bar_chart(importance_df, height=300)
    st.dataframe(importance_df.reset_index(), use_container_width=True)


def report_download(bundle, contributors, recommendations):
    lines = [
        "PHONE HEATING RISK REPORT",
        f"Scenario: {bundle['scenario']}",
        f"Predicted risk: {RISK_MAP[bundle['prediction']]['label']}",
        f"Confidence: {bundle['confidence']:.1f}%",
        f"Risk score: {bundle['risk_score']:.1f}/100",
        f"Top contributors: {', '.join(contributors)}",
        "",
        "Recommended actions:",
    ]
    for title, body in recommendations:
        lines.append(f"- {title}: {body}")
    report_text = "\n".join(lines)
    st.download_button(
        "Download risk report",
        report_text.encode("utf-8"),
        file_name="phone_heating_risk_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


def simulate_next_values(current_values, scenario_name, volatility):
    scenario_targets = {
        "Normal usage": [28, 33.0, 0, 35, 3, 18],
        "Gaming": [90, 46.5, 0, 92, 9, 78],
        "Video streaming": [62, 40.5, 0, 75, 5, 88],
        "Charging with heavy use": [74, 44.8, 1, 68, 7, 46],
        "Video recording": [78, 45.2, 0, 82, 5, 35],
        "5G navigation in sunlight": [70, 42.2, 0, 97, 8, 90],
        "Fast charging with hotspot": [76, 45.0, 1, 84, 7, 94],
    }
    target = scenario_targets[scenario_name]
    noise_scale = np.array([3.5, 0.45, 0.0, 4.0, 0.55, 4.5]) * volatility
    current = np.array(current_values, dtype=float)
    target = np.array(target, dtype=float)
    drift = (target - current) * 0.22
    next_values = current + drift + np.random.normal(0, noise_scale)
    next_values[0] = np.clip(next_values[0], 10, 100)
    next_values[1] = np.clip(next_values[1], 28.0, 52.0)
    next_values[2] = target[2]
    next_values[3] = np.clip(next_values[3], 20, 100)
    next_values[4] = np.clip(np.round(next_values[4]), 1, 15)
    next_values[5] = np.clip(next_values[5], 5, 100)
    return [
        int(round(next_values[0])),
        round(float(next_values[1]), 2),
        int(round(next_values[2])),
        int(round(next_values[3])),
        int(round(next_values[4])),
        int(round(next_values[5])),
    ]


def render_live_monitor(model):
    st.markdown('<div class="section-title">Live sensor simulation</div>', unsafe_allow_html=True)
    st.write("Simulate real-time phone telemetry and watch the predicted heating risk change as readings stream in.")

    control_cols = st.columns(4)
    live_scenario = control_cols[0].selectbox("Live scenario", list(SCENARIO_DATA.keys()), key="live_scenario")
    simulation_steps = control_cols[1].slider("Stream length", 5, 30, 12, key="live_steps")
    delay_seconds = control_cols[2].slider("Delay per reading", 0.1, 1.0, 0.3, 0.1, key="live_delay")
    volatility = control_cols[3].slider("Sensor volatility", 0.5, 1.8, 1.0, 0.1, key="live_volatility")

    if "live_history" not in st.session_state:
        st.session_state.live_history = []

    action_cols = st.columns([1, 1, 3])
    start_stream = action_cols[0].button("Start live stream", use_container_width=True)
    clear_stream = action_cols[1].button("Clear history", use_container_width=True)

    if clear_stream:
        st.session_state.live_history = []

    live_metrics = st.empty()
    live_chart = st.empty()
    live_table = st.empty()

    if start_stream:
        st.session_state.live_history = []
        current_values = list(SCENARIO_DATA[live_scenario])
        for step in range(1, simulation_steps + 1):
            current_values = simulate_next_values(current_values, live_scenario, volatility)
            prediction, probabilities, confidence, risk_score = make_prediction(model, current_values)
            st.session_state.live_history.append(
                {
                    "step": step,
                    "scenario": live_scenario,
                    "cpu": current_values[0],
                    "temp": current_values[1],
                    "charging": current_values[2],
                    "brightness": current_values[3],
                    "apps": current_values[4],
                    "network": current_values[5],
                    "predicted_risk": RISK_MAP[prediction]["label"],
                    "confidence": round(confidence, 2),
                    "risk_score": risk_score,
                }
            )
            history_df = pd.DataFrame(st.session_state.live_history)
            latest = history_df.iloc[-1]

            with live_metrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current CPU", f"{int(latest['cpu'])}%")
                m2.metric("Current temp", f"{latest['temp']:.1f} C")
                m3.metric("Predicted risk", latest["predicted_risk"])
                m4.metric("Risk score", f"{latest['risk_score']:.1f}/100")

            with live_chart.container():
                chart_df = history_df.set_index("step")[["temp", "cpu", "risk_score"]]
                st.line_chart(chart_df, height=280)

            with live_table.container():
                st.dataframe(history_df.tail(8), use_container_width=True)

            time.sleep(delay_seconds)

    if st.session_state.live_history:
        history_df = pd.DataFrame(st.session_state.live_history)
        st.download_button(
            "Download live monitoring log",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="phone_live_monitor_log.csv",
            mime="text/csv",
            use_container_width=True,
        )


inject_styles()
model = load_model()
df = load_dataset()
accuracy, report_df, matrix_df = evaluate_model(df)
training_summary = load_training_summary()

if "prediction_bundle" not in st.session_state:
    default_values = SCENARIO_DATA["Normal usage"]
    initial_prediction = make_prediction(model, default_values)
    st.session_state.prediction_bundle = {
        "scenario": "Normal usage",
        "values": default_values,
        "prediction": initial_prediction[0],
        "probabilities": initial_prediction[1],
        "confidence": initial_prediction[2],
        "risk_score": initial_prediction[3],
    }


with st.sidebar:
    st.markdown('<div class="sidebar-title">Project navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Choose a section from the sidebar menu.</div>', unsafe_allow_html=True)
    selected_section = st.radio(
        "Open section",
        [
            "Smart predictor",
            "Live monitor",
            "Batch monitor",
            "Dataset analytics",
            "Model intelligence",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.title("Project overview")
    st.write("Phone Heating Risk Prediction using ML")
    st.write(
        "A complete risk-analysis dashboard for predicting overheating state from smartphone telemetry, "
        "understanding why it happens, and acting on the result."
    )
    st.markdown("**Core features**")
    st.write("Single prediction, batch CSV analysis, analytics, model evaluation, feature importance, and smart recommendations.")
    st.markdown("**Required CSV columns for batch mode**")
    st.code(", ".join(FEATURE_COLUMNS))


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Phone Heating Risk Prediction using ML</div>
        <div class="hero-text">
            An interactive machine learning system that predicts overheating risk, explains important thermal factors,
            evaluates model behavior, and supports both single-device and batch-device analysis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_cols = st.columns(5)
summary_cols[0].markdown(
    '<div class="mini-card"><div class="mini-value">3</div><div class="mini-label">Risk classes</div></div>',
    unsafe_allow_html=True,
)
summary_cols[1].markdown(
    f'<div class="mini-card"><div class="mini-value">{len(df)}</div><div class="mini-label">Dataset rows</div></div>',
    unsafe_allow_html=True,
)
summary_cols[2].markdown(
    '<div class="mini-card"><div class="mini-value">6</div><div class="mini-label">Telemetry signals</div></div>',
    unsafe_allow_html=True,
)
summary_cols[3].markdown(
    f'<div class="mini-card"><div class="mini-value">{accuracy * 100:.1f}%</div><div class="mini-label">Estimated accuracy</div></div>',
    unsafe_allow_html=True,
)
summary_cols[4].markdown(
    '<div class="mini-card"><div class="mini-value">Live</div><div class="mini-label">Sensor stream ready</div></div>',
    unsafe_allow_html=True,
)

if selected_section == "Smart predictor":
    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Configure device state</div>', unsafe_allow_html=True)

        scenario_name = st.selectbox("Usage scenario", list(SCENARIO_DATA.keys()))
        base_cpu, base_temp, base_charging, base_brightness, base_apps, base_network = SCENARIO_DATA[scenario_name]

        with st.form("predict_form"):
            cpu = st.slider("CPU usage (%)", 0, 100, int(base_cpu))
            temp = st.slider("Device temperature (C)", 20.0, 55.0, float(base_temp), 0.1)
            charging = st.toggle("Phone is charging", value=bool(base_charging))
            brightness = st.slider("Screen brightness (%)", 0, 100, int(base_brightness))
            apps = st.slider("Running apps", 1, 15, int(base_apps))
            network = st.slider("Network usage (%)", 0, 100, int(base_network))
            submitted = st.form_submit_button("Predict heating risk", use_container_width=True)

        st.markdown(
            '<div class="helper">Use a preset scenario for demonstration or tune the telemetry values to simulate a real phone state.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        metric_cols = st.columns(3)
        metric_cols[0].metric("CPU", f"{cpu}%")
        metric_cols[1].metric("Temperature", f"{temp:.1f} C")
        metric_cols[2].metric("Charging", "Yes" if charging else "No")

        extra_cols = st.columns(3)
        extra_cols[0].metric("Brightness", f"{brightness}%")
        extra_cols[1].metric("Apps", apps)
        extra_cols[2].metric("Network", f"{network}%")

        if submitted:
            values = [cpu, temp, int(charging), brightness, apps, network]
            prediction, probabilities, confidence, risk_score = make_prediction(model, values)
            st.session_state.prediction_bundle = {
                "scenario": scenario_name,
                "values": values,
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": confidence,
                "risk_score": risk_score,
            }

    with right_col:
        bundle = st.session_state.prediction_bundle
        risk = RISK_MAP[bundle["prediction"]]
        contributors = infer_contributors(bundle["values"])
        recommendations = build_recommendations(bundle["values"], bundle["prediction"])
        prob_df = pd.DataFrame(
            {
                "Risk": [RISK_MAP[0]["label"], RISK_MAP[1]["label"], RISK_MAP[2]["label"]],
                "Probability": bundle["probabilities"],
            }
        ).set_index("Risk")

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="risk-badge" style="background:{risk['color']}22;color:{risk['color']};border:1px solid {risk['color']}55;">
                Predicted risk: {risk['label']}
            </div>
            <div style="font-size:2.3rem;font-weight:800;color:#f7f9ff;">{bundle['confidence']:.1f}% confidence</div>
            <p style="color:#c7d2e6;font-size:1rem;">{risk['description']}</p>
            <p style="color:#eef4ff;"><strong>Recommended action:</strong> {risk['actions']}</p>
            <p style="color:#92a6c6;font-size:0.95rem;">Scenario selected: {bundle['scenario']}</p>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(bundle["risk_score"] / 100, 0.0), 1.0), text=f"Thermal risk score: {bundle['risk_score']:.1f}/100")
        st.caption("Model probability distribution")
        st.bar_chart(prob_df, height=220)
        st.markdown("</div>", unsafe_allow_html=True)

        top_cols = st.columns(3)
        for col, name in zip(top_cols, contributors):
            col.markdown(
                f'<div class="mini-card"><div class="mini-value">{name}</div><div class="mini-label">Top thermal contributor</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Personalized cooling recommendations</div>', unsafe_allow_html=True)
        tip_cols = st.columns(len(recommendations))
        for col, (title, body) in zip(tip_cols, recommendations):
            col.markdown(
                f"""
                <div class="tip-card">
                    <div class="tip-title">{title}</div>
                    <div class="tip-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        report_download(bundle, contributors, recommendations)
        risk_guide()

elif selected_section == "Live monitor":
    render_live_monitor(model)

elif selected_section == "Batch monitor":
    st.markdown('<div class="section-title">Batch device screening</div>', unsafe_allow_html=True)
    st.write(
        "Upload a CSV containing multiple phone telemetry records to classify risk in bulk and export the results."
    )
    upload = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    sample_csv = pd.DataFrame([SCENARIO_DATA["Normal usage"], SCENARIO_DATA["Gaming"]], columns=FEATURE_COLUMNS)
    st.download_button(
        "Download sample CSV template",
        sample_csv.to_csv(index=False).encode("utf-8"),
        file_name="phone_heating_template.csv",
        mime="text/csv",
    )
    if upload is not None:
        batch_prediction_table(model, upload)

elif selected_section == "Dataset analytics":
    dataset_insights(df)

elif selected_section == "Model intelligence":
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.markdown('<div class="section-title">Feature importance</div>', unsafe_allow_html=True)
        feature_importance_section(model)
    with right_col:
        st.markdown('<div class="section-title">Model evaluation snapshot</div>', unsafe_allow_html=True)
        st.metric("Holdout accuracy", f"{accuracy * 100:.2f}%")
        st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
    st.dataframe(matrix_df, use_container_width=True)

    if training_summary:
        st.markdown('<div class="section-title">Critical-aware retraining summary</div>', unsafe_allow_html=True)
        retrain_cols = st.columns(4)
        retrain_cols[0].metric("Original critical rows", training_summary["original_class_counts"]["2"])
        retrain_cols[1].metric("Augmented critical rows", training_summary["augmented_class_counts"]["2"])
        retrain_cols[2].metric("Critical recall", f"{training_summary['critical_recall'] * 100:.1f}%")
        retrain_cols[3].metric("Retrained accuracy", f"{training_summary['test_accuracy'] * 100:.1f}%")
        st.caption("Saved from the imbalance-aware retraining pipeline.")

    st.markdown('<div class="section-title">Project highlights</div>', unsafe_allow_html=True)
    highlight_text = StringIO()
    highlight_text.write("1. Predicts safe, warning, and critical heating risk levels.\n")
    highlight_text.write("2. Includes a live sensor stream simulator for real-time monitoring.\n")
    highlight_text.write("3. Accepts both single-phone input and batch CSV screening.\n")
    highlight_text.write("4. Shows confidence, risk score, and likely heat contributors.\n")
    highlight_text.write("5. Uses critical-aware retraining plus evaluation metrics and analytics.\n")
    st.code(highlight_text.getvalue())
