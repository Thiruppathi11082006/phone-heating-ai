import streamlit as st

# ---------------- PAGE ----------------
st.set_page_config(page_title="Phone Heating Risk Prediction", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top,#0b1220,#06090f);
}

/* Main Title */
.main-title {
    text-align:center;
    color:#00e5ff;
    font-size:42px;
    font-weight:bold;
    text-shadow:0px 0px 25px rgba(0,229,255,0.6);
}

/* Glass Panel */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius:18px;
    padding:30px;
    backdrop-filter: blur(14px);
    border:1px solid rgba(255,255,255,0.08);
}

/* Section Title */
.section-title {
    font-size:22px;
    font-weight:bold;
    color:#ffffff;
    margin-bottom:15px;
    text-align:center;
    letter-spacing:1px;
}

/* Status */
.status-live {
    color:#00ff9f;
    font-weight:bold;
    text-align:center;
    margin-bottom:10px;
}

/* Info Cards */
.info-card {
    background:#111827;
    padding:18px;
    border-radius:14px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.06);
}

/* Result Card */
.big-card {
    margin-top:80px;
    padding:60px;
    border-radius:20px;
    text-align:center;
    background:#111827;
    box-shadow:0px 0px 40px red;
    color:white;
}

/* Footer */
.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "result" not in st.session_state:
    st.session_state.result = (
        "SAFE",
        "lime",
        "✅ Device temperature is stable. No action required."
    )

# ==================================================
# HOME PAGE
# ==================================================
if st.session_state.page == "home":

    st.markdown("""
    <div class="main-title">
    📱 PHONE HEATING RISK PREDICTION USING ML
    </div>
    <p class="status-live">🟢 AI Monitoring System Active</p>
    <hr>
    """, unsafe_allow_html=True)

    # -------- Device Parameters Panel --------
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">
    ⚙️ Device Parameters
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        cpu = st.slider("CPU Usage (%)", 0, 100, 50)
        temp = st.slider("Temperature (°C)", 20, 50, 35)

    with c2:
        charging = st.toggle("Charging Mode")
        brightness = st.slider("Brightness (%)", 0, 100, 60)

    with c3:
        apps = st.slider("Running Apps", 0, 15, 5)
        network = st.slider("Network Usage (%)", 0, 100, 40)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------- Predict Button --------
    if st.button("🔥 Predict Heating Risk", use_container_width=True):

        warnings = []
        critical = 0

        if cpu >= 75:
            warnings.append("CPU usage high → Close heavy applications")
            critical += 1

        if temp >= 42:
            warnings.append("Temperature critical → Stop device usage immediately")
            critical += 2

        if brightness >= 75:
            warnings.append("Brightness too high → Reduce brightness")
            critical += 1

        if apps >= 10:
            warnings.append("Too many apps → Close background apps")
            critical += 1

        if network >= 75:
            warnings.append("High network usage → Stop downloads")
            critical += 1

        if critical == 0:
            st.session_state.result = (
                "SAFE",
                "lime",
                "✅ Device temperature is stable. No action required."
            )
        elif critical <= 2:
            st.session_state.result = (
                "WARNING",
                "orange",
                "<br>".join(warnings)
            )
        else:
            st.session_state.result = (
                "CRITICAL HEATING RISK",
                "red",
                "<br>".join(warnings)
            )

        st.session_state.page = "result"
        st.rerun()

    # -------- Risk Guide --------
    st.markdown("### 🔎 Risk Level Guide")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("""
        <div class="info-card">
        🟢 <b>SAFE</b><br>
        Normal device operation
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("""
        <div class="info-card">
        🟠 <b>WARNING</b><br>
        Heating starting to increase
        </div>
        """, unsafe_allow_html=True)

    with g3:
        st.markdown("""
        <div class="info-card">
        🔴 <b>CRITICAL</b><br>
        Immediate cooling required
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
    ⚡ Machine Learning Fundamentals Project • AI Thermal Safety Monitor
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# RESULT PAGE
# ==================================================
elif st.session_state.page == "result":

    status, color, message = st.session_state.result

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    action_text = ""
    if status != "SAFE":
        action_text = "<h3>⚠ Take Action Immediately</h3>"

    st.markdown(f"""
    <div class="big-card">
        <h1 style="color:{color}; font-size:60px;">{status}</h1>
        <p style="font-size:22px;">{message}</p>
        {action_text}
    </div>
    """, unsafe_allow_html=True)
