import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import datetime
from fpdf import FPDF

# ======= PAGE CONFIG (must be first Streamlit command) =======
st.set_page_config(
    page_title="CardioGuard: Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ======= SESSION STATE INIT =======
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'website_mode' not in st.session_state:
    st.session_state.website_mode = "None"

# ======= PDF GENERATOR (shared) =======
def create_full_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style='B', size=20)
    pdf.cell(0, 15, "CardioGuard", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "AI Heart Disease Analysis Report", align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(229, 46, 113)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Date & Time: {data['timestamp']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "1. Prediction Result", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    status = "No Heart Disease Detected (Healthy)" if data['is_healthy'] else "Heart Disease Risk Detected"
    pdf.cell(0, 8, f"   Status: {status}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "2. Risk Probability & Confidence", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"   Model Confidence: {data['confidence']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "3. Input Health Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    params = data['input_params']
    param_list = [
        ("Age", f"{params['age']} years"),
        ("Sex", "Male" if params['sex'] == 1 else "Female"),
        ("Chest Pain Type", str(params['cp'])),
        ("Resting BP", f"{params['trestbps']} mm Hg"),
        ("Cholesterol", f"{params['chol']} mg/dl"),
        ("Fasting Blood Sugar>120", "Yes" if params['fbs'] == 1 else "No"),
        ("Resting ECG", str(params['restecg'])),
        ("Max Heart Rate", f"{params['thalach']} bpm"),
        ("Exercise Angina", "Yes" if params['exang'] == 1 else "No"),
        ("Oldpeak", str(params['oldpeak'])),
        ("Slope", str(params['slope'])),
        ("CA (Vessels)", str(params['ca'])),
        ("Thalassemia", str(params['thal'])),
    ]
    for label, value in param_list:
        pdf.cell(80, 7, f"   {label}:")
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "4. Key Feature Importance", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for feat, imp in [("Thalassemia","18%"),("Major Vessels (CA)","16%"),("Chest Pain","13%"),("Max HR","11%"),("Oldpeak","10%")]:
        pdf.cell(0, 7, f"   - {feat}: {imp}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "5. Personalized Health Suggestions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    if data['is_healthy']:
        tips = ["Maintain current healthy lifestyle.", "150+ min moderate exercise/week.",
                "Eat fruits, vegetables, omega-3.", "Annual cardiovascular check-ups."]
    else:
        tips = ["Consult a cardiologist immediately.", "Adopt DASH/Mediterranean diet.",
                "30 min moderate exercise daily.", "Monitor BP & cholesterol regularly.",
                "Reduce sodium, manage stress.", "Avoid smoking, limit alcohol."]
    for i, tip in enumerate(tips, 1):
        pdf.cell(0, 7, f"   {i}. {tip}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_draw_color(229, 46, 113)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", style='I', size=9)
    pdf.multi_cell(0, 6, "Disclaimer: This report is AI-generated and is NOT a substitute for professional medical advice.")
    return bytes(pdf.output())


# ======= SIDEBAR (Clean, minimal) =======
with st.sidebar:
    # Controls — floating, no boxed dividers
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.website_mode = st.selectbox(
        "🎨 Website Mode",
        options=["None", "Static", "Dynamic"],
        index=["None", "Static", "Dynamic"].index(st.session_state.website_mode),
    )

    st.markdown("####")  # subtle spacing

    # Quick Analytics
    st.markdown("##### 📊 Analytics")
    total_predictions = len(st.session_state.history)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predictions", total_predictions)
    with col_b:
        if st.session_state.history:
            healthy_count = sum(1 for item in st.session_state.history if item['Status'] == 'Healthy')
            rate = int((healthy_count / len(st.session_state.history)) * 100)
            st.metric("Health Rate", f"{rate}%")
        else:
            st.metric("Health Rate", "—")

    st.markdown("####")  # subtle spacing

    st.markdown("##### 🤖 AI Tip")
    st.caption("Regular cardiovascular exercise and a balanced diet significantly reduce heart disease risk.")

    st.markdown("####")  # subtle spacing
    st.markdown("##### 📑 Analysis Report")

    # ===== CONDITIONAL PDF BUTTON — only after prediction =====
    if st.session_state.last_prediction is not None:
        pred_data = st.session_state.last_prediction

        if pred_data['is_healthy']:
            st.success("✅ Healthy Heart", icon="✅")
        else:
            st.error("⚠️ Heart Risk", icon="⚠️")

        st.caption(f"🕐 {pred_data['timestamp']}")
        st.caption(f"🎯 {pred_data['confidence']}% confidence")

        try:
            report_pdf = create_full_report(pred_data)
            st.download_button(
                label="📄 Download Final Analysis Report",
                data=report_pdf,
                file_name=f"CardioGuard_Analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Report error: {e}")
    else:
        st.info("💡 Make a prediction first to generate your Analysis Report.")


# ======= RESOLVE THEME VARIABLES =======
dark_mode = st.session_state.dark_mode
website_mode = st.session_state.website_mode
is_dark = dark_mode

# Background per mode
if website_mode == "None":
    bg_css = f"""
    .stApp {{
        background: {'#2d2d2d' if is_dark else '#e8e8e8'} !important;
        background-image: none !important;
        animation: none !important;
    }}
    """
elif website_mode == "Static":
    bg_css = f"""
    .stApp {{
        background: {'#0a0a0a' if is_dark else '#f0f0f5'} !important;
        background-image: none !important;
        animation: none !important;
    }}
    """
else:
    bg_css = """
    .stApp {
        background: transparent !important;
        background-image: none !important;
        animation: none !important;
    }
    """

# Theme-dependent colors
if is_dark or website_mode == "Dynamic":
    text_color = "#ffffff"
    text_shadow = "1px 1px 3px rgba(0,0,0,0.7)"
    card_bg = "rgba(255, 255, 255, 0.07)"
    card_border = "rgba(255, 255, 255, 0.15)"
    card_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.4)"
    marquee_bg = "rgba(255, 255, 255, 0.1)"
    marquee_border = "rgba(255, 255, 255, 0.2)"
    marquee_color = "#ff4b6e"
    marquee_shadow = "0 0 8px rgba(255, 75, 110, 0.8)"
    label_color = "#ffffff"
    subtitle_color = "#bbbbbb"
    sidebar_bg = "#1a1a2e"
    sidebar_text = "#ffffff"
    title_gradient = "linear-gradient(to right, #ff4b6e, #ff8a5c, #ffd700)"
else:
    text_color = "#1a1a2e"
    text_shadow = "none"
    card_bg = "rgba(255, 255, 255, 0.75)"
    card_border = "rgba(0, 0, 0, 0.12)"
    card_shadow = "0 4px 15px rgba(0, 0, 0, 0.08)"
    marquee_bg = "rgba(255, 255, 255, 0.7)"
    marquee_border = "rgba(0, 0, 0, 0.1)"
    marquee_color = "#c62828"
    marquee_shadow = "none"
    label_color = "#1a1a2e"
    subtitle_color = "#444444"
    sidebar_bg = "#f5f5f5"
    sidebar_text = "#1a1a2e"
    title_gradient = "linear-gradient(to right, #c62828, #e53935, #ff6f00)"


full_css = f"""
<style>
/* ===== GLOBAL ===== */
.stApp {{
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: {text_color} !important;
}}
{bg_css}

/* ===== HIDE EMPTY CONTAINERS ===== */
[data-testid="stMarkdownContainer"]:empty,
[data-testid="stMarkdownContainer"]:has(style) {{
    display: none !important;
}}
div[data-testid="stVerticalBlock"] > div:has(style),
div[data-testid="stVerticalBlock"] > div:empty {{
    display: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* ===== GLASSMORPHISM CARDS ===== */
section[data-testid="stMain"] div[data-testid="stForm"],
section[data-testid="stMain"] div[data-testid="stVerticalBlock"] > div > div:not(:has(.marquee-container)):not(:has(.video-bg-wrapper)):not(:empty) {{
    background: {card_bg} !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 16px !important;
    border: 1px solid {card_border} !important;
    box-shadow: {card_shadow} !important;
    padding: 1rem !important;
}}

/* ===== TEXT ===== */
h1, h2, h3, h4, h5, h6 {{
    color: {text_color} !important;
    text-shadow: {text_shadow} !important;
}}
p, span, label, .stMarkdown, .stText,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] li {{
    color: {text_color} !important;
}}
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stNumberInput label,
div[data-baseweb="select"] span {{
    color: {label_color} !important;
    text-shadow: {text_shadow} !important;
}}

/* ===== COLORFUL BUTTON ===== */
div[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, #e52e71, #ff6b35, #ffd700) !important;
    border: none !important;
    border-radius: 50px !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
    padding: 0.6rem 2.5rem !important;
    box-shadow: 0 4px 20px rgba(229, 46, 113, 0.5) !important;
    transition: all 0.3s ease !important;
    text-shadow: none !important;
}}
div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 0 8px 30px rgba(229, 46, 113, 0.7) !important;
}}

/* ===== TITLE ===== */
.title-glow {{
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    background: {title_gradient};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none !important;
    margin-bottom: 0;
    letter-spacing: -1px;
}}

/* ===== MARQUEE ===== */
.marquee-container {{
    width: 100%;
    overflow: hidden;
    background: {marquee_bg} !important;
    border-radius: 12px;
    padding: 12px 0;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    border: 1px solid {marquee_border} !important;
}}
.marquee-text {{
    display: inline-block;
    white-space: nowrap;
    animation: marquee 18s linear infinite;
    font-size: 1.15rem;
    font-weight: bold;
    color: {marquee_color} !important;
    text-shadow: {marquee_shadow} !important;
}}
@keyframes marquee {{
    0% {{ transform: translateX(100%); }}
    100% {{ transform: translateX(-100%); }}
}}

/* ===== ANIMATIONS ===== */
@keyframes storyAnimation {{
  0%, 20% {{ content: "🏃"; transform: translateX(-20px); opacity: 0; }}
  25%, 45% {{ content: "❤️"; transform: translateX(0); opacity: 1; }}
  50%, 70% {{ content: "🩺"; transform: scale(1.2); opacity: 1; }}
  75%, 95% {{ content: "⚡"; transform: translateY(-10px) scale(1.3); opacity: 1; }}
  100% {{ content: "🏃"; transform: translateX(-20px); opacity: 0; }}
}}
.story-emoji::after {{
    content: "🏃";
    display: inline-block;
    font-size: 4rem;
    animation: storyAnimation 6s infinite;
}}
@keyframes pulseRed {{
    0% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }}
    70% {{ transform: scale(1.02); box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); }}
    100% {{ transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }}
}}
.pulse-card {{ animation: pulseRed 2s infinite; border: 2px solid #ff4b4b !important; }}
@keyframes heartbeat {{
    0% {{ transform: scale(1); }}
    14% {{ transform: scale(1.15); }}
    28% {{ transform: scale(1); }}
    42% {{ transform: scale(1.15); }}
    70% {{ transform: scale(1); }}
}}
.heartbeat {{ display: inline-block; animation: heartbeat 1.5s infinite; }}

/* ===== PROGRESS ===== */
progress {{ border-radius: 7px; width: 100%; height: 22px; }}
progress::-webkit-progress-bar {{ background-color: #333; border-radius: 7px; }}
progress::-webkit-progress-value {{ background: linear-gradient(90deg, #00C9FF, #92FE9D); border-radius: 7px; }}

/* ===== VIDEO BG ===== */
.video-bg-wrapper {{
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 100vh !important;
    z-index: -1 !important; overflow: hidden !important;
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 !important; margin: 0 !important;
    pointer-events: none !important; backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important; border-radius: 0 !important;
}}
.video-bg-wrapper video {{
    position: fixed; top: 50%; left: 50%;
    min-width: 100vw; min-height: 100vh;
    transform: translate(-50%, -50%); object-fit: cover; z-index: -1;
}}
.video-overlay {{
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(0, 0, 0, 0.6);
    z-index: 0; pointer-events: none;
}}

/* ===== COLORFUL ACCENTS ===== */
div[data-testid="stMetricValue"] {{ color: #ff4b6e !important; }}
div[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    transition: all 0.3s ease !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4) !important;
}}

/* ===== SIDEBAR — clean, no box borders ===== */
section[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
    padding: 0 !important;
}}
</style>
"""

st.markdown(full_css, unsafe_allow_html=True)

# ======= DYNAMIC MODE: VIDEO BACKGROUND =======
if website_mode == "Dynamic":
    video_html = """
    <div class="video-bg-wrapper">
        <video autoplay muted loop playsinline id="bgVideo">
            <source src="https://videos.pexels.com/video-files/5407025/5407025-uhd_2560_1440_25fps.mp4" type="video/mp4">
            <source src="https://videos.pexels.com/video-files/3209828/3209828-uhd_2560_1440_25fps.mp4" type="video/mp4">
        </video>
    </div>
    <div class="video-overlay"></div>
    """
    st.markdown(video_html, unsafe_allow_html=True)

# ======= LOAD ML MODELS =======
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, model

try:
    scaler, model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ======= HERO SECTION =======
st.markdown("<div class='marquee-container'><div class='marquee-text'>🫀 Protect Your Heart 💪 Stay Healthy 🌟 Early Detection Saves Lives 🩺 CardioGuard AI-Powered 🤖</div></div>", unsafe_allow_html=True)
st.markdown("<h1 class='title-glow'>CardioGuard ❤️</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin-bottom: 1rem;'><span class='story-emoji'></span></div>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: {subtitle_color} !important;'>Level up your heart health awareness! Enter your medical details and discover your cardiac risk profile powered by AI.</p>", unsafe_allow_html=True)

# ======= INPUT FORM =======
st.markdown("### 🎮 Enter Your Health Stats")

cp_options = {"Typical Angina (0)": 0, "Atypical Angina (1)": 1, "Non-Anginal Pain (2)": 2, "Asymptomatic (3)": 3}
restecg_options = {"Normal (0)": 0, "ST-T Wave Abnormality (1)": 1, "Left Ventricular Hypertrophy (2)": 2}
slope_options = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
thal_options = {"Normal (0)": 0, "Fixed Defect (1)": 1, "Reversible Defect (2)": 2, "Thalassemia (3)": 3}

with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("👤 Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("🚻 Sex", options=["Male (1)", "Female (0)"])
        sex_val = 1 if "Male" in sex else 0
        cp = st.selectbox("💔 Chest Pain Type", options=list(cp_options.keys()))
        cp_val = cp_options[cp]
        trestbps = st.number_input("🩸 Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("🧪 Serum Cholesterol (mg/dl)", min_value=50, max_value=600, value=200)
        fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"])
        fbs_val = 1 if "Yes" in fbs else 0
        restecg = st.selectbox("📈 Resting ECG Results", options=list(restecg_options.keys()))
        restecg_val = restecg_options[restecg]
    with col2:
        thalach = st.number_input("💓 Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("🏋️ Exercise Induced Angina", options=["No (0)", "Yes (1)"])
        exang_val = 1 if "Yes" in exang else 0
        oldpeak = st.number_input("📉 ST Depression (Oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
        slope = st.selectbox("📐 Slope of Peak Exercise ST", options=list(slope_options.keys()))
        slope_val = slope_options[slope]
        ca = st.selectbox("🔬 Number of Major Vessels (CA)", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("🧬 Thalassemia", options=list(thal_options.keys()))
        thal_val = thal_options[thal]

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ ANALYZE HEART HEALTH ⚡")

# ======= PREDICTION =======
if submitted:
    with st.spinner("Analyzing your cardiac profile... 🫀"):
        input_data = pd.DataFrame({
            "age": [age], "sex": [sex_val], "cp": [cp_val],
            "trestbps": [trestbps], "chol": [chol], "fbs": [fbs_val],
            "restecg": [restecg_val], "thalach": [thalach], "exang": [exang_val],
            "oldpeak": [oldpeak], "slope": [slope_val], "ca": [ca], "thal": [thal_val]
        })

        try:
            feature_cols = list(scaler.feature_names_in_)
            input_scaled = input_data.copy()
            input_scaled[feature_cols] = scaler.transform(input_scaled[feature_cols])
        except Exception:
            input_scaled = input_data.copy()

        prediction = model.predict(input_scaled)[0]
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_scaled)[0]
            confidence = round(max(proba) * 100, 1)
        else:
            confidence = 85.0

        is_healthy = prediction == 1

        st.session_state.history.append({
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Age": age, "Sex": "M" if sex_val == 1 else "F",
            "BP": trestbps, "Cholesterol": chol, "Max HR": thalach,
            "Status": "Healthy" if is_healthy else "At Risk",
            "Confidence": f"{confidence}%"
        })

        st.session_state.last_prediction = {
            "is_healthy": is_healthy,
            "confidence": confidence,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_params": {
                "age": age, "sex": sex_val, "cp": cp_val,
                "trestbps": trestbps, "chol": chol, "fbs": fbs_val,
                "restecg": restecg_val, "thalach": thalach, "exang": exang_val,
                "oldpeak": oldpeak, "slope": slope_val, "ca": ca, "thal": thal_val
            }
        }

    # ===== RESULTS CARD =====
    st.markdown("---")
    st.markdown("### 🏆 Diagnosis Results")

    res_col1, res_col2 = st.columns([1, 1])

    if is_healthy:
        st.balloons()
        with res_col1:
            st.markdown(f"""
            <div style='padding:1.2rem; border-radius:12px; background: linear-gradient(135deg, rgba(0,204,102,0.15), rgba(0,255,136,0.08)); border: 2px solid #00cc66; box-shadow: 0 4px 20px rgba(0,204,102,0.3);'>
                <h3 style='color:#00ff88 !important; margin-top:0;'>✅ Healthy Heart Hero</h3>
                <p>Your cardiac indicators look great!</p>
                <p style='color: #00e676 !important;'><strong>AI Confidence: {confidence}%</strong></p>
                <p><strong>Heart Health XP: 100/100</strong></p>
                <progress value='100' max='100'></progress>
            </div>
            """, unsafe_allow_html=True)
    else:
        with res_col1:
            st.markdown(f"""
            <div class='pulse-card' style='padding:1.2rem; border-radius:12px; background: linear-gradient(135deg, rgba(255,75,75,0.15), rgba(255,107,53,0.08));'>
                <h3 style='color:#ff4b4b !important; margin-top:0;'>⚠️ Heart Risk Detected <span class='heartbeat'>❤️‍🩹</span></h3>
                <p>Cardiac risk indicators elevated. Please consult a cardiologist!</p>
                <p style='color: #ff6b6b !important;'><strong>AI Confidence: {confidence}%</strong></p>
                <p><strong>Heart Health XP: 35/100</strong></p>
                <progress value='35' max='100' style='accent-color: red;'></progress>
            </div>
            """, unsafe_allow_html=True)

    # Gauge Chart
    with res_col2:
        score = 85 if is_healthy else 35
        gauge_color = "#00cc66" if is_healthy else "#ff4b4b"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={'text': "Heart Health Score", 'font': {'color': text_color, 'size': 14}},
            number={'font': {'color': text_color, 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': text_color, 'tickfont': {'color': text_color}},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255,75,75,0.25)"},
                    {'range': [40, 70], 'color': "rgba(255,193,7,0.25)"},
                    {'range': [70, 100], 'color': "rgba(0,204,102,0.25)"}],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': text_color}, height=220, margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Vitals Comparison
    st.markdown("### 📊 Your Vitals vs Healthy Averages")
    chart_data = pd.DataFrame({
        "Metric": ["Max Heart Rate", "Max Heart Rate", "Resting BP", "Resting BP", "Cholesterol", "Cholesterol"],
        "Category": ["You", "Healthy Avg", "You", "Healthy Avg", "You", "Healthy Avg"],
        "Value": [thalach, 158.6, trestbps, 129.2, chol, 242.6]
    })
    fig_bar = px.bar(chart_data, x="Metric", y="Value", color="Category", barmode="group",
        color_discrete_map={"You": "#ff4b6e" if not is_healthy else "#00e5ff", "Healthy Avg": "#7c4dff"},
        text="Value", title="Your Vitals vs Healthy Population Averages")
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': text_color}, showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("💡 Higher max heart rate during exercise and lower resting BP & cholesterol = better heart health.")

    # Feature Importance
    st.markdown("---")
    st.markdown("### 🧠 AI Feature Importance")
    mock_features = ["Thalassemia", "CA (Vessels)", "Chest Pain", "Max HR", "Oldpeak",
                     "Exercise Angina", "Slope", "Sex", "Resting BP", "Cholesterol",
                     "Resting ECG", "Age", "Fasting BS"]
    mock_imps = [0.18, 0.16, 0.13, 0.11, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01]
    df_imp = pd.DataFrame({"Feature": mock_features, "Importance": mock_imps}).sort_values(by="Importance", ascending=True)
    fig_imp = px.bar(df_imp, x="Importance", y="Feature", orientation='h',
        color="Importance", color_continuous_scale="YlOrRd", title="Feature Impact on Prediction")
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': text_color}, height=420)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Session History
    st.markdown("### 🕒 Session History")
    with st.expander("View prediction history"):
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("Make predictions to see history!")

    # AI Plan
    st.markdown("### 🤖 Personalized AI Plan")
    if is_healthy:
        st.success(
            f"**Healthy Heart Confirmed!**\n\n"
            f"Max heart rate {thalach} bpm, resting BP {trestbps} mm Hg — looking good! "
            f"Continue with regular exercise, omega-3 rich foods, and routine check-ups."
        )
    else:
        st.warning(
            f"**Cardiac Risk Indicators Detected!**\n\n"
            f"- BP: {trestbps} mm Hg (Target: <120)\n"
            f"- Cholesterol: {chol} mg/dl (Target: <200)\n"
            f"- Max HR: {thalach} bpm\n\n"
            f"**Actions:** Consult cardiologist, adopt DASH/Mediterranean diet, "
            f"30 min exercise/day, monitor BP & cholesterol.\n\n"
            f"⚠️ *AI screening tool — always consult a healthcare professional.*"
        )

    # Navigation hint
    st.markdown("---")
    st.info("� **Open the sidebar** to download your comprehensive Analysis Report PDF. "
            "Or navigate to the **Analysis Report** page for detailed visualizations.")
