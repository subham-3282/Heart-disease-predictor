import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from fpdf import FPDF

st.set_page_config(
    page_title="CardioGuard: Analysis Report",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======= SESSION STATE CHECK =======
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

is_dark = st.session_state.dark_mode

# Theme colors
text_color = "#ffffff" if is_dark else "#1a1a2e"
bg_color = "#1a1a2e" if is_dark else "#f5f5f5"
card_bg = "rgba(255,255,255,0.07)" if is_dark else "rgba(255,255,255,0.85)"
card_border = "rgba(255,255,255,0.15)" if is_dark else "rgba(0,0,0,0.1)"
subtitle = "#bbbbbb" if is_dark else "#555555"

# Styling
st.markdown(f"""
<style>
.stApp {{
    background: {bg_color} !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: {text_color} !important;
}}
h1, h2, h3, h4, h5, h6 {{ color: {text_color} !important; }}
p, span, label, div[data-testid="stMarkdownContainer"] p {{ color: {text_color} !important; }}

.report-title {{
    text-align: center;
    font-size: 2.5rem;
    font-weight: 900;
    background: linear-gradient(to right, #ff4b6e, #ff8a5c, #ffd700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}
.report-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}}

div[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1.1rem !important; padding: 0.8rem 2rem !important;
    transition: all 0.3s ease !important;
}}
div[data-testid="stDownloadButton"] > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(118, 75, 162, 0.5) !important;
}}

section[data-testid="stSidebar"] {{
    background: {'#1a1a2e' if is_dark else '#f5f5f5'} !important;
}}
section[data-testid="stSidebar"] * {{ color: {text_color} !important; text-shadow: none !important; }}
</style>
""", unsafe_allow_html=True)


# ======= PDF GENERATOR =======
def create_full_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style='B', size=20)
    pdf.cell(0, 15, "CardioGuard", align='C', ln=1)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "AI Heart Disease Analysis Report", align='C', ln=1)
    pdf.ln(10)
    pdf.set_draw_color(229, 46, 113)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Date & Time: {data['timestamp']}", ln=1)
    pdf.ln(8)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "1. Prediction Result", ln=1)
    pdf.set_font("Helvetica", size=11)
    s = "No Heart Disease Detected (Healthy)" if data['is_healthy'] else "Heart Disease Risk Detected"
    pdf.cell(0, 8, f"   Status: {s}", ln=1)
    pdf.ln(6)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "2. Risk Probability & Confidence", ln=1)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"   Model Confidence: {data['confidence']}%", ln=1)
    pdf.ln(6)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "3. Input Health Parameters", ln=1)
    pdf.set_font("Helvetica", size=11)
    params = data['input_params']
    for label, value in [
        ("Age", f"{params['age']} years"), ("Sex", "Male" if params['sex']==1 else "Female"),
        ("Chest Pain Type", str(params['cp'])), ("Resting BP", f"{params['trestbps']} mm Hg"),
        ("Cholesterol", f"{params['chol']} mg/dl"), ("Fasting BS>120", "Yes" if params['fbs']==1 else "No"),
        ("Resting ECG", str(params['restecg'])), ("Max HR", f"{params['thalach']} bpm"),
        ("Exercise Angina", "Yes" if params['exang']==1 else "No"),
        ("Oldpeak", str(params['oldpeak'])), ("Slope", str(params['slope'])),
        ("CA", str(params['ca'])), ("Thalassemia", str(params['thal']))
    ]:
        pdf.cell(80, 7, f"   {label}:")
        pdf.cell(0, 7, value, ln=1)
    pdf.ln(5)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "4. Key Feature Importance", ln=1)
    pdf.set_font("Helvetica", size=11)
    for f, i in [("Thalassemia","18%"),("CA","16%"),("Chest Pain","13%"),("Max HR","11%"),("Oldpeak","10%")]:
        pdf.cell(0, 7, f"   - {f}: {i}", ln=1)
    pdf.ln(5)

    pdf.set_font("Helvetica", style='B', size=13)
    pdf.cell(0, 10, "5. Health Suggestions", ln=1)
    pdf.set_font("Helvetica", size=11)
    tips = ["Maintain healthy lifestyle.", "150+ min exercise/week.", "Annual check-ups."] if data['is_healthy'] else [
        "Consult cardiologist.", "Adopt DASH diet.", "30 min exercise daily.",
        "Monitor BP & cholesterol.", "Manage stress.", "Avoid smoking."]
    for n, t in enumerate(tips, 1):
        pdf.cell(0, 7, f"   {n}. {t}", ln=1)

    pdf.ln(10)
    pdf.set_draw_color(229, 46, 113)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica", style='I', size=9)
    pdf.multi_cell(0, 6, "Disclaimer: AI-generated report. NOT a substitute for professional medical advice.")
    return bytes(pdf.output())


# ======= SIDEBAR — PDF download persists here =======
with st.sidebar:
    st.markdown("##### 🏠 [Back to Dashboard](./)")
    if st.session_state.last_prediction is not None:
        st.markdown("---")
        st.markdown("##### 📑 Export Report")
        try:
            pdf = create_full_report(st.session_state.last_prediction)
            st.download_button(
                "📄 Download Final Analysis Report", data=pdf,
                file_name=f"CardioGuard_Analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf", use_container_width=True
            )
        except Exception as e:
            st.error(f"Error: {e}")

# ======= MAIN CONTENT =======
st.markdown("<h1 class='report-title'>📑 Analysis Report</h1>", unsafe_allow_html=True)

if st.session_state.last_prediction is None:
    st.markdown("---")
    st.warning("⚠️ **No prediction data available.** Please go to the main dashboard and make a prediction first.")
    st.markdown("[🏠 Go to Dashboard](./)")
    st.stop()

pred = st.session_state.last_prediction
params = pred['input_params']

# ===== Overview Cards =====
st.markdown("### 🔍 Prediction Overview")
c1, c2, c3 = st.columns(3)
with c1:
    status_emoji = "✅" if pred['is_healthy'] else "⚠️"
    status_text = "Healthy Heart" if pred['is_healthy'] else "Heart Risk"
    st.metric("Status", f"{status_emoji} {status_text}")
with c2:
    st.metric("AI Confidence", f"{pred['confidence']}%")
with c3:
    st.metric("Prediction Time", pred['timestamp'].split(' ')[1] if ' ' in pred['timestamp'] else pred['timestamp'])

# ===== Patient Parameters Table =====
st.markdown("---")
st.markdown("### 📋 Patient Health Parameters")

param_df = pd.DataFrame({
    "Parameter": ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                   "Fasting Blood Sugar>120", "Resting ECG", "Max Heart Rate",
                   "Exercise Angina", "ST Depression", "Slope", "Major Vessels (CA)", "Thalassemia"],
    "Value": [
        f"{params['age']} years", "Male" if params['sex']==1 else "Female",
        ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"][params['cp']],
        f"{params['trestbps']} mm Hg", f"{params['chol']} mg/dl",
        "Yes" if params['fbs']==1 else "No",
        ["Normal","ST-T Abnormality","LV Hypertrophy"][params['restecg']],
        f"{params['thalach']} bpm", "Yes" if params['exang']==1 else "No",
        str(params['oldpeak']),
        ["Upsloping","Flat","Downsloping"][params['slope']],
        str(params['ca']),
        ["Normal","Fixed Defect","Reversible Defect","Thalassemia"][params['thal']]
    ],
    "Normal Range": [
        "—", "—", "—", "90-120 mm Hg", "<200 mg/dl",
        "No", "Normal", "120-180 bpm", "No", "<1.0", "—", "0", "Normal"
    ]
})
st.dataframe(param_df, use_container_width=True, hide_index=True)

# ===== Visualizations =====
st.markdown("---")
st.markdown("### 📊 Detailed Visualizations")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Gauge
    score = 85 if pred['is_healthy'] else 35
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=score,
        title={'text': "Heart Health Score", 'font': {'color': text_color}},
        number={'font': {'color': text_color}},
        delta={'reference': 70, 'increasing': {'color': '#00cc66'}, 'decreasing': {'color': '#ff4b4b'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': text_color, 'tickfont': {'color': text_color}},
            'bar': {'color': '#00cc66' if pred['is_healthy'] else '#ff4b4b'},
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255,75,75,0.2)'},
                {'range': [40, 70], 'color': 'rgba(255,193,7,0.2)'},
                {'range': [70, 100], 'color': 'rgba(0,204,102,0.2)'}],
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font={'color': text_color},
        height=300, margin=dict(l=30, r=30, t=60, b=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with viz_col2:
    # Vitals vs Healthy
    vitals = pd.DataFrame({
        "Metric": ["Max HR", "Max HR", "BP", "BP", "Chol", "Chol"],
        "Who": ["You", "Healthy", "You", "Healthy", "You", "Healthy"],
        "Value": [params['thalach'], 158.6, params['trestbps'], 129.2, params['chol'], 242.6]
    })
    fig_v = px.bar(vitals, x="Metric", y="Value", color="Who", barmode="group",
                   color_discrete_map={"You": "#ff4b6e", "Healthy": "#7c4dff"},
                   text="Value", title="Your Vitals vs Healthy Averages")
    fig_v.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': text_color}, height=300)
    fig_v.update_traces(textposition='outside')
    st.plotly_chart(fig_v, use_container_width=True)

# Feature Importance (full width)
st.markdown("### 🧠 Feature Importance Analysis")
features = ["Thalassemia", "CA (Vessels)", "Chest Pain", "Max HR", "Oldpeak",
            "Angina", "Slope", "Sex", "BP", "Cholesterol", "ECG", "Age", "Fast BS"]
importances = [0.18, 0.16, 0.13, 0.11, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01]
df_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True)
fig_imp = px.bar(df_imp, x="Importance", y="Feature", orientation='h',
                 color="Importance", color_continuous_scale="YlOrRd",
                 title="Which factors most influenced the AI prediction?")
fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': text_color}, height=420)
st.plotly_chart(fig_imp, use_container_width=True)

# Radar Chart — Patient Risk Profile
st.markdown("### 🎯 Risk Profile Radar")
radar_categories = ['BP Risk', 'Cholesterol Risk', 'Heart Rate Score',
                     'Exercise Tolerance', 'Vascular Health', 'Genetic Risk']
# Normalize values to 0-1 scale for radar
bp_risk = min(params['trestbps'] / 200, 1.0)
chol_risk = min(params['chol'] / 400, 1.0)
hr_score = 1 - min(params['thalach'] / 200, 1.0)
ex_tol = params['exang']
vasc = min(params['ca'] / 4, 1.0)
gen_risk = min(params['thal'] / 3, 1.0)
radar_vals = [bp_risk, chol_risk, hr_score, ex_tol, vasc, gen_risk]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_vals + [radar_vals[0]],
    theta=radar_categories + [radar_categories[0]],
    fill='toself',
    fillcolor='rgba(255,75,110,0.3)' if not pred['is_healthy'] else 'rgba(0,229,255,0.3)',
    line=dict(color='#ff4b6e' if not pred['is_healthy'] else '#00e5ff', width=2),
    name='Risk Profile'
))
fig_radar.update_layout(
    polar=dict(
        bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(visible=True, range=[0, 1], tickfont={'color': text_color}),
        angularaxis=dict(tickfont={'color': text_color})
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font={'color': text_color},
    height=400,
    showlegend=False
)
st.plotly_chart(fig_radar, use_container_width=True)

# Session History
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 Full Session History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

# AI Suggestions
st.markdown("---")
st.markdown("### 🤖 AI Health Recommendations")
if pred['is_healthy']:
    st.success(
        "**Your heart health is excellent!** Continue regular exercise, balanced nutrition, "
        "and annual cardiovascular check-ups. Stay proactive about your heart health."
    )
else:
    st.warning(
        "**Elevated risk indicators detected.** Key recommendations:\n"
        "1. 🩺 Schedule a cardiologist consultation\n"
        "2. 🥗 Adopt a heart-healthy diet (DASH/Mediterranean)\n"
        "3. 🏃 30+ min moderate exercise daily\n"
        "4. 📊 Monitor BP & cholesterol regularly\n"
        "5. 🧘 Stress management through mindfulness\n\n"
        "⚠️ *This is an AI screening tool — always consult a healthcare professional.*"
    )

# PDF Download at bottom
st.markdown("---")
col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
with col_dl2:
    try:
        pdf = create_full_report(st.session_state.last_prediction)
        st.download_button(
            "📄 Download Complete Analysis Report", data=pdf,
            file_name=f"CardioGuard_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", use_container_width=True
        )
    except:
        pass
