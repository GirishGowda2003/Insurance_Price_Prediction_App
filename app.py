import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Price Predictor",
    page_icon="🏥",
    # layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf0;
}

.stApp {
    background: radial-gradient(ellipse at top left, #0d1b3e 0%, #0a0f1e 50%, #0d1b2a 100%);
    min-height: 100vh;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00c6ff22, #0072ff22);
    border: 1px solid #0072ff55;
    color: #60b4ff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
}

.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    background: linear-gradient(135deg, #ffffff 30%, #60b4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    line-height: 1.15;
}

.hero p {
    color: #7a8aaa;
    font-size: 0.97rem;
    font-weight: 300;
    margin: 0;
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #0072ff44, transparent);
    margin: 1.5rem 0;
}

/* ── Card ── */
.card {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 32px #00000055, inset 0 1px 0 #ffffff08;
}

.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #0072ff;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #0072ff33, transparent);
}

/* ── Inputs ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0a0f1e !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: #0072ff !important;
    box-shadow: 0 0 0 3px #0072ff22 !important;
}

label, .stSelectbox label, .stNumberInput label {
    color: #7a8aaa !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0072ff, #00c6ff) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px #0072ff44 !important;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px #0072ff66 !important;
    background: linear-gradient(135deg, #005ce6, #00aadd) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Success Result ── */
.result-box {
    background: linear-gradient(135deg, #003d2b, #00291e);
    border: 1px solid #00c97644;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 0 30px #00c97622;
}

.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00c976;
    margin-bottom: 0.5rem;
}

.result-amount {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    background: linear-gradient(135deg, #00ff99, #00c976);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}

.result-note {
    color: #4a7a65;
    font-size: 0.78rem;
    margin-top: 0.4rem;
}

/* ── Error Box ── */
div[data-testid="stAlert"] {
    background: #1a0a0a !important;
    border: 1px solid #ff444444 !important;
    border-radius: 12px !important;
    color: #ff8888 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #2a3a5a;
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    letter-spacing: 0.05em;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Powered Estimator</div>
    <h1>Insurance Price Predictor</h1>
    <p>Fill in your details below to get an instant premium estimate</p>
</div>
<div class="styled-divider"></div>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
try:
    with open('gb_model (1).pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ 'gb_model (1).pkl' not found. Place it in the same folder as app.py.")
    st.stop()
except EOFError:
    st.error("❌ 'gb_model (1).pkl' is empty or corrupted. Re-save it from your training notebook.")
    st.stop()

# ── Personal Info Card ────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title"><h2>👤 Personal Information</h2></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age    = st.number_input('Age',    min_value=1,   max_value=100, value=25)
    bmi    = st.number_input('BMI',    min_value=10.0, max_value=80.0, value=30.0)
    gender = st.selectbox('Gender',   ('male', 'female'))
with col2:
    smoker   = st.selectbox('Smoker',   ('yes', 'no'))
    children = st.number_input('Children', min_value=0, max_value=10, value=2)
    region   = st.selectbox('Region',  ('southwest', 'southeast', 'northwest', 'northeast'))
st.markdown('</div>', unsafe_allow_html=True)

# ── Encoding ──────────────────────────────────────────────────────────────────
smoker_encoded = 1 if smoker == 'yes' else 0
sex_male       = 1 if gender == 'male' else 0
sex_female     = 1 if gender == 'female' else 0
region_dict    = {'southeast': 3, 'northeast': 2, 'northwest': 1, 'southwest': 0}
region_encoded = region_dict[region]

# ── Build Input DataFrame ─────────────────────────────────────────────────────
input_features = pd.DataFrame({
    'age':        [age],
    'bmi':        [bmi],
    'children':   [children],
    'Smoker':     [smoker_encoded],
    'sex_female': [sex_female],
    'sex_male':   [sex_male],
    'Region':     [region_encoded]
})

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button('🔍 Calculate My Premium'):
    prediction = model.predict(input_features)
    output = round(np.exp(prediction[0]), 2)
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">📋 Estimated Annual Premium</div>
        <div class="result-amount">${output:,.2f}</div>
        <div class="result-note">Based on your provided health & demographic details</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Powered by Gradient Boosting · For estimation purposes only</div>', unsafe_allow_html=True)
