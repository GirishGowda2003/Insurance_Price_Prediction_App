import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load model ────────────────────────────────────────────────────────────────
try:
    with open('gb_model (1).pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ 'gb_model (1).pkl' not found. Place it in the same folder as app.py.")
    st.stop()
except EOFError:
    st.error("❌ 'gb_model (1).pkl' is empty or corrupted. Re-save it from your training notebook.")
    st.stop()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title('🏥 Insurance Price Prediction App')

age      = st.number_input('Age',      min_value=1,    max_value=100,  value=25)
gender   = st.selectbox('Gender',      ('male', 'female'))
bmi      = st.number_input('BMI',      min_value=10.0, max_value=80.0, value=30.0)
smoker   = st.selectbox('Smoker',      ('yes', 'no'))
children = st.number_input('Children', min_value=0,    max_value=10,   value=2)
region   = st.selectbox('Region',      ('southwest', 'southeast', 'northwest', 'northeast'))

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

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button('Predict'):
    prediction = model.predict(input_features)
    output = round(np.exp(prediction[0]), 2)
    st.success(f'💰 Predicted Insurance Price: **${output:,.2f}**')