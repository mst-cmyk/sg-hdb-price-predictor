import streamlit as st
import pandas as pd
import numpy as np
import joblib
import category_encoders as ce

# ==========================================
# 1. UI Configuration
# ==========================================
st.set_page_config(page_title="HDB Valuation AI", page_icon="🏢", layout="centered")
st.title("🏢 Singapore HDB Resale Valuation Engine")
st.markdown("Powered by an **XGBoost Regression Model** for real-time property valuation.")
st.divider()

# ==========================================
# 2. Load Models & Tools (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load('xgb_model.joblib')
    encoder = joblib.load('target_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    cols = joblib.load('X_train_cols.joblib')
    return model, encoder, scaler, cols

model, encoder, scaler, cols = load_models()

# ==========================================
# 3. Interactive Input Form
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Property Details")
    town = st.selectbox("Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
    flat_type = st.selectbox("Flat Type", ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'])
    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30, max_value=250, value=90)
    mid_storey = st.slider("Floor Level", min_value=1, max_value=50, value=8)

with col2:
    st.subheader("📅 Lease & Timing")
    flat_model = st.selectbox("Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Premium Apartment', 'Maisonette', 'Apartment'])
    lease_commence_date = st.number_input("Lease Commence Year", min_value=1966, max_value=2024, value=2015)
    sale_year = st.selectbox("Planned Sale Year", [2025, 2026])
    sale_month = st.slider("Planned Sale Month", 1, 12, 6)

# ==========================================
# 4. Prediction Logic
# ==========================================
st.divider()
if st.button("🚀 Calculate AI Valuation", use_container_width=True):
    with st.spinner("Analyzing historical transaction data..."):
        
        # Calculate derived features
        flat_age = sale_year - lease_commence_date
        remaining_lease_years = 99 - flat_age
        
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            'town': [town],
            'flat_type': [flat_type],
            'floor_area_sqm': [floor_area_sqm],
            'flat_model': [flat_model],
            'sale_year': [sale_year],
            'sale_month': [sale_month],
            'flat_age': [flat_age],
            'remaining_lease_years': [remaining_lease_years],
            'mid_storey': [mid_storey]
        })

        # Apply the exact same preprocessing pipeline
        input_data[['town', 'flat_model']] = encoder.transform(input_data[['town', 'flat_model']])
        input_data = pd.get_dummies(input_data, columns=['flat_type'])
        
        # Align columns to match the training data exactly (fills missing flat_types with 0)
        input_data = input_data.reindex(columns=cols, fill_value=0) 
        
        numerical_cols = ['floor_area_sqm', 'sale_year', 'sale_month', 'flat_age', 'remaining_lease_years', 'mid_storey']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Make the prediction and inverse the log transformation
        log_pred = model.predict(input_data)
        actual_pred = np.expm1(log_pred)[0]

        # Display the result elegantly
        st.success("Prediction Complete! Based on market trends, the estimated value is:")
        st.metric(label="Estimated Resale Price", value=f"SGD ${actual_pred:,.2f}")