import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Customer Channel Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for "Attractive Design"
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('model3.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("🎯 Hybrid vs Online Store Predictor")
st.markdown("Enter customer metrics below to predict their preferred shopping channel.")

# Create Columns for Input Fields
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender (0=Male, 1=Female)", [0, 1])
    city_tier = st.selectbox("City Tier (1, 2, or 3)", [1, 2, 3])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=50000)

with col2:
    st.subheader("🌐 Digital Behavior")
    daily_internet_hours = st.slider("Daily Internet Hours", 0.0, 24.0, 5.0)
    social_media_hours = st.slider("Social Media Hours", 0.0, 24.0, 2.0)
    smartphone_usage_years = st.number_input("Smartphone Usage (Years)", 0, 20, 5)
    tech_savvy_score = st.slider("Tech Savvy Score", 1, 10, 5)

with col3:
    st.subheader("🛍️ Shopping Habits")
    monthly_online_orders = st.number_input("Monthly Online Orders", 0, 50, 5)
    monthly_store_visits = st.number_input("Monthly Store Visits", 0, 50, 2)
    avg_delivery_days = st.number_input("Avg Delivery Days", 1, 15, 3)

# Collapsible section for more niche features
with st.expander("Advanced Psychological & Spend Metrics"):
    c4, c5 = st.columns(2)
    with c4:
        online_payment_trust = st.slider("Online Payment Trust Score", 1, 10, 8)
        discount_sensitivity = st.slider("Discount Sensitivity", 1, 10, 5)
        impulse_buying = st.slider("Impulse Buying Score", 1, 10, 5)
        brand_loyalty = st.slider("Brand Loyalty Score", 1, 10, 5)
        avg_online_spend = st.number_input("Avg Online Spend", value=1000)
        avg_store_spend = st.number_input("Avg Store Spend", value=1000)
    with c5:
        return_frequency = st.slider("Return Frequency", 1, 10, 2)
        delivery_fee_sensitivity = st.slider("Delivery Fee Sensitivity", 1, 10, 5)
        free_return_importance = st.slider("Free Return Importance", 1, 10, 7)
        product_availability = st.slider("Product Availability Score", 1, 10, 8)
        need_touch_feel = st.slider("Need Touch/Feel Score", 1, 10, 5)
        environmental_awareness = st.slider("Environmental Awareness", 1, 10, 5)
        time_pressure = st.slider("Time Pressure Level", 1, 10, 5)

# Prepare data for prediction
# Note: Feature order must match the model's 'feature_names_in_'
features = [
    age, monthly_income, daily_internet_hours, smartphone_usage_years,
    social_media_hours, online_payment_trust, tech_savvy_score,
    monthly_online_orders, monthly_store_visits, avg_online_spend,
    avg_store_spend, discount_sensitivity, return_frequency,
    avg_delivery_days, delivery_fee_sensitivity, free_return_importance,
    product_availability, impulse_buying, need_touch_feel,
    brand_loyalty, environmental_awareness, time_pressure, gender, city_tier
]

if st.button("Predict Shopping Preference"):
    prediction = model.predict([features])
    probability = model.predict_proba([features])
    
    st.divider()
    
    result = prediction[0]
    color = "#28a745" if result == "Online Store" else "#17a2b8"
    
    st.markdown(f"""
        <div class="prediction-box" style="background-color: {color};">
            Predicted Channel: {result}
        </div>
        """, unsafe_allow_html=True)
    
    st.write(f"**Confidence:** {np.max(probability)*100:.2f}%")
