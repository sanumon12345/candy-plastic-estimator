# app.py - run with: python -m streamlit run app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Candy Eco-Plastic Estimator", 
    page_icon="🍬", 
    layout="wide"
)

# --- Load Trained Pipeline ---
@st.cache_resource
def load_model():
    return joblib.load("models/candy_plastic_model.joblib")

bundle = load_model()
pipeline = bundle["pipeline"]

st.title("🍬 Candy Company Plastic Waste & Sales Predictor")
st.markdown("Estimate future candy sales volume and calculate total environmental plastic footprint.")

# --- SIDEBAR: Input Parameters ---
st.sidebar.header("📊 Production & Market Inputs")

target_year = st.sidebar.slider("Target Production Year", 2025, 2030, 2026, 1)
marketing_budget = st.sidebar.slider("Marketing Budget ($)", 5000, 100000, 25000, 1000)
unit_price = st.sidebar.slider("Candy Unit Price ($)", 0.50, 5.00, 1.50, 0.10)
plastic_per_candy = st.sidebar.slider("Plastic per Candy (grams)", 0.2, 5.0, 1.2, 0.1)

# --- Build Sample DataFrame for ML Model ---
sample = pd.DataFrame([[marketing_budget, unit_price, plastic_per_candy, target_year]],
                      columns=bundle["features"])

# --- PREDICTION & CALCULATIONS ---
predicted_sales = int(pipeline.predict(sample)[0])
total_plastic_grams = predicted_sales * plastic_per_candy
total_plastic_kg = total_plastic_grams / 1000.0
total_plastic_tons = total_plastic_kg / 1000.0

# --- DISPLAY METRICS ---
st.subheader("📌 Forecasted Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Predicted Candy Sales", value=f"{predicted_sales:,} units")
with col2:
    st.metric(label="Total Plastic Waste (KG)", value=f"{total_plastic_kg:,.2f} kg")
with col3:
    st.metric(label="Plastic Waste (Metric Tons)", value=f"{total_plastic_tons:,.3f} Tons")

st.divider()

# --- ENVIRONMENTAL IMPACT LEVEL ---
st.subheader("🌍 Environmental Impact Assessment")
if total_plastic_kg < 100:
    st.success("🟢 Low Eco-Footprint: Plastic waste is well controlled.")
elif total_plastic_kg < 300:
    st.warning("🟡 Moderate Footprint: Consider reducing packaging plastic thickness.")
else:
    st.error("🔴 High Footprint Alert! High volume of plastic waste. Sustainable packaging alternatives recommended.")

st.divider()

# --- FUTURE PROJECTION & COMPARISON CHARTS ---
st.subheader("📈 Multi-Year Growth & Plastic Impact Projection")
st.write("Simulating company growth from 2025 to 2030 based on your current settings:")

future_years = list(range(2025, 2031))
future_sales = []
future_plastic_tons = []

# Simulate gradual market expansion year-over-year
for i, yr in enumerate(future_years):
    temp_budget = marketing_budget * (1 + (i * 0.05))
    temp_sample = pd.DataFrame([[temp_budget, unit_price, plastic_per_candy, yr]], 
                               columns=bundle["features"])
    s_pred = int(pipeline.predict(temp_sample)[0])
    p_pred_tons = (s_pred * plastic_per_candy) / 1000000.0
    
    future_sales.append(s_pred)
    future_plastic_tons.append(p_pred_tons)

# Create two columns for separate, clean charts
chart_col1, chart_col2 = st.columns(2)

sales_df = pd.DataFrame({
    "Year": [str(y) for y in future_years],
    "Projected Sales (Units)": future_sales
}).set_index("Year")

plastic_df = pd.DataFrame({
    "Year": [str(y) for y in future_years],
    "Plastic Waste (Metric Tons)": future_plastic_tons
}).set_index("Year")

with chart_col1:
    st.markdown("##### 📊 Projected Sales Volume")
    st.bar_chart(sales_df)

with chart_col2:
    st.markdown("##### 🗑️ Projected Plastic Waste")
    st.bar_chart(plastic_df, color="#ff4b4b")

# --- FOOTER ---
st.caption(f"Model Performance: R² Score = {bundle['r2_score']:.2%}, MAE = {bundle['mae']:.0f} units.")