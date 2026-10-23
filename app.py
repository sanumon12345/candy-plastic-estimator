import pandas as pd
import streamlit as st
import joblib

# Load model
model_data = joblib.load("models/candy_plastic_model.joblib")
model = model_data["pipeline"]

# --- Sidebar Inputs ---
st.sidebar.header("Production & Market Inputs")
target_production_year = st.sidebar.slider("Target Production Year", 2025, 2030, 2025)
marketing_budget = st.sidebar.slider("Marketing Budget ($)", 5000, 60000, 37000)
unit_price = st.sidebar.slider("Candy Unit Price ($)", 0.5, 4.5, 1.5)
plastic_per_candy_g = st.sidebar.slider("Plastic per Candy (grams)", 0.5, 3.0, 1.2)

# --- Compute Multi-Year Projections First ---
years = list(range(2025, 2031))
sales_projections = []
plastic_projections_kg = []

for yr in years:
    # Create input row for each year
    input_df = pd.DataFrame({
        "marketing_budget": [marketing_budget],
        "unit_price": [unit_price],
        "plastic_per_candy_g": [plastic_per_candy_g],
        "production_year": [yr]
    })
    pred_sales = model.predict(input_df)[0]
    total_plastic_kg = (pred_sales * plastic_per_candy_g) / 1000.0
    
    sales_projections.append(pred_sales)
    plastic_projections_kg.append(total_plastic_kg)

# --- Sync KPI Cards with the Selected Slider Year ---
selected_index = years.index(target_production_year)
current_sales = sales_projections[selected_index]
current_plastic_kg = plastic_projections_kg[selected_index]
current_plastic_tons = current_plastic_kg / 1000.0

# --- Display KPIs ---
st.title("Candy Company Plastic Waste & Sales Predictor")
st.subheader(f"Forecasted Key Performance Indicators for {target_production_year}")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Candy Sales", f"{int(current_sales):,} units")
col2.metric("Total Plastic Waste (KG)", f"{current_plastic_kg:,.2f} kg")
col3.metric("Plastic Waste (Metric Tons)", f"{current_plastic_tons:,.3f} Tons")

# --- Display Charts ---
st.subheader("Multi-Year Growth & Plastic Impact Projection")
chart_data = pd.DataFrame({
    "Year": years,
    "Projected Sales": sales_projections,
    "Plastic Waste (KG)": plastic_projections_kg
})

st.bar_chart(chart_data.set_index("Year")["Projected Sales"])
st.bar_chart(chart_data.set_index("Year")["Plastic Waste (KG)"])