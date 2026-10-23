import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Candy Company Plastic Waste & Sales Predictor",
    page_icon="🍬",
    layout="wide"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("models/candy_plastic_model.joblib")

model_data = load_model()
model = model_data["pipeline"]

# --- Sidebar Inputs ---
st.sidebar.header("Production & Market Inputs")

target_production_year = st.sidebar.slider("Target Production Year", 2025, 2030, 2025)
marketing_budget = st.sidebar.slider("Marketing Budget ($)", 5000, 60000, 37000)
unit_price = st.sidebar.slider("Candy Unit Price ($)", 0.5, 4.5, 1.5)
plastic_per_candy_g = st.sidebar.slider("Plastic per Candy (grams)", 0.5, 3.0, 1.2)

# --- Compute Multi-Year Projections ---
years = list(range(2025, 2031))
sales_projections = []
plastic_projections_kg = []

for yr in years:
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

# --- Main Page Layout ---
st.title("🍬 Candy Company Plastic Waste & Sales Predictor")
st.write("Estimate future candy sales volume and calculate total environmental plastic footprint.")

st.subheader(f"📌 Forecasted Key Performance Indicators ({target_production_year})")

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Candy Sales", f"{int(current_sales):,} units")
col2.metric("Total Plastic Waste (KG)", f"{current_plastic_kg:,.2f} kg")
col3.metric("Plastic Waste (Metric Tons)", f"{current_plastic_tons:,.3f} Tons")

# --- Environmental Impact Alert ---
st.subheader("🌍 Environmental Impact Assessment")
if current_plastic_tons > 0.5:
    st.error("🔴 High Footprint Alert! High volume of plastic waste. Sustainable packaging alternatives recommended.")
else:
    st.success("🟢 Eco-Friendly Footprint: Plastic waste levels are within acceptable operational limits.")

# --- Multi-Year Growth & Plastic Impact Projection Charts ---
st.subheader("📈 Multi-Year Growth & Plastic Impact Projection")
st.write(f"Simulating company growth from 2025 to 2030 based on your current settings:")

chart_data = pd.DataFrame({
    "Year": years,
    "Projected Sales": sales_projections,
    "Plastic Waste (KG)": plastic_projections_kg
})

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**Projected Sales Volume**")
    st.bar_chart(chart_data.set_index("Year")["Projected Sales"])

with col_chart2:
    st.markdown("**Projected Plastic Waste (KG)**")
    st.bar_chart(chart_data.set_index("Year")["Plastic Waste (KG)"])

# --- Footer Details ---
st.markdown("---")
st.caption(f"Model Performance: R² Score = 99.63%, MAE = 9015 units. | Project Developed by Abdul Nihal K (Roll No: 40)")