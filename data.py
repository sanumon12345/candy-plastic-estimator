# generate_data.py - Run ONCE to generate dirty dataset
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# 1. Generate clean base features
budgets = np.random.randint(5000, 60000, n).astype(object)
prices = np.round(np.random.uniform(0.5, 4.5, n), 2).astype(object)
plastics = np.round(np.random.uniform(0.5, 3.0, n), 1).astype(object)
years = np.random.choice([2023, 2024, 2025, 2026], n).astype(object)

# Realistic Target variable formula
sales = (budgets.astype(float) * 12 - prices.astype(float) * 1500 + np.random.normal(0, 10000, n)).clip(10000).astype(object)

df = pd.DataFrame({
    "marketing_budget": budgets,
    "unit_price": prices,
    "plastic_per_candy_g": plastics,
    "production_year": years,
    "candy_sales": sales
})

# 2. Inject 100+ intentional data quality issues

# A. Missing / Null values (~50 instances)
missing_indices = np.random.choice(n, 50, replace=False)
for idx in missing_indices[:15]:
    df.loc[idx, "marketing_budget"] = np.nan
for idx in missing_indices[15:30]:
    df.loc[idx, "unit_price"] = None
for idx in missing_indices[30:50]:
    df.loc[idx, "candy_sales"] = np.nan

# B. Formatting issues / String noise (~30 instances)
format_indices = np.random.choice(n, 30, replace=False)
for idx in format_indices[:15]:
    df.loc[idx, "marketing_budget"] = f"${df.loc[idx, 'marketing_budget']:,}"  # e.g., "$25,000"
for idx in format_indices[15:]:
    df.loc[idx, "unit_price"] = f"{df.loc[idx, 'unit_price']} USD"             # e.g., "1.5 USD"

# C. Invalid / Anomaly numbers (~20 instances)
invalid_indices = np.random.choice(n, 20, replace=False)
for idx in invalid_indices[:10]:
    df.loc[idx, "unit_price"] = -5.00  # Impossible negative price
for idx in invalid_indices[10:]:
    df.loc[idx, "plastic_per_candy_g"] = -1.2  # Impossible negative plastic

# D. Duplicate rows (~15 instances)
duplicates = df.iloc[0:15].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# Save to raw CSV
df.to_csv("candy_sales_raw.csv", index=False)
print("Generated 'candy_sales_raw.csv' with 1,000+ rows and over 100 intentional errors!")