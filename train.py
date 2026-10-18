# train.py - Data cleaning, preprocessing, and training pipeline
import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("=== STEP 1: LOADING RAW DATASET ===")
raw_df = pd.read_csv("candy_sales_raw.csv")
print(f"Raw dataset loaded. Total rows: {len(raw_df)}")

# --- STEP 2: PREPROCESSING & DATA CLEANING ---
print("\n=== STEP 2: PREPROCESSING & CLEANING ===")

df = raw_df.copy()

# A. Clean formatting issues (convert strings with '$', ',', 'USD' to numeric)
df['marketing_budget'] = df['marketing_budget'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['marketing_budget'] = pd.to_numeric(df['marketing_budget'], errors='coerce')

df['unit_price'] = df['unit_price'].astype(str).str.replace(' USD', '', regex=False)
df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')

df['plastic_per_candy_g'] = pd.to_numeric(df['plastic_per_candy_g'], errors='coerce')
df['candy_sales'] = pd.to_numeric(df['candy_sales'], errors='coerce')

print("1. String formatting cleaned (Currency & Text units removed).")

# B. Remove duplicates
initial_len = len(df)
df = df.drop_duplicates()
print(f"2. Removed {initial_len - len(df)} duplicate rows.")

# C. Handle missing / null values
missing_count = df.isnull().sum().sum()
df = df.dropna()
print(f"3. Handled missing/null values (Removed {missing_count} total null cells).")

# D. Remove invalid numbers (e.g. negative prices or negative plastic)
invalid_mask = (df['unit_price'] < 0) | (df['plastic_per_candy_g'] < 0)
invalid_count = invalid_mask.sum()
df = df[~invalid_mask]
print(f"4. Removed {invalid_count} rows with invalid negative numeric values.")

print(f"\nFinal Cleaned Dataset Size: {len(df)} rows.")

# Save cleaned dataset for inspection
df.to_csv("candy_sales_cleaned.csv", index=False)
print("Saved cleaned data -> 'candy_sales_cleaned.csv'")

# --- STEP 3: FEATURE SELECTION & TRAIN-TEST SPLIT ---
feature_names = ["marketing_budget", "unit_price", "plastic_per_candy_g", "production_year"]
X = df[feature_names]
y = df["candy_sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- STEP 4: MODEL TRAINING PIPELINE ---
print("\n=== STEP 4: TRAINING MACHINE LEARNING PIPELINE ===")
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(n_estimators=200, random_state=42)
)
pipeline.fit(X_train, y_train)

# --- STEP 5: EVALUATION ---
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} units")

# --- STEP 6: SAVE MODEL BUNDLE ---
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({
    "pipeline": pipeline,
    "features": feature_names,
    "r2_score": r2,
    "mae": mae
}, "models/candy_plastic_model.joblib")

print("\nModel saved successfully -> models/candy_plastic_model.joblib")