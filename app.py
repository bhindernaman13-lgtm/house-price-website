import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ====================================
# PATH SETUP (STREAMLIT SAFE)
# ====================================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "Cleaned_data.csv"

# ====================================
# DEBUG: CHECK FILES EXIST
# ====================================

if not MODEL_PATH.exists():
    st.error("❌ Model file not found: house_price_model.pkl")
    st.write("Files in directory:", list(BASE_DIR.iterdir()))
    st.stop()

if not DATA_PATH.exists():
    st.error("❌ Data file not found: Cleaned_data.csv")
    st.stop()

# ====================================
# LOAD MODEL & DATA
# ====================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

data = pd.read_csv(DATA_PATH)

# ====================================
# UI
# ====================================

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 Bangalore House Price Prediction")

location = st.selectbox(
    "Select Location",
    sorted(data["location"].unique())
)

total_sqft = st.number_input("Total Square Feet", min_value=300.0, step=50.0)
bath = st.number_input("Bathrooms", min_value=1, step=1)
bhk = st.number_input("BHK", min_value=1, step=1)

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[location, total_sqft, bath, bhk]],
        columns=["location", "total_sqft", "bath", "bhk"]
    )

    price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {round(price, 2)} Lakhs")




