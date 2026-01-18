import streamlit as st
import pandas as pd
import pickle
import os

# ====================================
# 1. LOAD THE TRAINED MODEL
# ====================================

# Get current working directory
current_dir = os.getcwd()

# Build full path of model file
model_path = os.path.join(current_dir, "house_price_model.pkl")

# Load model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# ====================================
# 2. LOAD CLEANED DATA (FOR LOCATIONS)
# ====================================

data_path = os.path.join(current_dir, "Cleaned_data.csv")
data = pd.read_csv(data_path)

# ====================================
# 3. STREAMLIT PAGE SETTINGS
# ====================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Bangalore House Price Prediction")
st.write("Fill the details below to predict house price")

# ====================================
# 4. USER INPUTS
# ====================================

location = st.selectbox(
    "Select Location",
    sorted(data["location"].unique())
)

total_sqft = st.number_input(
    "Total Square Feet",
    min_value=300.0,
    step=50.0
)

bath = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    step=1
)

bhk = st.number_input(
    "Number of BHK",
    min_value=1,
    step=1
)

# ====================================
# 5. PREDICTION
# ====================================

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[location, total_sqft, bath, bhk]],
        columns=["location", "total_sqft", "bath", "bhk"]
    )

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated House Price: ₹ {round(prediction, 2)} Lakhs")


