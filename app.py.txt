import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("house_price_model.pkl", "rb"))
data = pd.read_csv("Cleaned_data.csv")

st.title("🏠 House Price Prediction App")

location = st.selectbox("Select Location", sorted(data['location'].unique()))
sqft = st.number_input("Total Square Feet", min_value=300)
bath = st.number_input("Bathrooms", min_value=1)
bhk = st.number_input("BHK", min_value=1)

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[location, sqft, bath, bhk]],
        columns=['location', 'total_sqft', 'bath', 'bhk']
    )

    price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {round(price,2)} Lakhs")
