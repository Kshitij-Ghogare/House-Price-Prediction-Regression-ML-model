import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="🏠 Pune House Price Prediction", layout="centered")

st.title("🏠 Pune House Price Prediction")
st.write("Enter details of the property to estimate its price.")

# Input fields
square_feet = st.number_input("Square Feet", min_value=200, max_value=10000, step=50)
num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
property_age = st.number_input("Property Age (Years)", min_value=0, max_value=200, step=1)

area = st.selectbox("Area", ["Kothrud", "Wakad", "Baner", "Hinjewadi", "Unknown"])
parking_available = st.selectbox("Parking Available", [0, 1])  # 0 = No, 1 = Yes

if st.button("Predict Price"):
    # Make a dataframe with same column names as training
    input_df = pd.DataFrame([{
        "square_feet": square_feet,
        "num_bedrooms": num_bedrooms,
        "num_bathrooms": num_bathrooms,
        "property_age": property_age,
        "area": area,
        "parking_available": parking_available
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")
