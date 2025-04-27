import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_rf_model.pkl')

st.title("California Housing Price Prediction 🏠")

# Take user inputs for all 13 features
longitude = st.number_input("Longitude", value=-120.0, format="%.6f")
latitude = st.number_input("Latitude", value=37.0, format="%.6f")
housing_median_age = st.number_input("Housing Median Age", value=25)
total_rooms = st.number_input("Total Rooms", value=2000)
total_bedrooms = st.number_input("Total Bedrooms", value=400)
population = st.number_input("Population", value=1000)
households = st.number_input("Households", value=500)
median_income = st.number_input("Median Income", value=3.5, format="%.2f")

# Categorical variable 'ocean_proximity' dummy variables
st.subheader("Ocean Proximity 🌊")
ocean_proximity_choices = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
ocean_proximity = st.selectbox("Select Ocean Proximity", ocean_proximity_choices)

# Create dummy variables manually
ocean_features = {
    "<1H OCEAN": 0,
    "INLAND": 0,
    "ISLAND": 0,
    "NEAR BAY": 0,
    "NEAR OCEAN": 0
}

# Set the selected ocean_proximity feature to 1
ocean_features[ocean_proximity] = 1

# When user clicks "Predict"
if st.button("Predict"):
    # Combine all features into a single array (in the correct order)
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms,
                            total_bedrooms, population, households, median_income,
                            ocean_features["<1H OCEAN"], ocean_features["INLAND"],
                            ocean_features["ISLAND"], ocean_features["NEAR BAY"],
                            ocean_features["NEAR OCEAN"]]])

    # Predict
    prediction = model.predict(input_data)

    st.success(f"🏡 Predicted Median House Value: **${prediction[0]:,.2f}**")
