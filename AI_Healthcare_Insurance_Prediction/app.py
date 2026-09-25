import streamlit as st
import pandas as pd
import joblib

# Page title
st.title("Healthcare Insurance Cost Prediction")

st.write("Enter customer details to predict medical insurance charges.")

# Load saved files
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/ridge_model.pkl")

# User inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Prediction button
if st.button("Predict Insurance Cost"):

    # Create dataframe using the same columns as training data
    customer_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Preprocess the input
    processed_data = preprocessor.transform(customer_data)

    # Make prediction
    prediction = model.predict(processed_data)

    # Display result
    st.success(
        f"Estimated Insurance Cost: ${prediction[0]:,.2f}"
    )