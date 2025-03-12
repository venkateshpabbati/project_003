import streamlit as st
import requests

st.title("Linear Regression Prediction")
features = st.text_input("Enter features (comma-separated):")

if st.button("Predict"):
    try:
        data = [float(i) for i in features.split(",")]
        response = requests.post("http://127.0.0.1:8000/predict", json={"data": data})
        prediction = response.json()["prediction"]
        st.success(f"Prediction: {prediction}")
    except Exception as e:
        st.error(f"Error: {e}")