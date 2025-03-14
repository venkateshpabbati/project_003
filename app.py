import streamlit as st
import requests

st.title("Linear Regression Prediction")
features = st.text_input("Enter features (comma-separated):")

if st.button("Predict"):
    if features:
        try:
            data = [float(i) for i in features.split(",")]
            response = requests.post("http://127.0.0.1:8000/predict", json={"data": data})
            response.raise_for_status()  # Raise an error for bad status codes
            prediction = response.json().get("prediction")
            if prediction is not None:
                st.success(f"Prediction: {prediction}")
            else:
                st.error("Error: No prediction returned from the server.")
        except ValueError:
            st.error("Error: Please enter valid numbers separated by commas.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: Request failed - {e}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Error: Please enter some features.")