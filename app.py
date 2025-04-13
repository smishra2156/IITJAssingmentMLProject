import streamlit as st
import joblib
import gdown
import os
import numpy as np

# Title
st.title("Diabetes Prediction App")
st.write("Enter health information to predict the likelihood of diabetes.")

# Download the pipeline if not already downloaded
file_path = "diabetes_pipeline.pkl"
if not os.path.exists(file_path):
    st.info("Downloading model file...")
    url = "https://drive.google.com/file/d/1aE_daexQbhtTrtFBThxK1_ytCvGaKn_i"  # due to size issue uploaded file on my google drive
    gdown.download(url, file_path, quiet=False)

# Load the pipeline
try:
    pipeline = joblib.load(file_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# User input form
with st.form("prediction_form"):
    st.write("### Enter the following inputs:")

    # Example features (adjust based on your dataset)
    glucose = st.number_input("Blood Glucose Level", min_value=0, max_value=500, value=100)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    physical_activity = st.selectbox("Physically Active?", ["Yes", "No"])
    family_history = st.selectbox("Family History of Diabetes?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")

# Map input
if submitted:
    # Convert categorical to numeric if needed
    activity = 1 if physical_activity == "Yes" else 0
    family = 1 if family_history == "Yes" else 0

    input_data = np.array([[glucose, bmi, age, activity, family]])
    
    # Predict
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    st.write(f"### Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    st.write(f"**Probability of Diabetes:** {prob:.2%}")
