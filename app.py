import streamlit as st
import joblib
from src.preprocessing_utils import prepare_input_data

# Load model
model = joblib.load("models/random_forest_model.pkl")

st.set_page_config(page_title="Addiction Risk Predictor")

st.title("Behavioral Addiction Risk Prediction App")

st.markdown("Predict addiction risk level based on lifestyle and behavioral inputs.")

# -------- INPUTS --------

age = st.slider("Age", 18, 80, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

annual_income_usd = st.number_input("Annual Income (USD)", 0, 200000, 30000)

smokes_per_day = st.slider("Cigarettes per Day", 0, 40, 5)

drinks_per_week = st.slider("Drinks per Week", 0, 40, 3)

mental_health_status = st.selectbox(
    "Mental Health Status",
    ["Good", "Average", "Poor"]
)

exercise_frequency = st.selectbox(
    "Exercise Frequency",
    ["Low", "Medium", "High"]
)

sleep_hours = st.slider("Sleep Hours per Night", 3, 12, 7)

bmi = st.slider("BMI", 15.0, 40.0, 23.0)

# -------- PREDICTION --------

if st.button("Predict Risk"):

    input_data = {
        "age": age,
        "gender": gender,
        "annual_income_usd": annual_income_usd,
        "smokes_per_day": smokes_per_day,
        "drinks_per_week": drinks_per_week,
        "mental_health_status": mental_health_status,
        "exercise_frequency": exercise_frequency,
        "sleep_hours": sleep_hours,
        "bmi": bmi
    }

    processed_data = prepare_input_data(input_data)

    prediction = model.predict(processed_data)[0]

    st.success(f"Predicted Addiction Risk Level: {prediction}")