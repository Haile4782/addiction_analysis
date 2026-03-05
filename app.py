import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Set up project root and add src to path
proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_root, "src"))

from src.feature_engineering import encode_features
from src.data_cleaning import create_target_variable

# Load model with caching
@st.cache_resource
def load_model():
    model_path = os.path.join(proj_root, "models", "random_forest_model.pkl")
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

st.title("Addiction Risk Predictor")
st.markdown("Enter your lifestyle and demographic details to get a prediction of **addiction risk level** (High / Low).")

with st.form("user_input"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        smokes_per_day = st.number_input("Cigarettes per day", min_value=0, max_value=50, value=0, step=1)
        drinks_per_week = st.number_input("Drinks per week", min_value=0.0, max_value=50.0, value=0.0, step=0.5)

    with col2:
        exercise_freq = st.selectbox("Exercise frequency", ["Never", "Rarely", "Weekly", "Daily"])
        diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
        sleep_hours = st.number_input("Sleep hours per night", min_value=0.0, max_value=24.0, value=8.0, step=0.1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

    mental_health = st.selectbox("Mental health status", ["Poor", "Average", "Good"])
    social_support = st.selectbox("Social support", ["None", "Weak", "Moderate", "Strong"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    with st.spinner("Predicting..."):
        # Exact column order from your cleaned dataset
        columns = [
            'id', 'name', 'age', 'gender', 'country', 'city', 'education_level',
            'employment_status', 'annual_income_usd', 'marital_status', 'children_count',
            'smokes_per_day', 'drinks_per_week', 'age_started_smoking', 'age_started_drinking',
            'attempts_to_quit_smoking', 'attempts_to_quit_drinking', 'has_health_issues',
            'mental_health_status', 'exercise_frequency', 'diet_quality', 'sleep_hours',
            'bmi', 'social_support', 'therapy_history'
        ]

        # Create single-row DataFrame
        input_raw = pd.DataFrame([{
            'id': 1,
            'name': 'Test User',
            'age': age,
            'gender': gender,
            'country': 'Unknown',
            'city': 'Unknown',
            'education_level': 'Unknown',
            'employment_status': 'Unknown',
            'annual_income_usd': 50000,
            'marital_status': 'Unknown',
            'children_count': 0,
            'smokes_per_day': smokes_per_day,
            'drinks_per_week': drinks_per_week,
            'age_started_smoking': age if smokes_per_day > 0 else 0,
            'age_started_drinking': age if drinks_per_week > 0 else 0,
            'attempts_to_quit_smoking': 0,
            'attempts_to_quit_drinking': 0,
            'has_health_issues': False,
            'mental_health_status': mental_health,
            'exercise_frequency': exercise_freq,
            'diet_quality': diet_quality,
            'sleep_hours': sleep_hours,
            'bmi': bmi,
            'social_support': social_support,
            'therapy_history': 'None'
        }], columns=columns)

        # Preprocess exactly as in training
        processed = create_target_variable(input_raw)
        X_encoded, _ = encode_features(processed)

        # Use .values to completely bypass column name validation
        prediction = model.predict(X_encoded.values)[0]
        risk_level = "**High Risk**" if prediction == 1 else "**Low Risk**"

        st.success(f"Predicted Addiction Risk Level: {risk_level}")

        score = processed['addiction_score'].iloc[0]
        st.info(f"**Addiction Score** (smokes per day + drinks per week): **{score:.1f}**")