import streamlit as st
import pandas as pd
import joblib
import sys, os

# Set up paths
proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_root, "src"))

from src.feature_engineering import encode_features
from src.data_cleaning import create_target_variable

# Load model
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(proj_root, "models", "random_forest_model.pkl"))

model = load_model()

st.title("Addiction Risk Predictor")
st.write("Enter lifestyle data to predict addiction risk level.")

# Input form
age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
smokes_per_day = st.number_input("Cigarettes per day", min_value=0, max_value=50, value=0)
drinks_per_week = st.number_input("Drinks per week", min_value=0, max_value=50, value=0)
exercise_freq = st.selectbox("Exercise frequency", ["Never", "Rarely", "Weekly", "Daily"])
diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
sleep_hours = st.number_input("Sleep hours per night", min_value=0.0, max_value=24.0, value=8.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
mental_health = st.selectbox("Mental health status", ["Poor", "Average", "Good"])
social_support = st.selectbox("Social support", ["None", "Weak", "Moderate", "Strong"])

if st.button("Predict"):
    # Create a sample dataframe
    sample_data = pd.DataFrame({
        'id': [1],
        'name': ['Sample'],
        'age': [age],
        'gender': [gender],
        'country': ['Unknown'],
        'city': ['Unknown'],
        'education_level': ['Unknown'],
        'employment_status': ['Unknown'],
        'annual_income_usd': [50000],  # dummy
        'marital_status': ['Unknown'],
        'children_count': [0],  # dummy
        'smokes_per_day': [smokes_per_day],
        'drinks_per_week': [drinks_per_week],
        'age_started_smoking': [age if smokes_per_day > 0 else 0],
        'age_started_drinking': [age if drinks_per_week > 0 else 0],
        'attempts_to_quit_smoking': [0],
        'attempts_to_quit_drinking': [0],
        'has_health_issues': [False],
        'mental_health_status': [mental_health],
        'exercise_frequency': [exercise_freq],
        'diet_quality': [diet_quality],
        'sleep_hours': [sleep_hours],
        'bmi': [bmi],
        'social_support': [social_support],
        'therapy_history': ['None']
    })

    # Preprocess
    sample_processed = create_target_variable(sample_data)
    X, _ = encode_features(sample_processed)

    # Predict
    prediction = model.predict(X)[0]
    risk_level = "High Risk" if prediction == 1 else "Low Risk"

    st.success(f"Predicted Addiction Risk: {risk_level}")

    # Show score
    score = sample_processed['addiction_score'].iloc[0]
    st.info(f"Addiction Score: {score} (based on smoking + drinking)")

st.write("---")
st.write("This app demonstrates the ML model trained on addiction data.")