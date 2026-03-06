import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from src.preprocessing_utils import prepare_input_data

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Addiction Risk Intelligence System",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# LOAD MODEL (CACHED)
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/random_forest_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_artifacts()

# -------------------------
# HEADER
# -------------------------
st.title("🧠 Behavioral Addiction Risk Intelligence System")

st.markdown("""
AI-powered behavioral risk prediction using a tuned Random Forest model.  
Provides real-time prediction, confidence scoring, and explainability.
""")

st.divider()

# -------------------------
# INPUT SECTION
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics")

    age = st.slider("Age", 18, 80, 30)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    annual_income_usd = st.number_input(
        "Annual Income (USD)",
        0,
        200000,
        30000
    )

    bmi = st.slider(
        "BMI",
        15.0,
        40.0,
        23.0
    )

with col2:
    st.subheader("🏥 Lifestyle & Health")

    smokes_per_day = st.slider(
        "Cigarettes per Day",
        0,
        40,
        5
    )

    drinks_per_week = st.slider(
        "Drinks per Week",
        0,
        40,
        3
    )

    mental_health_status = st.selectbox(
        "Mental Health Status",
        ["Good", "Average", "Poor"]
    )

    exercise_frequency = st.selectbox(
        "Exercise Frequency",
        ["Low", "Medium", "High"]
    )

    sleep_hours = st.slider(
        "Sleep Hours per Night",
        3,
        12,
        7
    )

st.divider()

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔍 Analyze Risk"):

    try:

        input_data = {
            "age": age,
            "gender": gender,
            "annual_income_usd": annual_income_usd,
            "bmi": bmi,
            "smokes_per_day": smokes_per_day,
            "drinks_per_week": drinks_per_week,
            "mental_health_status": mental_health_status,
            "exercise_frequency": exercise_frequency,
            "sleep_hours": sleep_hours
        }

        # ----------------------------------
        # PREPROCESS INPUT
        # ----------------------------------
        processed_data = prepare_input_data(input_data)

        # Align with training features
        processed_data = processed_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # ----------------------------------
        # PREDICTION
        # ----------------------------------
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]

        st.subheader("📊 Risk Classification")

        if prediction == "Low":
            st.success(f"Predicted Risk Level: {prediction}")

        elif prediction == "Medium":
            st.warning(f"Predicted Risk Level: {prediction}")

        else:
            st.error(f"Predicted Risk Level: {prediction}")

        # ----------------------------------
        # CONFIDENCE GAUGE
        # ----------------------------------
        confidence = max(probabilities)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Model Confidence (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------
        # PROBABILITY BREAKDOWN
        # ----------------------------------
        st.subheader("📈 Class Probabilities")

        prob_df = pd.DataFrame({
            "Risk Level": model.classes_,
            "Probability": probabilities
        })

        st.bar_chart(
            prob_df.set_index("Risk Level")
        )

        # ----------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------
        st.subheader("🔎 Feature Contribution Analysis")

        try:

            feature_importance = model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": feature_importance
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            st.dataframe(
                importance_df,
                use_container_width=True
            )

            # Plotly importance chart
            fig2 = go.Figure()

            fig2.add_bar(
                x=importance_df["Importance"],
                y=importance_df["Feature"],
                orientation="h"
            )

            fig2.update_layout(
                title="Feature Importance for Model Decision",
                xaxis_title="Importance Score",
                yaxis_title="Feature"
            )

            st.plotly_chart(fig2, use_container_width=True)

        except Exception as explain_error:

            st.warning(
                f"Explainability temporarily unavailable: {explain_error}"
            )

    except Exception as e:

        st.error(f"Error during prediction: {e}")

st.divider()

st.markdown("""
---
**Model:** Random Forest (GridSearchCV tuned)  
**Explainability:** Model Feature Importance  
**Architecture:** Modular ML Pipeline  
**Author:** Haiyleyesus Abayneh  
""")