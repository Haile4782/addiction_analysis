import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


def prepare_input_data(input_dict):

    df = pd.DataFrame([input_dict])

    # Ensure numeric columns are correct
    numeric_cols = [
        "age", "annual_income_usd",
        "bmi", "smokes_per_day", "drinks_per_week",
        "sleep_hours"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # Normalize smoking & drinking
    scaler = MinMaxScaler()
    df[["smokes_scaled", "drinks_scaled"]] = scaler.fit_transform(
        df[["smokes_per_day", "drinks_per_week"]]
    )

    df["addiction_score"] = (
        0.5 * df["smokes_scaled"] +
        0.5 * df["drinks_scaled"]
    )

    df = df.drop(columns=["addiction_score"])

    df = pd.get_dummies(df, drop_first=True)

    training_columns = joblib.load("models/feature_columns.pkl")

    df = df.reindex(columns=training_columns, fill_value=0)

    return df