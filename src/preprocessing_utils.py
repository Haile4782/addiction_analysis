import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler


def prepare_input_data(input_dict):

    df = pd.DataFrame([input_dict])

    # Normalize smoking & drinking (same logic as training)
    scaler = MinMaxScaler()

    df[["smokes_scaled", "drinks_scaled"]] = scaler.fit_transform(
        df[["smokes_per_day", "drinks_per_week"]]
    )

    df["addiction_score"] = (
        0.5 * df["smokes_scaled"] +
        0.5 * df["drinks_scaled"]
    )

    # Drop addiction_score (not used for prediction)
    df = df.drop(columns=["addiction_score"])

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Load training columns
    training_columns = joblib.load("models/feature_columns.pkl")

    # Align columns
    df = df.reindex(columns=training_columns, fill_value=0)

    return df