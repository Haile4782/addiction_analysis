import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset and clean column names.
    """
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


def create_addiction_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite addiction score based on smoking and drinking behavior.
    """

    df = df.copy()

    required_cols = ["smokes_per_day", "drinks_per_week"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    scaler = MinMaxScaler()

    df[["smokes_scaled", "drinks_scaled"]] = scaler.fit_transform(
        df[["smokes_per_day", "drinks_per_week"]]
    )

    # Composite addiction score (weighted equally)
    df["addiction_score"] = (
        0.5 * df["smokes_scaled"] +
        0.5 * df["drinks_scaled"]
    )

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert addiction score into risk categories.
    """

    df = df.copy()

    df["addiction_risk"] = pd.qcut(
        df["addiction_score"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    return df