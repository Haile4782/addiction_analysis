import pandas as pd


def encode_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target y.
    Applies one-hot encoding to categorical features.
    """
    df = df.copy()

    X = df.drop(columns=["addiction_score", "addiction_risk"])
    y = df["addiction_risk"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y