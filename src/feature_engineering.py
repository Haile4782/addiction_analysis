import pandas as pd

def encode_features(df):
    X = df.drop(columns=["addiction_score", "addiction_risk"])
    y = df["addiction_risk"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y