import pandas as pd

def encode_features(df):
    """
    Prepare features for modeling:
    - Drop non-feature columns
    - One-hot encode categorical variables
    """
    # Target and score columns to drop from X
    drop_cols = ["addiction_score", "addiction_risk"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["addiction_risk"] if "addiction_risk" in df.columns else None

    # One-hot encoding (drop_first to avoid multicollinearity)
    X = pd.get_dummies(X, drop_first=True)

    print(f"Encoded features shape: {X.shape}")
    return X, y