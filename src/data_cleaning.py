import os
import pandas as pd

def load_data(path):
    """Load dataset from CSV"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


def cap_outliers(df, col):
    """Cap outliers using IQR method"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
    return df


def clean_data(df):
    """Clean dataset: duplicates, missing, outliers, logical checks"""
    df = df.copy()

    print("\n🔹 Initial shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df["social_support"] = df["social_support"].fillna("Unknown")

    # Cap outliers in numeric columns
    outlier_cols = ["smokes_per_day", "drinks_per_week", "sleep_hours", "bmi"]
    for col in outlier_cols:
        if col in df.columns:
            df = cap_outliers(df, col)

    # Logical checks
    df = df[df["age_started_smoking"] <= df["age"]]
    df = df[df["age_started_drinking"] <= df["age"]]

    print("✅ Final shape after cleaning:", df.shape)
    return df


def create_target_variable(df):
    """Create addiction_score and binary risk label"""
    df = df.copy()
    df["addiction_score"] = df["smokes_per_day"] + df["drinks_per_week"]
    df["addiction_risk"] = (df["addiction_score"] > 25).astype(int)
    return df


def save_data(df, path):
    """Save cleaned DataFrame"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved cleaned data to: {path}")


# Standalone run (for testing)
if __name__ == "__main__":
    raw_path = "data/raw/addiction_population_data.csv"
    clean_path = "data/cleaned/addiction_population_clean.csv"

    df = load_data(raw_path)
    df_clean = clean_data(df)
    df_clean = create_target_variable(df_clean)
    save_data(df_clean, clean_path)