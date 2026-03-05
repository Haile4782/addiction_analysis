# src/pipeline.py

from sklearn.model_selection import train_test_split

# Import from other modules
from src.data_cleaning import load_data, clean_data, create_target_variable, save_data
from src.feature_engineering import encode_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model


def run_pipeline(raw_data_path, clean_data_path=None, model_path=None):
    """
    Full ML pipeline: load → clean → engineer → split → train → evaluate → save
    """
    print("Starting full pipeline...")

    # 1. Load raw data
    df = load_data(raw_data_path)

    # 2. Clean
    df_clean = clean_data(df)

    # 3. Create target (score + binary label)
    df_processed = create_target_variable(df_clean)

    # Optional: save cleaned + target data
    if clean_data_path:
        save_data(df_processed, clean_data_path)

    # 4. Encode features
    X, y = encode_features(df_processed)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 6. Train model
    model = train_model(X_train, y_train)

    # 7. Evaluate
    evaluate_model(model, X_test, y_test)

    # 8. Save model
    if model_path:
        save_model(model, model_path)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    # Example usage when running directly
    raw_path = "data/raw/addiction_population_data.csv"
    clean_path = "data/cleaned/addiction_population_clean.csv"
    model_path = "models/random_forest_model.pkl"

    run_pipeline(raw_path, clean_path, model_path)