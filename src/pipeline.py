from sklearn.model_selection import train_test_split
from src.data_cleaning import load_data, clean_data, create_target_variable
from src.feature_engineering import encode_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model


def run_pipeline(data_path, model_path):

    df = load_data(data_path)

    # clean before engineering features/targets
    df = clean_data(df)

    df = create_target_variable(df)

    X, y = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, model_path)