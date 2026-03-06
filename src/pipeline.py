
import joblib
from sklearn.model_selection import train_test_split
from src.data_cleaning import (
    load_data,
    create_addiction_score,
    create_target_variable
)
from src.feature_engineering import encode_features
from src.model_training import train_model, save_model
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance
)


def run_pipeline(data_path: str, model_path: str):

    # Load
    df = load_data(data_path)
     
    # Create addiction score
    df = create_addiction_score(df)
    # Create target
    df = create_target_variable(df)

    # Encode
    X, y = encode_features(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = evaluate_model(model, X_test, y_test)

    # Visuals
    plot_confusion_matrix(
        y_test,
        y_pred,
        "visuals/model/confusion_matrix.html"
    )

    plot_feature_importance(
        model,
        X,
        "visuals/model/feature_importance.html"
    )

    # Save model
    save_model(model, model_path)

    print("Pipeline completed successfully.")