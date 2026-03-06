from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib


def train_model(X_train, y_train):
    """
    Train Random Forest with hyperparameter tuning.
    """

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)

    return best_model


def save_model(model, path: str):
    """Save trained model."""
    joblib.dump(model, path)