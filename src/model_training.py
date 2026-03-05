# src/model_training.py

import os
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    """
    Train a Random Forest model with balanced class weights
    """
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model


def save_model(model, path):
    """
    Save trained model to disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")