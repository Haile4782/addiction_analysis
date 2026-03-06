from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
import plotly.express as px


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return y_pred


def plot_confusion_matrix(y_test, y_pred, save_path):
    """
    Plot interactive confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        x=["Low", "Medium", "High"],
        y=["Low", "Medium", "High"],
        title="Confusion Matrix - Addiction Risk Model"
    )

    fig.write_html(save_path)
    fig.show()


def plot_feature_importance(model, X, save_path):
    """
    Plot top feature importance.
    """
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    fig = px.bar(
        importance_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Important Features"
    )

    fig.write_html(save_path)
    fig.show()