import shap
import pandas as pd

def explain_prediction(model, X_sample):

    try:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="interventional"
        )

        shap_values = explainer.shap_values(
            X_sample,
            check_additivity=False
        )

        return shap_values

    except Exception as e:
        return str(e)