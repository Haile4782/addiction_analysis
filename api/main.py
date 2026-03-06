from fastapi import FastAPI, Depends
import joblib
import mlflow
from api.schemas import PredictionInput
from api.auth import verify_api_key
from api.logger import logger
from src.preprocessing_utils import prepare_input_data

app = FastAPI(title="Addiction Risk ML API")

model = joblib.load("models/random_forest_model.pkl")


@app.post("/predict")
def predict(data: PredictionInput, api_key: str = Depends(verify_api_key)):

    input_dict = data.dict()
    processed = prepare_input_data(input_dict)

    prediction = model.predict(processed)[0]
    probability = max(model.predict_proba(processed)[0])

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_params(input_dict)
        mlflow.log_metric("confidence", float(probability))

    logger.info(f"Prediction: {prediction}, Confidence: {probability}")

    return {
        "prediction": prediction,
        "confidence": float(probability)
    }