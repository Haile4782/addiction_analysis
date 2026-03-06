from src.pipeline import run_pipeline


if __name__ == "__main__":

    DATA_PATH = "data/cleaned/addiction_population_clean.csv"
    MODEL_PATH = "models/random_forest_model.pkl"

    run_pipeline(DATA_PATH, MODEL_PATH)