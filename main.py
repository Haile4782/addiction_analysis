# main.py
# Entry point to run the full training pipeline

import sys
import os

# Add src folder to Python path
proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(proj_root, "src"))

from src.pipeline import run_pipeline

if __name__ == "__main__":
    # Define paths (relative to project root)
    data_path = os.path.join(proj_root, "data", "cleaned", "addiction_population_clean.csv")
    model_path = os.path.join(proj_root, "models", "random_forest_model.pkl")

    print(f"Running full pipeline...")
    print(f"  Data path: {data_path}")
    print(f"  Model save path: {model_path}")

    # Call the pipeline function with both arguments
    run_pipeline(data_path, model_path)

    print("Pipeline completed successfully!")