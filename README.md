<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Model-blue?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=flat" alt="Status">
</p>

<h1 align="center">Addiction Risk Predictor</h1>
<h3 align="center">Machine Learning System for Behavioral Addiction Risk Assessment</h3>

<p align="center">
  A complete end-to-end machine learning project that predicts high/low addiction risk using demographic and lifestyle data. 
  Includes full EDA, modular pipeline, trained model, and a live interactive Streamlit web app.
</p>

---

## ✨ Project Highlights

- **99.56% Accuracy** on test set using random forest classifier
- Interactive **Streamlit dashboard** for real-time risk prediction
- Complete modular pipeline (data cleaning → feature engineering → training → evaluation)
- Clean, reproducible code structure with `src/` architecture
- Live demo available: https://addictionanalysis-haiyleyesus.streamlit.app/

---

## 🛠️ Tech Stack

| Category             | Technology                          |
|----------------------|-------------------------------------|
| Language             | Python 3.13                         |
| Data Processing      | Pandas, NumPy                       |
| Modeling             | Scikit-learn (Random Forest)        |
| Visualization        | Plotly, Matplotlib, Seaborn         |
| Dashboard            | Streamlit                           |
| Environment          | Virtualenv + requirements.txt       |

---

## 📂 Project Structure

```
addiction_analysis/
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml
├── api/
│   ├── auth.py
│   ├── logger.py
│   ├── main.py
│   ├── schemas.py
│   └── config.toml
├── data/
│   ├── raw/
│   └── cleaned/
├── models/
│   ├── random_forest_model.pkl
│   └── feature_columns.pkl
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── evaluation.py
│   ├── explain.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── pipeline.py
│   └── preprocessing_utils.py
├── visuals/
│   ├── eda/
│   └── model/
├── .gitignore
├── app.py                   # Streamlit Dashboard
├── docker-compose.yml
├── Dockerfile                  
├── main.py                  # Run full pipeline
├── mlflow.db                   
├── README.md
├── requirements.txt
└── runtime.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository
```
git clone https://github.com/Haile4782/addiction_analysis.git
cd addiction_analysis
```

### 2. Create & activate environment
```
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the full training pipeline
```
python main.py
```

### 5. Launch the Streamlit Dashboard
```
streamlit run app.py
```

---

## 📈 Key Insights

- Smoking frequency and alcohol consumption are the strongest predictors of high addiction risk.
- Poor mental health and low sleep hours show strong correlation with higher addiction scores.
- Strong social support appears to be a protective factor against addiction.
- The model performs exceptionally well on the majority class (Low Risk) with minor challenges on the minority class (High Risk) due to class imbalance.

---

## 🎯 Live Demo

**Try the live prediction app here:**  
https://addictionanalysis-haiyleyesus.streamlit.app/

---

## 📌 Future Improvements

- Add SHAP explainability for individual predictions
- Implement class balancing techniques (SMOTE)
- Deploy as FastAPI backend + React frontend
- Add user authentication and prediction history
- Experiment with XGBoost / LightGBM

---

## 📜 License

https://www.mit.edu/~amini/LICENSE.md

---

Author by **Haiyleyesus Abayneh**  
[GitHub](https://github.com/Haile4782) | [Portfolio](https://www.datascienceportfol.io/haiyleyesusAB)