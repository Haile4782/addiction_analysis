# 📊 Cigarettes & Alcohol Addiction Analysis

## 🧭 Project Overview

This capstone project explores behavioral, demographic, and lifestyle patterns associated with cigarette and alcohol addiction using a synthetic population dataset.

The goal is to uncover insights that help understand relationships between **health, lifestyle, and addiction behaviors** through data cleaning, exploratory data analysis (EDA), and visualization.


## 🎯 Objectives

* Clean and validate a real-world style dataset
* Perform exploratory data analysis
* Identify patterns and correlations in addiction behavior
* Create clear visual insights for decision-making


## 📂 Dataset

**Source:** Kaggle - Cigarettes & Alcohol Addiction dataset

The dataset contains information on **3,000 individuals**, including:

* Demographics (age, gender, country)
* Lifestyle indicators (exercise, diet, sleep)
* Behavioral metrics (smoking, drinking)
* Health indicators (BMI, mental health)


## 🛠️ Project Workflow

### 1️⃣ Data Cleaning

Steps performed:

* Removed duplicate records
* Handled missing values
* Capped outliers using IQR method
* Applied logical validation checks
* Standardized dataset structure

After cleaning:

* Original records: **3000**
* Final records: **2295**

### 2️⃣ Exploratory Data Analysis (EDA)

Key analyses include:

* Addiction distribution
* Addiction score by gender
* Correlation heatmap
* Age vs addiction levels
* Income vs addiction score
* Mental health vs addiction score
* Sleep hours vs addiction score


### 3️⃣ Visualizations

The project includes **7 key visualizations** highlighting:

* Behavioral trends
* Demographic comparisons
* Correlation insights

Final charts are stored in:

visuals/

## 📁 Project Structure

capstone_project/
│
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
│
├── src/
│   └── data_cleaning.py
│
└── README.md

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib and Seaborn
* VS Code

## 🚀 How to Run

### 1️⃣ Clone the repository
git clone <your-repo-link>
cd capstone_project

### 2️⃣ Run data cleaning pipeline
python src/data_cleaning.py

### 3️⃣ Open EDA notebook
notebooks/eda1.ipynb - current(uncleaned) dataset  
notebooks/eda.ipynb  - cleaned dataset

## 📈 Key Insights (Example - update after final EDA)

* Higher smoking frequency correlates with poorer mental health
* Lower exercise frequency associates with higher BMI
* Income shows a moderate relationship with drinking patterns
* Strong social support relates to healthier lifestyle indicators

## 📚 Learning Outcomes

Through this project, I developed skills in:

* Data preprocessing & validation
* Exploratory analysis
* Data storytelling
* Reproducible workflows
* Project structuring for analytics

## 🤝 Team

Capstone project completed by a team of four data analysts.

## 📌 Future Improvements

* Build predictive models
* Perform clustering analysis
* Create interactive dashboard (Power BI / Tableau)

## 📜 License

This project is for educational and research purposes.

Author by **Haiyleyesus Abayneh**  
[GitHub](https://github.com/Haile4782) | [Portfolio](https://www.datascienceportfol.io/haiyleyesusAB)