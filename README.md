# ⚡ ChargeTime AI: Predicting Electric Vehicle Charging Session Duration Using Machine Learning

**Academic Mini Project** for *Foundations of AIML* (B.Tech CSE 3rd Year)  
**Author:** Sarthak Kharche  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📌 Project Overview
**ChargeTime AI** is a machine learning solution designed to predict the charging session duration of Electric Vehicles (EVs) in hours. Using charging session characteristics such as battery capacity, energy consumed, charging rate, state of charge (Start & End %), distance driven, temperature, vehicle age, charger type, vehicle model, time of day, day of week, and user type, the system estimates the required session duration before or during charging.

The project is built specifically using **Linear Regression** to demonstrate fundamental supervised learning principles, feature engineering, pipeline preprocessing, model evaluation, and deployment via an interactive Streamlit web application.

---

## 🎯 Problem Statement
With the rapid adoption of Electric Vehicles (EVs), efficient charging infrastructure utilization and accurate schedule planning require reliable estimates of charging session durations. Session durations vary significantly based on vehicle specifications, battery health, charging power, and ambient environmental conditions. ChargeTime AI applies supervised regression modeling to historical charging session data to solve this forecasting problem.

---

## 🎯 Objectives
1. **Data Cleaning & Inspection**: Analyze the raw Kaggle dataset, verify schema, data types, nulls, and duplicates.
2. **Exploratory Data Analysis (EDA)**: Identify key drivers of session duration and visualize feature interactions.
3. **Target Leakage Prevention**: Audit features and exclude post-session metadata (`Charging End Time`, `Charging Cost`) to prevent leakage.
4. **Pipeline Preprocessing**: Apply `StandardScaler` to numerical variables and `OneHotEncoder` to categorical variables using scikit-learn `ColumnTransformer`.
5. **Supervised Regression Modeling**: Train and evaluate a **Linear Regression** model (`sklearn.linear_model.LinearRegression`).
6. **Model Diagnostics & Interpretability**: Evaluate MAE, MSE, RMSE, and $R^2$ score against a Mean Baseline, inspect residual plots, and extract learned regression coefficients.
7. **Interactive Web Application**: Provide an intuitive Streamlit interface for real-time predictions.

---

## 📊 Dataset Information
- **Dataset Name:** Electric Vehicle Charging Patterns Dataset
- **Source:** Kaggle ([valakhorasani/electric-vehicle-charging-patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns))
- **Records:** ~1,320 EV charging sessions
- **Target Variable:** `Charging Duration (hours)`

---

## 🛠️ Machine Learning Workflow

```
Dataset (CSV)
   │
   ▼
Data Cleaning & Target Leakage Audit
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Feature Selection (X = 13 Features, y = Charging Duration)
   │
   ▼
80/20 Train-Test Split (random_state=42)
   │
   ▼
Pipeline Preprocessing (StandardScaler + OneHotEncoder)
   │
   ▼
Linear Regression Model Training & Baseline Training
   │
   ▼
Model Evaluation (MAE, MSE, RMSE, R²) & Diagnostics
   │
   ▼
Model & Preprocessor Serialization (.pkl)
   │
   ▼
Streamlit Web Application UI (app/app.py)
```

---

## 🔑 Features

### Input Features ($X$)
- **Numerical Features (8):**
  - `Battery Capacity (kWh)`: Battery capacity of the EV
  - `Energy Consumed (kWh)`: Total energy transferred during session
  - `Charging Rate (kW)`: Power delivery rate of the charging station
  - `State of Charge (Start %)`: Initial battery charge percentage
  - `State of Charge (End %)`: Final target battery charge percentage
  - `Distance Driven (since last charge) (km)`: Usage intensity indicator
  - `Temperature (°C)`: Ambient environment temperature
  - `Vehicle Age (years)`: Age of the vehicle in years
- **Categorical Features (5):**
  - `Vehicle Model`: Model of the EV (Tesla Model 3, Nissan Leaf, Chevy Bolt, BMW i3, Hyundai Kona)
  - `Charger Type`: Level 1, Level 2, DC Fast Charger
  - `Time of Day`: Morning, Afternoon, Evening, Night
  - `Day of Week`: Monday through Sunday
  - `User Type`: Commuter, Casual Driver, Long-Distance Traveler

### Target Feature ($y$)
- `Charging Duration (hours)` (Continuous Numerical Target)

---

## 💻 Technology Stack
- **Programming Language:** Python 3.10+
- **Data Analysis & Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Serialization:** Joblib
- **Web Interface:** Streamlit
- **Notebook Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
ChargeTime-AI/
├── data/
│   ├── raw/
│   │   └── ev_charging_patterns.csv
│   └── processed/
│       └── ev_charging_processed.csv
├── notebooks/
│   └── ChargeTime_AI_Analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/
│   ├── linear_regression_model.pkl
│   ├── baseline_model.pkl
│   └── preprocessor.pkl
├── app/
│   └── app.py
├── outputs/
│   ├── plots/
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_plot.png
│   │   └── eda_*.png
│   └── metrics/
│       ├── metrics_summary.json
│       └── regression_coefficients.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Installation
Clone or download the repository, navigate to the root directory, and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Running Data Preprocessing & Training
To execute the data preprocessing pipeline and train the Linear Regression model:

```bash
python src/train_model.py
```

### 4. Running Model Evaluation & Generating Plots
To calculate evaluation metrics, generate EDA figures, residual plots, and coefficient tables:

```bash
python src/evaluate_model.py
```

### 5. Running the Streamlit Web Application
Launch the interactive web application using Python's module launcher (recommended for Windows):

```bash
py -m streamlit run app/app.py
```
*or*
```bash
python -m streamlit run app/app.py
```

*(If `streamlit` is added to your PATH, `streamlit run app/app.py` also works).*

---

## 📈 Model Performance & Results

*(Evaluated on 20% unseen test set using `random_state=42`)*

| Metric | Linear Regression Model | Mean Baseline Model |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | **0.8722 hours** | 0.8579 hours |
| **Mean Squared Error (MSE)** | **1.2072** | 1.1565 |
| **Root Mean Squared Error (RMSE)** | **1.0987 hours** | 1.0754 hours |
| **R² Score** | **-0.0439** | 0.0000 |

---

## 🚀 Future Scope
1. **Algorithm Benchmarking**: Benchmark Linear Regression against Decision Trees, Random Forests, and XGBoost.
2. **Real-time Charging API**: Integrate real-time charging station load data and grid constraints.
3. **Weather API Integration**: Fetch dynamic ambient temperature and humidity via open weather APIs.
4. **Mobile Application**: Expose model via REST API (FastAPI) for mobile user integration.

---

## 👤 Author
**Sarthak Kharche**  
B.Tech Computer Science & Engineering (3rd Year)  
Foundations of AIML Mini Project  
