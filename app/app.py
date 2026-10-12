"""
app.py - Streamlit Web Application for ChargeTime AI.
Predicts Electric Vehicle Charging Session Duration using Machine Learning (Linear Regression).
Includes interactive predictor, EDA plot gallery, data viewer, and Google Colab code snippet section.
"""

import pathlib
import sys
import json
import pandas as pd
import streamlit as st

# Add src to sys.path so modules can be imported smoothly
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_charging_duration, load_prediction_artifacts, format_hours_to_hours_minutes
from data_preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, load_raw_data, clean_data

# Page configuration
st.set_page_config(
    page_title="ChargeTime AI - EV Session Duration Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .result-box {
        background-color: #F0FDF4;
        border: 2px solid #22C55E;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .result-title {
        font-size: 1.1rem;
        color: #15803D;
        font-weight: 600;
    }
    .result-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #166534;
    }
    .result-sub {
        font-size: 1.3rem;
        font-weight: 600;
        color: #15803D;
    }
    .code-box {
        background-color: #1E1E1E;
        color: #D4D4D4;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cached_artifacts():
    """Loads preprocessor and model into Streamlit cache."""
    try:
        model, preprocessor = load_prediction_artifacts()
        return model, preprocessor, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def get_cached_raw_data():
    """Loads and caches cleaned dataset."""
    try:
        df = load_raw_data()
        df_clean = clean_data(df)
        return df_clean
    except Exception:
        return None

def main():
    st.markdown('<div class="main-header">⚡ ChargeTime AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predicting Electric Vehicle Charging Session Duration Using Machine Learning (Linear Regression)</div>', unsafe_allow_html=True)

    model, preprocessor, err_msg = get_cached_artifacts()
    df_data = get_cached_raw_data()
    
    if err_msg:
        st.error(f"⚠️ Model files not loaded: {err_msg}")
        st.info("Please run the training pipeline first using `python src/train_model.py`.")
        return

    # Sidebar for Project Details & Navigation
    with st.sidebar:
        st.header("📌 Project Details")
        st.write("**Course:** Foundations of AIML (B.Tech CSE)")
        st.write("**Primary Model:** Linear Regression")
        st.write("**Target Variable:** `Charging Duration (hours)`")
        st.divider()
        st.write("**Author:** Sarthak Kharche")
        st.write("**Dataset:** EV Charging Patterns (~1,320 records)")
        st.markdown("[[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)](https://colab.research.google.com/)")

    # Main Tabs (Predictor, Analytics & Plots, Colab / Notebook Playground)
    tab1, tab2, tab3 = st.tabs([
        "⚡ Session Duration Predictor", 
        "📊 Data Analytics & Plots Gallery", 
        "📓 Google Colab / Code Playground"
    ])

    # TAB 1: PREDICTOR
    with tab1:
        st.subheader("📋 Enter Session Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🚗 Vehicle Information")
            vehicle_model = st.selectbox(
                "Vehicle Model",
                options=["BMW i3", "Chevy Bolt", "Hyundai Kona", "Nissan Leaf", "Tesla Model 3"],
                index=4
            )
            vehicle_age = st.number_input(
                "Vehicle Age (years)",
                min_value=0.0, max_value=20.0, value=3.0, step=0.5
            )
            battery_capacity = st.number_input(
                "Battery Capacity (kWh)",
                min_value=10.0, max_value=200.0, value=75.0, step=1.0
            )

        with col2:
            st.markdown("### 🔌 Charging Information")
            energy_consumed = st.number_input(
                "Energy Consumed (kWh)",
                min_value=0.1, max_value=200.0, value=42.0, step=0.5
            )
            charging_rate = st.number_input(
                "Charging Rate (kW)",
                min_value=1.0, max_value=350.0, value=11.0, step=0.5
            )
            charger_type = st.selectbox(
                "Charger Type",
                options=["Level 1", "Level 2", "DC Fast Charger"],
                index=1
            )
            soc_start = st.slider("State of Charge Start (%)", min_value=0, max_value=100, value=25)
            soc_end = st.slider("State of Charge End (%)", min_value=0, max_value=100, value=80)

        with col3:
            st.markdown("### 🌐 Usage & Environment")
            distance_driven = st.number_input(
                "Distance Driven Since Last Charge (km)",
                min_value=0.0, max_value=1000.0, value=120.0, step=5.0
            )
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-30.0, max_value=60.0, value=27.0, step=1.0
            )
            time_of_day = st.selectbox(
                "Time of Day",
                options=["Morning", "Afternoon", "Evening", "Night"],
                index=1
            )
            day_of_week = st.selectbox(
                "Day of Week",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=0
            )
            user_type = st.selectbox(
                "User Type",
                options=["Commuter", "Casual Driver", "Long-Distance Traveler"],
                index=0
            )

        st.divider()

        # Input Validation
        is_valid = True
        if soc_start >= soc_end:
            st.warning("⚠️ **Input Validation Warning:** State of Charge Start (%) should be strictly less than State of Charge End (%).")
            is_valid = False
            
        if energy_consumed > battery_capacity:
            st.info("💡 **Notice:** Energy Consumed exceeds Battery Capacity. Verify if multiple charging passes occurred.")

        predict_btn = st.button("⚡ Predict Charging Duration", type="primary", use_container_width=True)

        if predict_btn and is_valid:
            input_data = {
                "Battery Capacity (kWh)": float(battery_capacity),
                "Energy Consumed (kWh)": float(energy_consumed),
                "Charging Rate (kW)": float(charging_rate),
                "State of Charge (Start %)": float(soc_start),
                "State of Charge (End %)": float(soc_end),
                "Distance Driven (since last charge) (km)": float(distance_driven),
                "Temperature (°C)": float(temperature),
                "Vehicle Age (years)": float(vehicle_age),
                "Charger Type": charger_type,
                "Vehicle Model": vehicle_model,
                "Time of Day": time_of_day,
                "Day of Week": day_of_week,
                "User Type": user_type
            }

            with st.spinner("Applying preprocessing pipeline & predicting duration..."):
                res = predict_charging_duration(input_data, model=model, preprocessor=preprocessor)

            # Display Result Card
            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">ESTIMATED CHARGING SESSION DURATION</div>
                <div class="result-value">{res['predicted_duration_hours']:.2f} hours</div>
                <div class="result-sub">≈ {res['formatted_time']}</div>
            </div>
            """, unsafe_allow_html=True)

            if res["is_clipped"]:
                st.info("ℹ️ The mathematical Linear Regression combination produced a non-positive value for extreme parameters. The output was safely validated and set to minimum session duration.")

        # Expandable Model Information Section
        st.divider()
        with st.expander("ℹ️ About the Machine Learning Model"):
            st.markdown("""
            ### Model Architecture & Evaluation
            * **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
            * **Problem Type:** Supervised Regression
            * **Dataset:** EV Charging Patterns Dataset (~1,320 records)
            * **Target Variable:** `Charging Duration (hours)`
            """)
            
            metrics_file = BASE_DIR / "outputs" / "metrics" / "metrics_summary.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                    
                lr_m = metrics_data.get("linear_regression", {})
                bm_m = metrics_data.get("mean_baseline", {})

                st.subheader("📊 Evaluation Metrics (Test Set)")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("MAE (Mean Absolute Error)", f"{lr_m.get('MAE', 0):.4f} hrs", f"Baseline: {bm_m.get('MAE', 0):.4f}")
                m_col2.metric("MSE (Mean Squared Error)", f"{lr_m.get('MSE', 0):.4f}", f"Baseline: {bm_m.get('MSE', 0):.4f}")
                m_col3.metric("RMSE (Root MSE)", f"{lr_m.get('RMSE', 0):.4f} hrs", f"Baseline: {bm_m.get('RMSE', 0):.4f}")
                m_col4.metric("R² Score", f"{lr_m.get('R2', 0):.4f}", f"Baseline: {bm_m.get('R2', 0):.4f}")

            coef_file = BASE_DIR / "outputs" / "metrics" / "regression_coefficients.csv"
            if coef_file.exists():
                st.subheader("📈 Learned Regression Coefficients")
                st.caption("Positive coefficient = associated increase in predicted duration, holding other features constant. Negative = associated decrease.")
                coef_df = pd.read_csv(coef_file)
                st.dataframe(coef_df, use_container_width=True)

    # TAB 2: ANALYTICS & PLOTS GALLERY
    with tab2:
        st.subheader("📊 Exploratory Data Analysis & Diagnostic Plots")
        
        if df_data is not None:
            st.markdown("### 📄 Raw Dataset Sample (~1,320 Sessions)")
            st.dataframe(df_data.head(15), use_container_width=True)
            
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Dataset Shape:** " + str(df_data.shape))
                st.markdown("**Total Sessions:** " + str(len(df_data)))
            with p2:
                st.markdown("**Missing Values:** " + str(df_data.isnull().sum().sum()))
                st.markdown("**Duplicates:** " + str(df_data.duplicated().sum()))
                
        st.divider()
        st.markdown("### 🖼️ Saved Visualizations")
        
        plots_dir = BASE_DIR / "outputs" / "plots"
        if plots_dir.exists():
            img_col1, img_col2 = st.columns(2)
            
            actual_vs_pred = plots_dir / "actual_vs_predicted.png"
            residual_p = plots_dir / "residual_plot.png"
            heatmap_p = plots_dir / "eda_correlation_heatmap.png"
            dist_p = plots_dir / "eda_duration_distribution.png"
            
            with img_col1:
                if actual_vs_pred.exists():
                    st.image(str(actual_vs_pred), caption="Actual vs. Predicted Charging Duration", use_container_width=True)
                if heatmap_p.exists():
                    st.image(str(heatmap_p), caption="Correlation Heatmap of Numerical Variables", use_container_width=True)

            with img_col2:
                if residual_p.exists():
                    st.image(str(residual_p), caption="Residual Error Analysis", use_container_width=True)
                if dist_p.exists():
                    st.image(str(dist_p), caption="Target Variable Distribution", use_container_width=True)

    # TAB 3: GOOGLE COLAB & NOTEBOOK CODE PLAYGROUND
    with tab3:
        st.subheader("📓 Google Colab & Jupyter Notebook Code Playground")
        st.markdown("""
        You can execute the complete **ChargeTime AI ML Pipeline** in **Google Colab** or locally in Jupyter Notebook.
        
        ### 🚀 How to Run in Google Colab:
        1. Open Google Colab at [colab.research.google.com](https://colab.research.google.com/).
        2. Create a New Notebook or upload `notebooks/ChargeTime_AI_Analysis.ipynb`.
        3. Run the Colab setup cell below to install dependencies and execute the model pipeline.
        """)
        
        st.markdown("### 📋 Google Colab Self-Contained Execution Script")
        st.code('''# ==========================================
# CHARGETIME AI - GOOGLE COLAB EXECUTION SCRIPT
# ==========================================

# Step 1: Install required packages
!pip install -q pandas numpy scikit-learn matplotlib seaborn joblib

# Step 2: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 3: Load dataset (Replace URL or upload ev_charging_patterns.csv)
url = "ev_charging_patterns.csv" # or local path in Colab
df = pd.read_csv(url)

# Step 4: Define Features and Target
num_cols = ["Battery Capacity (kWh)", "Energy Consumed (kWh)", "Charging Rate (kW)", 
            "State of Charge (Start %)", "State of Charge (End %)", 
            "Distance Driven (since last charge) (km)", "Temperature (°C)", "Vehicle Age (years)"]
cat_cols = ["Vehicle Model", "Charger Type", "Time of Day", "Day of Week", "User Type"]
target_col = "Charging Duration (hours)"

X = df[num_cols + cat_cols]
y = df[target_col]

# Step 5: Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_train_tr = preprocessor.fit_transform(X_train)
X_test_tr = preprocessor.transform(X_test)

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_tr, y_train)

# Step 8: Evaluate Predictions
y_pred = model.predict(X_test_tr)

print("--- MODEL EVALUATION RESULTS ---")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f} hours")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} hours")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
''', language="python")

        st.markdown("### 📄 Notebook Location in Repository")
        st.info("The full 22-section Jupyter Notebook with pre-rendered plots is available at: `notebooks/ChargeTime_AI_Analysis.ipynb`")

if __name__ == "__main__":
    main()
