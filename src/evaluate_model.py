"""
evaluate_model.py - Model evaluation, metrics calculation, coefficient analysis, and visualization generator for ChargeTime AI.
"""

import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from data_preprocessing import prepare_and_split_data, get_feature_names, load_raw_data, clean_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"
BASELINE_PATH = BASE_DIR / "models" / "baseline_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

PLOTS_DIR = BASE_DIR / "outputs" / "plots"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

# Styling setup for figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("crest")

def compute_metrics(y_true, y_pred):
    """Computes MAE, MSE, RMSE, and R2 score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def generate_eda_plots(df):
    """Generates and saves exploratory data analysis visualizations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution of Charging Duration
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COL], kde=True, color="#1f77b4", bins=25)
    plt.title("Distribution of Charging Session Duration", fontsize=14, pad=12)
    plt.xlabel("Charging Duration (hours)", fontsize=12)
    plt.ylabel("Session Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_distribution.png", dpi=300)
    plt.close()

    # 2. Charging Duration vs Energy Consumed
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Energy Consumed (kWh)", y=TARGET_COL, alpha=0.7, color="#2ca02c")
    plt.title("Charging Duration vs. Energy Consumed", fontsize=14, pad=12)
    plt.xlabel("Energy Consumed (kWh)", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_vs_energy.png", dpi=300)
    plt.close()

    # 3. Charging Duration vs Charging Rate
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Charging Rate (kW)", y=TARGET_COL, alpha=0.7, color="#ff7f0e")
    plt.title("Charging Duration vs. Charging Rate", fontsize=14, pad=12)
    plt.xlabel("Charging Rate (kW)", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_vs_rate.png", dpi=300)
    plt.close()

    # 4. Charging Duration vs Battery Capacity
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Battery Capacity (kWh)", y=TARGET_COL, alpha=0.7, color="#9467bd")
    plt.title("Charging Duration vs. Battery Capacity", fontsize=14, pad=12)
    plt.xlabel("Battery Capacity (kWh)", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_vs_capacity.png", dpi=300)
    plt.close()

    # 5. Charging Duration vs Starting SOC
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="State of Charge (Start %)", y=TARGET_COL, alpha=0.7, color="#8c564b")
    plt.title("Charging Duration vs. Starting State of Charge (%)", fontsize=14, pad=12)
    plt.xlabel("State of Charge (Start %)", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_vs_start_soc.png", dpi=300)
    plt.close()

    # 6. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    num_cols = NUMERICAL_FEATURES + [TARGET_COL]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap of Numerical Variables", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_correlation_heatmap.png", dpi=300)
    plt.close()

    # 7. Charging Duration by Charger Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Charger Type", y=TARGET_COL, palette="Blues")
    plt.title("Charging Duration by Charger Type", fontsize=14, pad=12)
    plt.xlabel("Charger Type", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_by_charger_type.png", dpi=300)
    plt.close()

    # 8. Charging Duration by User Type
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="User Type", y=TARGET_COL, palette="Greens")
    plt.title("Charging Duration by User Type", fontsize=14, pad=12)
    plt.xlabel("User Type", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_by_user_type.png", dpi=300)
    plt.close()

    # 9. Charging Duration by Time of Day
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Time of Day", y=TARGET_COL, palette="Purples")
    plt.title("Charging Duration by Time of Day", fontsize=14, pad=12)
    plt.xlabel("Time of Day", fontsize=12)
    plt.ylabel("Charging Duration (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_duration_by_time_of_day.png", dpi=300)
    plt.close()

    print("All EDA plots generated and saved to outputs/plots/")

def generate_regression_plots(y_test, y_pred, residuals):
    """Generates Actual vs Predicted and Residual plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="#1f77b4", label="Test Session Predictions")
    max_val = max(max(y_test), max(y_pred)) + 0.5
    min_val = min(min(y_test), min(y_pred)) - 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal Perfect Fit (Y = X)")
    plt.title("Actual vs. Predicted Charging Duration (Hours)", fontsize=14, pad=12)
    plt.xlabel("Actual Charging Duration (hours)", fontsize=12)
    plt.ylabel("Predicted Charging Duration (hours)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # Residual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color="#d62728")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Residual Plot (Errors vs. Predicted Values)", fontsize=14, pad=12)
    plt.xlabel("Predicted Charging Duration (hours)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_plot.png", dpi=300)
    plt.close()
    
    print("Regression evaluation plots saved to outputs/plots/")

def get_coefficients_table(model, feature_names):
    """Creates a DataFrame of learned regression coefficients."""
    coefs = model.coef_
    df_coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs
    })
    df_coef["Abs_Coefficient"] = df_coef["Coefficient"].abs()
    df_coef = df_coef.sort_values(by="Abs_Coefficient", ascending=False).drop(columns=["Abs_Coefficient"]).reset_index(drop=True)
    return df_coef

def run_evaluation():
    """Main function to run full evaluation and output reports."""
    print("--- Running Model Evaluation & Analytics ---")
    
    # Load raw data for EDA plots
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    generate_eda_plots(cleaned_df)
    
    # Load dataset splits and preprocessor
    (
        X_train, X_test, y_train, y_test,
        X_train_tr, X_test_tr,
        preprocessor, feature_names
    ) = prepare_and_split_data()
    
    # Load trained models
    if not MODEL_PATH.exists() or not BASELINE_PATH.exists():
        raise FileNotFoundError("Trained models not found. Please run src/train_model.py first.")
        
    model = joblib.load(MODEL_PATH)
    baseline = joblib.load(BASELINE_PATH)
    
    # Generate predictions
    y_pred = model.predict(X_test_tr)
    y_pred_base = baseline.predict(X_test_tr)
    residuals = y_test - y_pred
    
    # Compute evaluation metrics
    lr_metrics = compute_metrics(y_test, y_pred)
    base_metrics = compute_metrics(y_test, y_pred_base)
    
    print("\n==========================================")
    print("       MODEL EVALUATION RESULTS           ")
    print("==========================================")
    print("Linear Regression Model Performance:")
    print(f"  MAE  : {lr_metrics['MAE']:.4f} hours")
    print(f"  MSE  : {lr_metrics['MSE']:.4f}")
    print(f"  RMSE : {lr_metrics['RMSE']:.4f} hours")
    print(f"  R²   : {lr_metrics['R2']:.4f}")
    print("------------------------------------------")
    print("Mean Baseline Model Performance:")
    print(f"  MAE  : {base_metrics['MAE']:.4f} hours")
    print(f"  MSE  : {base_metrics['MSE']:.4f}")
    print(f"  RMSE : {base_metrics['RMSE']:.4f} hours")
    print(f"  R²   : {base_metrics['R2']:.4f}")
    print("==========================================")
    
    # Save metrics JSON
    metrics_data = {
        "linear_regression": lr_metrics,
        "mean_baseline": base_metrics,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "intercept": float(model.intercept_)
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    # Generate Regression Plots
    generate_regression_plots(y_test, y_pred, residuals)
    
    # Coefficients Table
    coef_df = get_coefficients_table(model, feature_names)
    print("\nLearned Regression Coefficients (Top 10):")
    print(coef_df.head(10).to_string(index=False))
    
    coef_df.to_csv(METRICS_DIR / "regression_coefficients.csv", index=False)
    print(f"\nAll coefficients saved to {METRICS_DIR / 'regression_coefficients.csv'}")

if __name__ == "__main__":
    run_evaluation()
