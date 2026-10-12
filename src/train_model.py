"""
train_model.py - Model training script for ChargeTime AI using Linear Regression.
"""

import pathlib
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from data_preprocessing import prepare_and_split_data

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"
BASELINE_PATH = BASE_DIR / "models" / "baseline_model.pkl"

def train_linear_regression(X_train_transformed, y_train):
    """
    Trains primary Linear Regression model on preprocessed training data.
    """
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    return model

def train_mean_baseline(X_train_transformed, y_train):
    """
    Trains a Mean Prediction Baseline model (DummyRegressor) for comparison.
    """
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train_transformed, y_train)
    return baseline

def run_training():
    """
    Executes complete training pipeline:
    1. Preprocesses and splits data
    2. Trains Linear Regression model
    3. Trains Baseline model
    4. Saves models to disk
    """
    print("--- Starting ChargeTime AI Model Training ---")
    (
        X_train, X_test, y_train, y_test,
        X_train_tr, X_test_tr,
        preprocessor, feature_names
    ) = prepare_and_split_data()
    
    # Train Linear Regression model
    model = train_linear_regression(X_train_tr, y_train)
    print("Linear Regression model successfully trained.")
    
    # Train Baseline model
    baseline = train_mean_baseline(X_train_tr, y_train)
    print("Mean Baseline model successfully trained.")
    
    # Save models
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(baseline, BASELINE_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Baseline saved to: {BASELINE_PATH}")
    
    return model, baseline, preprocessor, X_train, X_test, y_train, y_test, X_train_tr, X_test_tr, feature_names

if __name__ == "__main__":
    run_training()
