"""
predict.py - Reusable prediction function for ChargeTime AI.
"""

import pathlib
import pandas as pd
import numpy as np
import joblib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

def format_hours_to_hours_minutes(decimal_hours):
    """Converts decimal hours (e.g., 2.37) into a readable hours & minutes string (e.g., '2 hours 22 minutes')."""
    if decimal_hours < 0:
        return "0 hours 0 minutes"
    total_minutes = int(round(decimal_hours * 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours} hr" + ("s" if hours > 1 else ""))
    if minutes > 0 or hours == 0:
        parts.append(f"{minutes} min" + ("s" if minutes != 1 else ""))
    return " ".join(parts)

def load_prediction_artifacts():
    """Loads preprocessor and model from disk."""
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Trained model or preprocessor file not found in models/ directory. "
            "Please run 'python src/train_model.py' to build the model files first."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def predict_charging_duration(input_data, model=None, preprocessor=None):
    """
    Receives user input (dict or pandas DataFrame), applies preprocessing, and predicts charging duration.
    
    Parameters:
        input_data (dict or pd.DataFrame): Input charging session parameters.
        model (LinearRegression, optional): Loaded model. If None, loaded from disk.
        preprocessor (ColumnTransformer, optional): Loaded preprocessor. If None, loaded from disk.
        
    Returns:
        dict: Containing:
            - 'predicted_duration_hours': float
            - 'formatted_time': str (e.g. '2 hours 22 minutes')
            - 'is_clipped': bool (True if negative prediction was clipped to 0.1 hours minimum)
    """
    if model is None or preprocessor is None:
        model, preprocessor = load_prediction_artifacts()
        
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("input_data must be a dict or a pandas DataFrame.")
        
    # Transform using preprocessor
    X_transformed = preprocessor.transform(df_input)
    
    # Predict using model
    raw_prediction = float(model.predict(X_transformed)[0])
    
    # Safety handling for negative or zero predictions
    is_clipped = False
    if raw_prediction <= 0:
        # Minimum physical session estimate (e.g. 0.05 hours ~ 3 minutes)
        predicted_hours = 0.05
        is_clipped = True
    else:
        predicted_hours = raw_prediction
        
    formatted_str = format_hours_to_hours_minutes(predicted_hours)
    
    return {
        "predicted_duration_hours": round(predicted_hours, 2),
        "raw_prediction_hours": round(raw_prediction, 4),
        "formatted_time": formatted_str,
        "is_clipped": is_clipped
    }

if __name__ == "__main__":
    # Test sample prediction matching example in user request
    sample_input = {
        "Battery Capacity (kWh)": 75.0,
        "Energy Consumed (kWh)": 42.0,
        "Charging Rate (kW)": 11.0,
        "State of Charge (Start %)": 25.0,
        "State of Charge (End %)": 80.0,
        "Distance Driven (since last charge) (km)": 120.0,
        "Temperature (°C)": 27.0,
        "Vehicle Age (years)": 3.0,
        "Charger Type": "Level 2",
        "Vehicle Model": "Tesla Model 3",
        "Time of Day": "Afternoon",
        "Day of Week": "Monday",
        "User Type": "Commuter"
    }
    
    res = predict_charging_duration(sample_input)
    print("Sample Prediction Result:")
    print(f"Predicted Duration : {res['predicted_duration_hours']} hours")
    print(f"Formatted Time     : {res['formatted_time']}")
    print(f"Clipped Status     : {res['is_clipped']}")
