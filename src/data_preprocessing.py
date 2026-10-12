"""
data_preprocessing.py - Data loading, cleaning, and preprocessing pipeline for ChargeTime AI.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "ev_charging_patterns.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "ev_charging_processed.csv"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

NUMERICAL_FEATURES = [
    "Battery Capacity (kWh)",
    "Energy Consumed (kWh)",
    "Charging Rate (kW)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
]

CATEGORICAL_FEATURES = [
    "Vehicle Model",
    "Charger Type",
    "Time of Day",
    "Day of Week",
    "User Type",
]

TARGET_COL = "Charging Duration (hours)"

# Columns dropped to avoid target leakage / metadata redundancy
DROPPED_COLS = [
    "User ID",
    "Charging Station ID",
    "Charging Station Location",
    "Charging Start Time",
    "Charging End Time",  # Direct target leakage
    "Charging Cost (USD)",  # Post-session financial outcome
]

def load_raw_data(data_path=RAW_DATA_PATH):
    """Loads the raw EV charging dataset from CSV."""
    if not pathlib.Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset file not found at {data_path}. "
            "Please ensure ev_charging_patterns.csv is placed in data/raw/"
        )
    df = pd.read_csv(data_path)
    return df

def clean_data(df):
    """
    Cleans dataset by handling missing values, duplicates, and string trimming.
    Returns cleaned DataFrame.
    """
    df_cleaned = df.copy()
    
    # Drop duplicates if any
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Strip leading/trailing spaces from categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
    # Handle missing values if any
    # Numerical: median imputation, Categorical: mode imputation
    for col in NUMERICAL_FEATURES:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            
    for col in CATEGORICAL_FEATURES:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            
    return df_cleaned

def build_preprocessor():
    """Constructs the ColumnTransformer preprocessing pipeline."""
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_and_split_data(data_path=RAW_DATA_PATH, test_size=0.2, random_state=42):
    """
    Full data pipeline:
    1. Loads raw dataset
    2. Cleans data
    3. Extracts features X and target y
    4. Performs Train-Test Split
    5. Fits preprocessor ONLY on training set to prevent data leakage
    6. Transforms X_train and X_test
    7. Saves preprocessor object to models/preprocessor.pkl
    """
    df = load_raw_data(data_path)
    df_cleaned = clean_data(df)
    
    X = df_cleaned[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df_cleaned[TARGET_COL].copy()
    
    # Save processed dataset copy
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract feature names after OneHotEncoding
    feature_names = get_feature_names(preprocessor)
    
    # Save preprocessor
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    return X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed, preprocessor, feature_names

def get_feature_names(preprocessor):
    """Extracts post-transformation feature names from ColumnTransformer."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            cat_encoder = trans
            cat_cols = cat_encoder.get_feature_names_out(cols)
            feature_names.extend(cat_cols)
    return feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_tr, X_te, prep, f_names = prepare_and_split_data()
    print("Data Preprocessing Completed Successfully!")
    print(f"X_train shape: {X_train.shape} -> Transformed: {X_tr.shape}")
    print(f"X_test shape: {X_test.shape} -> Transformed: {X_te.shape}")
    print(f"Feature count post-encoding: {len(f_names)}")
