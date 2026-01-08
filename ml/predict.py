# predict.py
# -----------------------------------
# Prediction module for CO2 Emission System

import joblib
import pandas as pd

MODEL_PATH = "models/co2_emission_xgb_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
SCALER_PATH = "models/scaler.pkl"


def predict_co2(input_data):
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode fuel type
    df = pd.get_dummies(df, columns=["Fuel Type"])

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[feature_columns]

    # Scale input
    df[df.columns] = scaler.transform(df[df.columns])

    prediction = model.predict(df)[0]
    return round(prediction, 2)
