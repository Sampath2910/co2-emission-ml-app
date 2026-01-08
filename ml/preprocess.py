# preprocess.py
# -----------------------------------
# Data preprocessing module for
# CO2 Emission Prediction System

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


FEATURES_PATH = "models/feature_columns.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    required_columns = [
        "Engine Size(L)",
        "Cylinders",
        "Fuel Type",
        "Fuel Consumption Comb (L/100 km)",
        "CO2 Emissions(g/km)"
    ]

    df = df[required_columns]
    df.dropna(inplace=True)

    # One-hot encode Fuel Type
    df = pd.get_dummies(df, columns=["Fuel Type"])

    # Separate features and target
    X = df.drop("CO2 Emissions(g/km)", axis=1)
    y = df["CO2 Emissions(g/km)"]

    # Scale numeric columns
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    # Save feature columns & scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(list(X.columns), FEATURES_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return X, y
