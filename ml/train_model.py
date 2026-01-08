# train_model.py
# -----------------------------------
# Train XGBoost model for CO2 Emission Prediction

import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .preprocess import load_and_preprocess_data


def train_and_save_model():
    # Load and preprocess data
    X, y = load_and_preprocess_data("data/CO2 Emissions_Canada.csv")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Initialize XGBoost Regressor
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    print("XGBoost Model Training Completed")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Save model
    joblib.dump(model, "models/co2_emission_xgb_model.pkl")
    print("Model saved successfully in models/ folder")


if __name__ == "__main__":
    train_and_save_model()
