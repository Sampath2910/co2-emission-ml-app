from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import joblib
import numpy as np

# =========================
# Flask Configuration
# =========================

app = Flask(
    __name__,
    static_folder="../frontend",
    template_folder="../frontend"
)

CORS(app)

# =========================
# Resolve Paths Safely
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "co2_emission_xgb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# =========================
# Load ML Artifacts
# =========================

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Model, features, and scaler loaded successfully")

# =========================
# Serve Frontend
# =========================

@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# =========================
# Prediction API
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Build feature dict
        input_data = {
            "Engine Size(L)": float(data["Engine Size(L)"]),
            "Cylinders": int(data["Cylinders"]),
            "Fuel Consumption Comb (L/100 km)": float(data["Fuel Consumption Comb (L/100 km)"])
        }

        fuel_type = data["Fuel Type"]

        # One-hot encoding (same as training)
        for ft in ["Fuel Type_E", "Fuel Type_N", "Fuel Type_X", "Fuel Type_Z"]:
            input_data[ft] = 0

        input_data[f"Fuel Type_{fuel_type}"] = 1

        # Convert to ordered array
        feature_vector = np.array(
            [input_data[col] for col in feature_columns]
        ).reshape(1, -1)

        # Scale input
        feature_vector = scaler.transform(feature_vector)

        # Predict
        prediction = float(model.predict(feature_vector)[0])

        # Emission classification
        if prediction < 120:
            rating = "Low Emission"
            alert = "✅ Eco-friendly"
        elif prediction < 180:
            rating = "Moderate Emission"
            alert = "⚠ Average emission"
        else:
            rating = "High Emission"
            alert = "❌ High emission – not eco-friendly"

        return jsonify({
            "predicted_co2_gkm": round(prediction, 2),
            "rating": rating,
            "alert": alert
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =========================
# Run App
# =========================

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)

