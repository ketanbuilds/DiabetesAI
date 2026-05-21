"""
Diabetes Prediction System — Flask Backend
Shri Vaishnav Institute of Information Technology
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# ── Load model & scaler once at startup ──────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(BASE, "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))

FEATURE_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# ── HTML (full single-page app embedded) ─────────────────────────────────────
HTML = open(os.path.join(BASE, "index.html")).read()

@app.route("/")
def index():
    return HTML

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array([[float(data[f]) for f in FEATURE_COLS]])
        scaled   = scaler.transform(features)
        pred     = int(model.predict(scaled)[0])
        prob     = float(model.predict_proba(scaled)[0][1])

        # Risk factors analysis
        glucose = float(data['Glucose'])
        bmi     = float(data['BMI'])
        age     = float(data['Age'])
        bp      = float(data['BloodPressure'])
        insulin = float(data['Insulin'])
        dpf     = float(data['DiabetesPedigreeFunction'])

        risk_factors = []
        if glucose > 140:  risk_factors.append({"label": "High Glucose", "value": f"{glucose} mg/dL", "level": "high"})
        elif glucose > 100: risk_factors.append({"label": "Borderline Glucose", "value": f"{glucose} mg/dL", "level": "medium"})
        else:               risk_factors.append({"label": "Normal Glucose", "value": f"{glucose} mg/dL", "level": "low"})

        if bmi > 30:   risk_factors.append({"label": "Obese BMI", "value": f"{bmi}", "level": "high"})
        elif bmi > 25: risk_factors.append({"label": "Overweight BMI", "value": f"{bmi}", "level": "medium"})
        else:          risk_factors.append({"label": "Normal BMI", "value": f"{bmi}", "level": "low"})

        if bp > 90:   risk_factors.append({"label": "High Blood Pressure", "value": f"{bp} mmHg", "level": "high"})
        elif bp > 80: risk_factors.append({"label": "Elevated BP", "value": f"{bp} mmHg", "level": "medium"})
        else:         risk_factors.append({"label": "Normal BP", "value": f"{bp} mmHg", "level": "low"})

        if age > 45:  risk_factors.append({"label": "Age Risk", "value": f"{int(age)} years", "level": "high"})
        elif age > 35: risk_factors.append({"label": "Moderate Age Risk", "value": f"{int(age)} years", "level": "medium"})
        else:          risk_factors.append({"label": "Low Age Risk", "value": f"{int(age)} years", "level": "low"})

        if dpf > 0.8:  risk_factors.append({"label": "High Genetic Risk", "value": f"{dpf:.3f}", "level": "high"})
        elif dpf > 0.4: risk_factors.append({"label": "Moderate Genetic Risk", "value": f"{dpf:.3f}", "level": "medium"})
        else:           risk_factors.append({"label": "Low Genetic Risk", "value": f"{dpf:.3f}", "level": "low"})

        # Feature importance from model
        importances = model.feature_importances_.tolist()
        feature_data = [
            {"name": n, "importance": round(v * 100, 1)}
            for n, v in zip(FEATURE_COLS, importances)
        ]
        feature_data.sort(key=lambda x: x["importance"], reverse=True)

        return jsonify({
            "prediction": pred,
            "probability": round(prob * 100, 1),
            "label": "DIABETIC" if pred == 1 else "NON-DIABETIC",
            "risk_factors": risk_factors,
            "feature_importance": feature_data,
            "input": {k: data[k] for k in FEATURE_COLS}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Diabetes Prediction System")
    print("  Open your browser → http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000)
