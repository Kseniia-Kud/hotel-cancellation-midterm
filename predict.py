"""Web service: POST /predict with one booking as JSON -> cancellation probability.

Run locally:  python predict.py            (or: gunicorn --bind 0.0.0.0:9696 predict:app)
"""
import joblib
from flask import Flask, jsonify, request

artifact = joblib.load("model.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
frequent_countries = set(artifact["frequent_countries"])
features = artifact["features"]

app = Flask("hotel-cancellation")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    booking = request.get_json()
    missing = [f for f in features if f not in booking]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    row = {f: booking[f] for f in features}
    if row["country"] not in frequent_countries:      # same rule as in training
        row["country"] = "other"

    proba = float(model.predict_proba([row])[0, 1])
    return jsonify({"cancellation_probability": round(proba, 3),
                    "threshold": round(threshold, 3),
                    "will_cancel": bool(proba >= threshold)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
