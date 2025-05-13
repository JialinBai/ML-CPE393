from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model2.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = data["features"]
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list"}), 400

    if isinstance(features[0], list):
        if not all(isinstance(row, list) and len(row) == 13 for row in features):
            return jsonify({"error": "Each input must be a list with 13 values"}), 400
    else:
        if len(features) != 13:
            return jsonify({"error": "Input must be a list with exactly 13 values"}), 400
        features = [features]

    input_features = np.array(features)

    predictions = model.predict(input_features)

    tree_predictions = np.stack([tree.predict(input_features) for tree in model.estimators_], axis=1)
    std_devs = np.std(tree_predictions, axis=1)

    max_std = std_devs.max() + 1e-6
    confidences = 1 - (std_devs / max_std)

    results = [
        {
            "prediction": round(float(pred), 2),
            "confidence": round(float(conf), 2)
        }
        for pred, conf in zip(predictions, confidences)
    ]

    if len(features) > 1:
        return jsonify({"predictions": results})

    return jsonify(results[0])

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
