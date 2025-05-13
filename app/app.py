from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

@app.route("/predict", methods=["POST"])
def predict():
   
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    if not isinstance(data["features"], list):
        return jsonify({"error": "'features' must be a list"}), 400

    if not all(isinstance(row, list) and len(row) == 13 for row in data["features"]):
        return jsonify({"error": "Each input must be a list with 13 values"}), 400

    input_features = np.array(data["features"])

    predictions = model.predict(input_features)

    Tree_predictions = np.stack([tree.predict(input_features) for tree in model.estimators_], axis=1)
    std_devs = np.std(Tree_predictions, axis=1)

    max_std = std_devs.max() + 1e-6  
    confidences = 1 - (std_devs / max_std)

    results = [
        {
            "prediction": round(float(pred), 2),
            "confidence": round(float(conf), 2)
        }
        for pred, conf in zip(predictions, confidences)
    ]

    return jsonify(results)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000) #check your port number ( if it is in use, change the port number)