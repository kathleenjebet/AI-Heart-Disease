from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enabling CORS for cross-origin requests

model = tf.keras.models.load_model(os.path.join("model", "heart_disease_model.h5"))

def prepare_image(img_path):
    img = Image.open(img_path)  # Open image using PIL
    img = img.resize((224, 224))  # Resize image
    img_array = np.array(img, dtype=np.float32)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    try:
        blood_pressure = float(request.form.get("blood_pressure", 0))
        cholesterol = float(request.form.get("cholesterol", 0))
        heart_rate = float(request.form.get("heart_rate", 0))
        age = int(request.form.get("age", 0))
    except ValueError:
        return jsonify({"error": "Invalid numeric inputs"}), 400

    img_array = prepare_image(filepath)
    prediction = model.predict(img_array)[0][0]

    result = "Positive for Heart Disease" if prediction > 0.5 else "Negative for Heart Disease"

    return jsonify({
        "result": result,
        "confidence": float(prediction),
        "inputs": {
            "blood_pressure": blood_pressure,
            "cholesterol": cholesterol,
            "heart_rate": heart_rate,
            "age": age
        }
    })

if __name__ == "__main__":
    app.run(debug=True)

