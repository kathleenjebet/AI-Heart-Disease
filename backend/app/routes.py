from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from .model import predict_heart_disease

main = Blueprint('main', __name__)

@main.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join('static', filename)
    file.save(filepath)

    prediction = predict_heart_disease(filepath)
    return jsonify(prediction)
