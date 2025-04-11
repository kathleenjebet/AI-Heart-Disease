import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model once
model_path = os.path.join("backend", "model", "heart_disease_model.h5")
model = tf.keras.models.load_model(model_path)

def predict_heart_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)[0][0]
        result = "Positive for Heart Disease" if prediction > 0.5 else "Negative for Heart Disease"
        return {
            "result": result,
            "confidence": round(float(prediction), 3)
        }
    except Exception as e:
        return {"error": str(e)}
