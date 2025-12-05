import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration for MobileNetV2
IMG_SIZE = 224


class RoadSentinelPredictor:
    def __init__(self, model_path='models/best_model_mobilenet.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the MobileNetV2 model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading MobileNet from {self.model_path}...")

        # MobileNet is a standard Keras model, so we don't need custom objects!
        # compile=False prevents errors if the optimizer state is weird
        self.model = load_model(self.model_path, compile=False)
        print("✅ Model loaded successfully.")

    def predict(self, image_file, threshold=0.5):
        try:
            # 1. Preprocess
            # Handle both file path (string) and file object (Streamlit)
            if isinstance(image_file, str):
                img = load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))
            else:
                img = load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))

            img_array = img_to_array(img)
            img_array = img_array / 255.0  # MobileNet expects 0-1 range
            img_array = np.expand_dims(img_array, axis=0)

            # 2. Inference
            prediction = self.model.predict(img_array, verbose=0)
            score = float(prediction[0][0])

            # 3. Logic (Dynamic Thresholding)
            if score > threshold:
                label = "ACCIDENT"
                confidence = score
                is_safe = False
                color = "red"
            else:
                label = "SAFE ROAD"
                confidence = 1.0 - score
                is_safe = True
                color = "green"

            return {
                "label": label,
                "confidence": confidence,
                "raw_score": score,
                "is_safe": is_safe,
                "color": color
            }

        except Exception as e:
            return {"error": str(e)}