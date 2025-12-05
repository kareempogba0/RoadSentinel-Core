import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224


class RoadSentinelPredictor:
    def __init__(self, model_path='models/best_model_mobilenet.keras'):
        self.model_path = model_path
        self.model = None
        self.load_inference_model()

    def load_inference_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading MobileNetV2 from {self.model_path}...")
        # MobileNet is standard Keras, no custom objects needed!
        self.model = load_model(self.model_path)
        print("✅ Model loaded.")

    def predict(self, image_file, threshold=0.5):
        try:
            # Preprocess
            if isinstance(image_file, str):
                img = load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))
            else:
                img = load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))

            img_array = img_to_array(img)
            img_array = img_array / 255.0  # MobileNet expects 0-1 range usually
            img_array = np.expand_dims(img_array, axis=0)

            # Inference
            prediction = self.model.predict(img_array, verbose=0)
            score = float(prediction[0][0])

            # Logic
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