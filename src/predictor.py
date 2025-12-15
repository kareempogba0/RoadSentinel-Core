import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration for MobileNetV2
IMG_SIZE = 224


# --- MODEL ARCHITECTURE DEFINITION (Exact match to train_transfer.py) ---
def create_mobilenet_classifier():
    """Builds the MobileNetV2 architecture identical to train_transfer.py"""

    # 1. Base Model (Loads structure, weights will be overwritten later)
    base_model = MobileNetV2(
        weights='imagenet',  # Load ImageNet weights to establish structure
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # Keep frozen

    # 2. Custom Top Layers (Must match training structure)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])

    return model


# ---------------------------------------------------------------


class RoadSentinelPredictor:
    def __init__(self, model_path='models/best_model_mobilenet.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Builds the architecture from code and loads ONLY weights (Final fix for batch_shape error)."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading MobileNet structure from code...")

        # 1. Build the clean, compatible model architecture
        self.model = create_mobilenet_classifier()

        # 2. Load ONLY the weights (Bypasses the batch_shape and version conflicts)
        self.model.load_weights(self.model_path)

        print(f"✅ Model loaded successfully from weights ({self.model_path}).")

    def predict(self, image_file, threshold=0.5):
        try:
            # 1. Preprocess
            if isinstance(image_file, str):
                img = load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))
            else:
                img = load_img(image_file, target_size=(IMG_SIZE, IMG_SIZE))

            img_array = img_to_array(img)
            img_array = img_array / 255.0
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