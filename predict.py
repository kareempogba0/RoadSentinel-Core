import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ---------------------------------------------------------
# 1. CUSTOM LAYERS
# ---------------------------------------------------------
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config


# ---------------------------------------------------------
# 2. MODEL ARCHITECTURE (Must match train.py)
# ---------------------------------------------------------
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
MLP_HEAD_UNITS = [128, 64]
NUM_CLASSES = 1


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    patches = Patches(PATCH_SIZE)(inputs)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[PROJECTION_DIM * 2, PROJECTION_DIM], dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    logits = layers.Dense(NUM_CLASSES)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# ---------------------------------------------------------
# 3. PREDICTION LOGIC
# ---------------------------------------------------------
MODEL_PATH = 'models/best_model.h5'


def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found.")
        return

    print("\n------------------------------------------------")
    print("🤖 RoadSentinel Inference Engine")
    print("------------------------------------------------")

    try:
        # Build empty structure
        model = create_vit_classifier()
        # Load weights
        model.load_weights(MODEL_PATH)
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    try:
        # Load image
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Batch of 1

        # Predict
        logits = model.predict(img_array, verbose=0)

        # Apply Sigmoid to get probability (0.0 to 1.0)
        probability = tf.nn.sigmoid(logits[0][0])
        score = float(probability)

        # Logic: In our training, 1.0 = Accident, 0.0 = Safe
        print(f"\n📸 Image: {os.path.basename(image_path)}")
        print(f"📊 Raw Probability: {score:.4f}")

        if score > 0.5:
            confidence = score
            print(f"🚨 RESULT: ACCIDENT DETECTED ({confidence:.1%} confidence)")
        else:
            confidence = 1.0 - score
            print(f"✅ RESULT: SAFE ROAD ({confidence:.1%} confidence)")

    except Exception as e:
        print(f"❌ Error processing image: {e}")


if __name__ == "__main__":
    import glob

    # Try to find a test image automatically
    test_dir = "data/car-accident-detection-1/test/images"
    files = glob.glob(f"{test_dir}/*.jpg") + glob.glob(f"{test_dir}/*.jpeg")

    if files:
        # Pick the first image found
        predict_image(files[0])
    else:
        print(f"No images found in {test_dir}")