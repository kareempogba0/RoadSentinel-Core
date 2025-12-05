"""
Car Accident Detection Model using Vision Transformer
This script trains a basic Vision Transformer (ViT) model on car accident detection data.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMAGE_SIZE = 224  # Standard ViT input size
PATCH_SIZE = 16   # Standard patch size for ViT
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
MLP_HEAD_UNITS = [128, 64]
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Data paths
BASE_DIR = Path("data/car-accident-detection-1")
TRAIN_IMAGES_DIR = BASE_DIR / "train" / "images"
TRAIN_LABELS_DIR = BASE_DIR / "train" / "labels"
VALID_IMAGES_DIR = BASE_DIR / "valid" / "images"
VALID_LABELS_DIR = BASE_DIR / "valid" / "labels"
TEST_IMAGES_DIR = BASE_DIR / "test" / "images"
TEST_LABELS_DIR = BASE_DIR / "test" / "labels"


class PatchEncoder(layers.Layer):
    """Layer to encode image patches for Vision Transformer"""
    
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_patches(images, patch_size):
    """Extract patches from images"""
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches


def mlp(x, hidden_units, dropout_rate):
    """Multi-layer perceptron"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier(
    input_shape,
    num_classes,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_layers,
    mlp_head_units,
):
    """Create Vision Transformer model"""
    inputs = layers.Input(shape=input_shape)
    
    # Create patches
    patches = layers.Lambda(lambda x: create_patches(x, patch_size))(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    
    # Classify outputs
    logits = layers.Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(features)
    
    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def load_image(image_path, label_value):
    """Load and preprocess image and use pre-calculated label"""
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0  # Normalize to [0, 1]

    # Ensure label is float32
    label = tf.cast(label_value, tf.float32)

    return img, label


def create_dataset(images_dir, labels_dir, batch_size, shuffle=True):
    """Create TensorFlow dataset from images and labels (Robust Version)"""

    # 1. Look for ALL common image extensions
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(list(Path(images_dir).glob(ext)))

    image_paths = sorted(image_paths)

    images_list = []
    labels_list = []
    accident_count = 0
    no_accident_count = 0

    print(f"\nProcessing {images_dir}...")

    for img_path in image_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")

        # Default: No Accident (Class 0)
        has_accident = 0.0

        # Check label
        if label_path.exists():
            if label_path.stat().st_size > 0:
                has_accident = 1.0
                accident_count += 1
            else:
                no_accident_count += 1
        else:
            no_accident_count += 1

        images_list.append(str(img_path))
        labels_list.append(has_accident)

    print(f"  Found {len(images_list)} total images")
    print(f"  - Accidents: {accident_count}")
    print(f"  - No Accidents: {no_accident_count}")

    if no_accident_count == 0:
        print("  ⚠️ CRITICAL WARNING: No 'Safe' examples found in this set!")
        print("  The model will only learn to predict 'Accident' for everything.")

    # Create dataset
    image_ds = tf.data.Dataset.from_tensor_slices(images_list)
    label_ds = tf.data.Dataset.from_tensor_slices(labels_list)
    ds = tf.data.Dataset.zip((image_ds, label_ds))

    # Load and preprocess
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds, len(images_list)


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate various metrics"""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1-Score
    metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # MSE (Mean Squared Error)
    metrics['mse'] = mean_squared_error(y_true, y_pred_proba)
    
    # RMSE (Root Mean Squared Error)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Initialize metrics with default values
    metrics['precision'] = 0.0
    metrics['recall'] = 0.0
    metrics['specificity'] = 0.0
    metrics['true_positives'] = 0
    metrics['true_negatives'] = 0
    metrics['false_positives'] = 0
    metrics['false_negatives'] = 0
    
    # Calculate Precision, Recall, Specificity
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
    
    return metrics


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    print("=" * 80)
    print("VISION TRANSFORMER FOR CAR ACCIDENT DETECTION")
    print("=" * 80)
    print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Load Datasets
    print("Loading datasets...")
    # We rename variables here to be consistent with the rest of the code
    train_ds, train_count = create_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, BATCH_SIZE)
    val_ds, val_count = create_dataset(VALID_IMAGES_DIR, VALID_LABELS_DIR, BATCH_SIZE, shuffle=False)
    test_ds, test_count = create_dataset(TEST_IMAGES_DIR, TEST_LABELS_DIR, BATCH_SIZE, shuffle=False)

    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")

    # 2. Build Model
    print("\nBuilding Vision Transformer model...")
    model = create_vit_classifier(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=1,  # Binary classification
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        projection_dim=PROJECTION_DIM,
        num_heads=NUM_HEADS,
        transformer_layers=TRANSFORMER_LAYERS,
        mlp_head_units=MLP_HEAD_UNITS,
    )

    # 3. Compile Model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    print("\nModel Architecture:")
    model.summary()

    # 4. Define Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            "best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    # 5. Calculate Class Weights
    print("\nTraining model with Class Weights...")
    print("-" * 80)
    print("Calculating class weights...")

    # Loop through training data to count exact positives and negatives
    y_train = []
    for _, labels in train_ds:
        y_train.extend(labels.numpy())

    neg = len([x for x in y_train if x == 0.0])
    pos = len([x for x in y_train if x == 1.0])
    total = neg + pos

    # Formula: (1 / Class Count) * (Total / 2)
    weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1.0
    weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print(f"Safe Road (Class 0): {neg} images | Weight: {weight_for_0:.2f}")
    print(f"Accident  (Class 1): {pos} images | Weight: {weight_for_1:.2f}")

    # 6. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,  # Variable name is now consistent
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # 7. Evaluate
    plot_training_history(history)

    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)

    y_true_list = []
    y_pred_proba_list = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy())
        y_pred_proba_list.extend(predictions.flatten())

    y_true = np.array(y_true_list)
    y_pred_proba = np.array(y_pred_proba_list)
    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

    print("\n📊 MODEL PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"Specificity:    {metrics['specificity']:.4f}")

    print("\n📈 CONFUSION MATRIX:")
    print("-" * 80)
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT:")
    print("=" * 80)

    unique_labels = np.unique(y_true)
    target_names = ['No Accident', 'Accident']

    if len(unique_labels) == 1:
        print(f"⚠️ WARNING: Test set contains only one class: {unique_labels[0]}")
    else:
        print(classification_report(y_true, y_pred, target_names=target_names))

    # Save metrics logic...
    metrics_output = {
        'model_type': 'Vision Transformer (ViT)',
        'metrics': metrics
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_output, f, indent=4)

    print("\n✅ Metrics and Model saved.")
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return model, metrics


if __name__ == "__main__":
    # Enable mixed precision for faster training (optional)
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    model, metrics = main()
