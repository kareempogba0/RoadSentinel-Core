import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import datetime

# 1. SETUP
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Paths (Keep your existing structure)
BASE_DIR = Path("data/car-accident-detection-1")
TRAIN_DIR = BASE_DIR / "train" / "images"
VALID_DIR = BASE_DIR / "valid" / "images"
TEST_DIR = BASE_DIR / "test" / "images"

# 2. DATA GENERATORS (With Augmentation to fake more data)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

print("Loading Data...")


# Automatically finds images and labels them based on folder structure
# NOTE: MobileNet expects folders like: data/train/accident/img.jpg
# Since your data is flat (all images in one folder), we need the dataframe method or flow_from_directory
# BUT your data is messy. Let's stick to the manual list method we used before but robustified.

def load_data_paths(img_dir, lbl_dir):
    # This rebuilds your custom logic to pair images with labels
    import glob
    img_paths = glob.glob(str(img_dir) + "/*.jpg") + glob.glob(str(img_dir) + "/*.jpeg") + glob.glob(
        str(img_dir) + "/*.png")

    file_list = []
    labels = []

    for img_p in img_paths:
        lbl_p = Path(str(img_p).replace("images", "labels")).with_suffix(".txt")

        # Logic: File Exists & Not Empty = Accident (1), Else = Safe (0)
        is_accident = 0
        if lbl_p.exists() and lbl_p.stat().st_size > 0:
            is_accident = 1

        file_list.append(str(img_p))
        labels.append(is_accident)

    return file_list, labels


# Load lists
X_train, y_train = load_data_paths(TRAIN_DIR, BASE_DIR / "train" / "labels")
X_val, y_val = load_data_paths(VALID_DIR, BASE_DIR / "valid" / "labels")


# Convert to TF Dataset
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # Normalize
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(process_path).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(process_path).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Training on {len(X_train)} images. Validation on {len(X_val)} images.")

# 3. BUILD MODEL (MobileNetV2)
base_model = MobileNetV2(
    weights='imagenet',  # Load pre-trained "knowledge"
    include_top=False,  # Remove the final classification layer (we build our own)
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Freeze base model initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Binary output: 0=Safe, 1=Accident
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 4. TRAIN
print("\nStarting Transfer Learning...")
checkpoint = ModelCheckpoint("models/best_model_mobilenet.keras", monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Calculate weights again (Crucial for your unbalanced data)
neg = y_train.count(0)
pos = y_train.count(1)
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1.0
weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1.0
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"Class Weights -> Safe: {weight_for_0:.2f}, Accident: {weight_for_1:.2f}")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weight
)

print("✅ Training Complete. Model saved to models/best_model_mobilenet.keras")