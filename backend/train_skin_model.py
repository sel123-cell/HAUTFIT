import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Removed sklearn and to_categorical because Keras dataset handles splitting automatically now

# --- 1. CONFIGURATION (New Section Needed) ---
# Go UP one level ("..") to find the dataset folder
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset_skin_tones")
IMG_HEIGHT = 180  # Increased from 64 to 180 for better accuracy with your new photos
IMG_WIDTH = 180
BATCH_SIZE = 32

# --- 2. LOAD DATASET (Replaces np.load and train_test_split) ---
# This part is changed because we are reading folders now, not an .npz file
print("Loading images from folders...")

# Load Training Data (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Load Validation Data (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Get class names to ensure they are correct (Should be Chinito, Mestizo, Moreno)
class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Performance tweak (optional but highly recommended for 3000 images)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. BUILD MODEL (Kept your structure, slight updates) ---
model = Sequential([
    # Updated input_shape to 180,180,3 to match the loader above
    # Added Rescaling layer to normalize pixels (0-1) automatically
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    
    # CHANGED: 4 -> 3 (Because you now have Chinito, Mestizo, Moreno)
    Dense(3, activation='softmax') 
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', # Changed to 'sparse' because labels are integers (0,1,2) now, not one-hot
              metrics=['accuracy'])

# --- 4. TRAIN ---
# Updated to use the new 'train_ds' and 'val_ds' variables
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save
model.save("skin_tone_cnn.h5")
print("✅ Model trained and saved as skin_tone_cnn.h5")