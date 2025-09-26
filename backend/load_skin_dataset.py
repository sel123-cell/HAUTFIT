import os
import numpy as np
from PIL import Image

# Dataset folder (relative to backend/)
dataset_dir = "dataset"

# Folder names
skin_tones = ["dark", "mid-dark", "mid-light", "light"]

# Map folder names to numeric labels
label_map = {"dark": 0, "mid-dark": 1, "mid-light": 2, "light": 3}

# Image size
IMG_SIZE = (64, 64)

# Lists to store images and labels
X = []
y = []

# Loop through each folder
for tone in skin_tones:
    folder_path = os.path.join(dataset_dir, tone)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    label = label_map[tone]
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Open image, convert to RGB, resize
            img = Image.open(file_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img_array = np.array(img)
            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Skipped {file_path}: {e}")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Dataset loaded successfully!")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)

# Save as compressed file for faster future loading
np.savez_compressed("skin_dataset.npz", X=X, y=y)
