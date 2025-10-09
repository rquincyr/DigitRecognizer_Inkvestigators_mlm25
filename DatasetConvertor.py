import pandas as pd
import numpy as np
import os
from PIL import Image

# File paths (escape backslashes OR use raw strings with r"")
train_path = r"C:\Users\27960\Desktop\学习资料（真)\Digit recongnizer\train.csv"
test_path  = r"C:\Users\27960\Desktop\学习资料（真)\Digit recongnizer\test.csv"

# Load datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Create output folders near your CSV files
output_train = r"C:\Users\27960\Desktop\学习资料（真)\Digit recongnizer\train_images"
output_test  = r"C:\Users\27960\Desktop\学习资料（真)\Digit recongnizer\test_images"

os.makedirs(output_train, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

# Convert train dataset to images
for idx, row in train.iterrows():
    label = row["label"]
    pixels = row.drop("label").values.reshape(28, 28).astype(np.uint8)
    img = Image.fromarray(pixels, mode="L")  # "L" = grayscale

    # Save as "index_label.png"
    img.save(os.path.join(output_train, f"{idx}_{label}.png"))

    if idx < 5:
        print(f"Saved {idx}_{label}.png")

# Convert test dataset to images
for idx, row in test.iterrows():
    pixels = row.values.reshape(28, 28).astype(np.uint8)
    img = Image.fromarray(pixels, mode="L")

    # Save as "index.png"
    img.save(os.path.join(output_test, f"{idx}.png"))

    if idx < 5:
        print(f"Saved {idx}.png")

