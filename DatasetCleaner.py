import pandas as pd
import numpy as np

# Load training dataset
train_path = r"C:\Users\27960\Desktop\学习资料（真)\Digit recongnizer\train.csv"
train = pd.read_csv(train_path)

print("=== Data Cleaning Report ===")

# 1. Check missing values
missing = train.isnull().sum().sum()
print(f"Missing values: {missing}")

# 2. Pixel value ranges
pixel_data = train.drop("label", axis=1).values
min_val, max_val = pixel_data.min(), pixel_data.max()
print(f"Pixel range: {min_val} to {max_val}")

# 3. Check shape
print(f"Train shape: {train.shape} (rows, columns)")

# 4. Class balance
print("Label counts:\n", train["label"].value_counts().sort_index())

# 5. Check duplicates
duplicates = train.drop("label", axis=1).duplicated().sum()
print(f"Duplicate images: {duplicates}")

# 6. Check empty images
empty_imgs = np.sum(pixel_data == 0, axis=1) == 784
print(f"Empty images: {empty_imgs.sum()}")

# 7. Normalize (scale to 0–1)
pixel_data = pixel_data / 255.0
print("Normalization done (all pixel values now between 0 and 1)")

# 8. Save cleaned dataset
cleaned_train = pd.DataFrame(pixel_data)
cleaned_train.insert(0, "label", train["label"])
output_path = r"C:\Users\27960\Desktop\学习资料（真)\Digit recongnizer\train_cleaned.csv"
cleaned_train.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}")
