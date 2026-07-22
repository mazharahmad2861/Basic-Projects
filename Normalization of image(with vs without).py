from PIL import Image, ImageEnhance
import numpy as np
from sklearn.neural_network import MLPRegressor

# 1. Load smiley face image, convert to grayscale, resize
img = Image.open("smiley.png").convert("L")
img = img.resize((28, 28))

# 2. Create a slightly modified version (brightness change)
enhancer = ImageEnhance.Brightness(img)
img_modified = enhancer.enhance(1.2)   # slightly brighter

# 3. Convert images to NumPy arrays
arr_original = np.array(img)
arr_modified = np.array(img_modified)

# 4. Compare pixel values
print("Original image pixel sample:", arr_original[0][:5])
print("Modified image pixel sample:", arr_modified[0][:5])

difference = np.abs(arr_original - arr_modified)
print("Average pixel difference:", difference.mean())
print("Maximum pixel difference:", difference.max())

# 5. Flatten images for MLP input
X_original = arr_original.flatten().reshape(1, -1)
X_modified = arr_modified.flatten().reshape(1, -1)

# Dummy regression target
y = [1]

# 6. Train MLP WITHOUT normalization
mlp_raw = MLPRegressor(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

mlp_raw.fit(X_original, y)
pred_raw = mlp_raw.predict(X_modified)

print("Prediction WITHOUT normalization:", round(pred_raw[0], 2))

# 7. Normalize pixel values (0–255 → 0–1)
X_original_norm = X_original / 255.0
X_modified_norm = X_modified / 255.0

# 8. Train MLP WITH normalization
mlp_norm = MLPRegressor(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

mlp_norm.fit(X_original_norm, y)
pred_norm = mlp_norm.predict(X_modified_norm)

print("Prediction WITH normalization:", round(pred_norm[0], 2))
