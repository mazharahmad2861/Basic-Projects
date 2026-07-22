import pandas as pd
from PIL import Image
import numpy as np

#Tabular Data
tabular_data = pd.DataFrame({
    "Category": ["Clothes", "Electronics", "Books", "Clothes"],
    "Price": [20, 100, 15, 25]
})

category_map = {"Clothes": 0, "Electronics": 1, "Books": 2}
token_map = {"Red": 1, "Shirt": 2, "Smartphone": 3, "Book": 4, "Blue": 5, "Jeans": 6}

# Convert categorical column to numbers
tabular_data["Category_Label"] = tabular_data["Category"].map(category_map)

# Keep only numeric columns
tabular_numeric = tabular_data[["Price", "Category_Label"]].values

# Image Data (Grayscale & Flatten)
img = Image.open("sample_product.png").convert("L")
img_array = np.array(img)

# Flatten the array
img_flat = img_array.flatten()

# Text Data (Token IDs)
product_names = ["Red Shirt", "Smartphone", "Book", "Blue Jeans"]
text_vectors = [[token_map[word] for word in name.split()] for name in product_names]

# Combine Everything for Each Product
combined_vectors = []
for i in range(len(tabular_data)):
    combined = np.concatenate([
        tabular_numeric[i],
        img_flat[:10],
        text_vectors[i]
    ])
    combined_vectors.append(combined)

combined_vectors = np.array(combined_vectors, dtype=object)


def main():
    print("Tabular numeric data:\n", tabular_numeric)
    print("\nImage numeric vector (first 10 pixels):", img_flat[:10])
    print("\nText numeric vectors:", text_vectors)
    print("\nCombined numeric vectors for all products:\n", combined_vectors)


if __name__ == "__main__":
    main()
