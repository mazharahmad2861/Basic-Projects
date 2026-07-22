from sklearn.datasets import load_digits
import numpy as np


def load_data():
    """Load the digits dataset into arrays."""
    digits = load_digits()       # Load the dataset
    X = digits.images            # Extract images (shape: n_samples x 8 x 8)
    y = digits.target            # Extract labels (shape: n_samples,)
    return X, y                  # Return images and labels
    

def explore_data(X, y):
    """
    Analyze the dataset: shapes, class distribution, and value range.
    
    Args:
        X (np.ndarray): Images array
        y (np.ndarray): Labels array
    
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {}

    # Total number of images
    num_samples = X.shape[0]
    info['num_samples'] = num_samples

    # Shape of one image (height, width)
    image_shape = X.shape[1:]   # (8, 8)
    info['image_shape'] = image_shape

    # Find all unique class labels
    classes = np.unique(y)      # e.g., [0,1,2,...,9]
    info['num_classes'] = len(classes)

    # Count how many images are in each class
    class_counts = {}
    for cls in classes:
        count = np.sum(y == cls)  # count images of this class
        class_counts[cls] = count
    info['class_distribution'] = class_counts

    # Find the minimum and maximum pixel values
    min_pixel = X.min()
    max_pixel = X.max()
    info['pixel_range'] = (min_pixel, max_pixel)

    # Check if there are any missing values
    has_missing = np.isnan(X).any() or np.isnan(y).any()
    info['has_missing_values'] = has_missing

    return info


if __name__ == "__main__":
    # Load dataset
    X, y = load_data()
    
    dataset_info = explore_data(X, y)
    
    # Print the dataset information
    print("\n=== Digits Dataset Information ===")
    print(f"Total samples       : {dataset_info['num_samples']}")
    print(f"Image shape         : {dataset_info['image_shape']}")
    print(f"Number of classes   : {dataset_info['num_classes']}")
    print(f"Pixel value range   : {dataset_info['pixel_range']}")
    print(f"Has missing values? : {dataset_info['has_missing_values']}")
    print("\nClass distribution:")
    for cls, count in dataset_info['class_distribution'].items():
        print(f"  Class {cls}: {count} samples")
