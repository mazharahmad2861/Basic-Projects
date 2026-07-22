import numpy as np
from sklearn.neural_network import MLPRegressor

# ---- STEP 1: CREATE DATA ----
# Features: [apartment size in sq ft, number of bedrooms]
X = np.array([
    [500, 1],
    [700, 2],
    [900, 2],
    [1100, 3]
])

# Target: monthly rent in USD
y = np.array([1200, 1600, 1800, 2200])

print("Training Data (X):")
print(X)
print("\nTarget Values (y):")
print(y)

# ---- STEP 2: BUILD MLP REGRESSOR ----
model = MLPRegressor(
    hidden_layer_sizes=(4,),  # one hidden layer with 4 neurons
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=0
)

# ---- STEP 3: TRAIN MODEL ----
model.fit(X, y)
print("\nModel training complete!")

# ---- STEP 4: MAKE PREDICTION ----
new_apartment = np.array([[800, 2]])
predicted_rent = model.predict(new_apartment)

print("\nNew Apartment:", new_apartment)
print("Predicted Rent: $", round(predicted_rent[0], 2))


