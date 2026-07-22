import numpy as np
from sklearn.neural_network import MLPClassifier

# 1. Create your dataset
X = np.array([[1],[2],[3],[4],[5],[6]])  # Example: hours of exercise
y = np.array([0,0,1,1,2,2])              # 0=Low, 1=Medium, 2=High

# 2. Build MLP Classifier
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(5,), 
    activation='relu', 
    max_iter=1000, 
    random_state=42
)

# 3. Train the model
mlp_clf.fit(X, y)

# 4. Predict probabilities and classes
probs = mlp_clf.predict_proba(X)
predictions = mlp_clf.predict(X)

# 5. Print results
import numpy as np
print("Predicted probabilities:\n", np.round(probs,2))
print("Predicted classes:", predictions)