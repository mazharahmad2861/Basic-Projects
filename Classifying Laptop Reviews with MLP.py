from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Example dataset
reviews = [
    "The laptop is fast and works perfectly",
    "Battery life is horrible and very disappointing",
    "Excellent performance and sleek design",
    "The screen quality is poor and colors are dull"
]

labels = [
    "positive",
    "negative",
    "positive",
    "negative"
]

# Step 2: Convert text into numeric vectors
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(reviews)  # Fill in the method

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.5, random_state=42
)

# Step 4: Define and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = mlp.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Step 6: Predict sentiment for a new review
new_review = ["I am very happy with this laptop"]
new_vector = tfidf_vectorizer.transform(new_review)
prediction = mlp.predict(new_vector)

print("New review prediction:", prediction[0])
