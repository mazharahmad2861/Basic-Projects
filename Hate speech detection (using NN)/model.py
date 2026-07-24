import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- File Paths ---
INPUT_FILE = 'cleaned_hate_dataset.csv'
OUTPUT_FILE = 'hate_speech_model.pkl'

def train_model():
    """Main function to train and save the hate speech classification model."""
    print("Starting Model Training")
    try:
        # Task 1: Load the Data
        df = pd.read_csv(INPUT_FILE)

        # Task 2: Prepare Features and Target
        X = df['cleaned_tweet']
        y = df['class']

        # Task 3: Create Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Task 4: Vectorize the Text
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Task 5: Train the Model
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation='relu',
            solver='adam',
            learning_rate_init=0.01,
            max_iter=40,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            alpha=0.001,
            random_state=42,
            verbose=True
        )
        model.fit(X_train_vec, y_train)

        # Task 6: Evaluate Performance
        predictions = model.predict(X_test_vec)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nTest Accuracy: {accuracy:.4f}")

        # Task 7: Save the Artifacts
        artifacts = {'vectorizer': vectorizer, 'model': model}
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(artifacts, f)

        print(f"\n--- Model and Vectorizer saved successfully to {OUTPUT_FILE}! ---")
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Prediction Example (This part is complete for you)

def clean_text(text):
    """Simple preprocessing: lowercase and remove non-letters"""
    return re.sub(r'[^a-z\s]', '', text.lower())

def run_prediction_example():
    """Loads the saved model and runs predictions on sample tweets."""
    print("\n--- Running Prediction Example ---")
    try:
        with open(OUTPUT_FILE, 'rb') as f:
            artifacts = pickle.load(f)
        
        vectorizer = artifacts['vectorizer']
        model = artifacts['model']
        
        sample_tweets = [
            "I hate you so much! You're the worst person ever.",
            "Check out my new blog post at http://example.com",
            "Had a great time at the concert last night!"
        ]
        
        for tweet in sample_tweets:
            cleaned_tweet = clean_text(tweet)
            tweet_vec = vectorizer.transform([cleaned_tweet])
            prediction = model.predict(tweet_vec)[0]
            
            print(f"\nTweet: '{tweet}'")
            print(f"Predicted Class: {prediction}")
            
    except FileNotFoundError:
        print(f"Could not load '{OUTPUT_FILE}'. Please train the model first by running the script.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    train_model()
    run_prediction_example()
    
    