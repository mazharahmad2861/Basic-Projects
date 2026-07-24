import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# Initial Setup: NLTK Stopwords
# A one-time check and download for the NLTK stopwords list.
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    # 1. Remove URLs and mentions
    # use this regular expression: # r'http\S+|www\S+|@\w+'
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)

    # 2. Remove hashtag symbol but keep the word
    text = re.sub(r'#', '', text)

    # 3. Convert to lowercase
    text = text.lower()

    # 4. Remove punctuation, digits, and non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Split into words
    words = text.split()

    # 6. Remove stopwords
    cleaned_words = [word for word in words if word not in STOP_WORDS]

    # 7. Join back into a single string
    return ' '.join(cleaned_words)


if __name__ == "__main__":
    INPUT_FILE = 'hate_dataset.csv'
    OUTPUT_FILE = 'cleaned_hate_dataset.csv'

    # 1. Load the dataset
    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{INPUT_FILE}'.")
        print("Please ensure the dataset is in the correct directory.")
        exit(1)

    # 2. Clean tweets
    print("Cleaning tweets...")
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)

    # 3. Balance dataset
    print("Balancing dataset...")
    df_hate = df[df['class'] == 0]      # Hate Speech
    df_offensive = df[df['class'] == 1] # Offensive
    df_neutral = df[df['class'] == 2]   # Neutral

    df_offensive_sampled = df_offensive.sample(n=3500, random_state=42)
    balanced_df = pd.concat([df_hate, df_offensive_sampled, df_neutral], ignore_index=True)

    # 4. Save final dataset
    print(f"Saving cleaned and balanced data to '{OUTPUT_FILE}'...")
    balanced_df[['cleaned_tweet', 'class']].to_csv(OUTPUT_FILE, index=False)

    print("Data cleaning and balancing complete.")
    print(f"Final dataset saved to {OUTPUT_FILE}")
    
    print(balanced_df.tail(5))