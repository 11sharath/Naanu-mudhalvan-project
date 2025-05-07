import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import os

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Sample dataset
def create_sample_data():
    data = {
        'post': [
            "COVID-19 is a hoax created by the government.",
            "NASA confirms water on the moon.",
            "Click here to win a free iPhone!",
            "The Earth is flat and NASA is lying.",
            "Apple releases new iPhone with satellite connectivity.",
            "Bill Gates is controlling people with vaccines.",
            "Reddit bans misinformation subreddits.",
            "Aliens have landed in California – confirmed!"
        ],
        'label': [0, 1, 0, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv("reddit_fake_news.csv", index=False)
    return df

# Main function
def main():
    if not os.path.exists("reddit_fake_news.csv"):
        df = create_sample_data()
    else:
        df = pd.read_csv("reddit_fake_news.csv")

    df['cleaned'] = df['post'].apply(clean_text)

    # Features
    x = df['cleaned']
    y = df['label']

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=500)
    x_vec = vectorizer.fit_transform(x)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=42)

    # Model training
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Evaluation
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    # Full prediction
    df['prediction'] = model.predict(x_vec)
    df['prediction_label'] = df['prediction'].apply(lambda x: "Real" if x == 1 else "Fake")

    # Save to Excel
    df.to_excel("predictions.xlsx", index=False)
    print("\n Predictions saved to 'predictions.xlsx'")

if __name__ == "__main__":
    main()
