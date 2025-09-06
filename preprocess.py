import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels
fake_df['label'] = 0  # Fake
true_df['label'] = 1  # Real

# Merge
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df[['text', 'label']]

# Handle missing or invalid text
df['text'] = df['text'].fillna('')  # Replace NaN with empty string
df['text'] = df['text'].astype(str)  # Convert all to string

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''  # Return empty string for invalid or empty input
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Drop rows where cleaned_text is empty
df = df[df['cleaned_text'] != '']

# For basic models: TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(X_tfidf, 'X_tfidf.pkl')
joblib.dump(y, 'y.pkl')

print("Preprocessing done. Files saved: preprocessed_data.csv, tfidf_vectorizer.pkl, X_tfidf.pkl, y.pkl")
