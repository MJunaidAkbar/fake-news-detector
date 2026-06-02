"""
Fake News Detection - Model Training Script
===========================================
Run this script ONCE to train and save the model.
Usage: python train_model.py

It will download the dataset automatically and save:
  - model/tfidf_vectorizer.pkl
  - model/fake_news_model.pkl
  - model/label_encoder.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Download NLTK data ────────────────────────────────────────────────────────
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─── Text Preprocessing ───────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and normalize news article text."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ─── Create Sample Dataset ────────────────────────────────────────────────────
# In real project: use kaggle "Fake and Real News Dataset"
# Here we create a representative sample for demonstration

print("Preparing dataset...")
fake = pd.read_csv('Fake.csv')
fake['label'] = 'FAKE'
real = pd.read_csv('True.csv')
real['label'] = 'REAL'
df = pd.concat([fake[['text','label']], 
                real[['text','label']]]).sample(frac=1, random_state=42)
df = df.reset_index(drop=True)
print(f"Dataset size: {len(df)} samples  (REAL: {sum(df.label=='REAL')}, FAKE: {sum(df.label=='FAKE')})")

# ─── Preprocess ───────────────────────────────────────────────────────────────
print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess_text)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─── Build & Train Pipeline ───────────────────────────────────────────────────
print("Training model (Logistic Regression + TF-IDF)...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )),
    ('clf', LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    ))
])

pipeline.fit(X_train, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print("\n" + "="*55)
print("         MODEL EVALUATION RESULTS")
print("="*55)
print(f"  Accuracy : {acc*100:.2f}%")
print()
print(classification_report(y_test, y_pred))
print("="*55)

# ─── Save Model ───────────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
with open('model/fake_news_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Save preprocessing function as well
with open('model/preprocess.pkl', 'wb') as f:
    pickle.dump(preprocess_text, f)

print("\nModel saved to: model/fake_news_pipeline.pkl")
print("Training complete! Now run: python app.py")
