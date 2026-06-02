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

real_news_samples = [
    "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The discovery was made by researchers from the Marine Biology Institute during a routine expedition.",
    "The Federal Reserve announced today that it will maintain interest rates at current levels following its quarterly meeting. The decision was unanimous among board members.",
    "NASA successfully launched a new weather satellite into orbit early Tuesday morning. The satellite will improve forecasting accuracy across North America.",
    "A new study published in the Journal of Medicine shows that regular exercise can reduce the risk of heart disease by up to 35 percent.",
    "The government released its annual budget report showing a slight decrease in the national deficit compared to the previous fiscal year.",
    "Researchers at MIT have developed a new battery technology that could double the range of electric vehicles on a single charge.",
    "The United Nations held its annual climate conference in Geneva this week, with representatives from 190 countries attending.",
    "Local authorities have confirmed that the water supply in the city meets all safety standards after recent tests were conducted.",
    "The stock market closed higher on Wednesday following positive economic data and strong corporate earnings reports.",
    "Health officials confirmed that the flu vaccination campaign has begun across all major cities with over 500 distribution centers.",
    "Scientists have confirmed that the global average temperature rose by 1.1 degrees Celsius above pre-industrial levels last year.",
    "The World Health Organization released new guidelines for managing chronic diseases in low-income countries.",
    "A new bridge connecting the two cities is set to open next month after three years of construction.",
    "The education ministry announced new curriculum changes that will be implemented starting from the next academic year.",
    "Archaeologists discovered ancient artifacts dating back 3,000 years during excavations in southern Italy.",
    "The central bank raised interest rates by 0.25 percent to combat rising inflation, as expected by most economists.",
    "A team of international researchers has mapped the entire genome of a rare tropical plant with potential medicinal uses.",
    "The annual trade deficit narrowed last quarter due to increased exports in the technology and agriculture sectors.",
    "Local hospitals reported a decrease in emergency room visits following the launch of a new community health initiative.",
    "Parliament passed new legislation to improve data privacy protections for citizens, effective from next January.",
    "The Olympic committee confirmed the host city for the next summer games following a vote by member nations.",
    "A large solar farm was inaugurated in the desert region, expected to power over 200,000 homes annually.",
    "The transportation department released traffic statistics showing a 15 percent reduction in road accidents this year.",
    "University enrollment numbers rose by 8 percent this academic year, driven largely by online program expansion.",
    "A diplomatic agreement was signed between the two neighboring countries to facilitate cross-border trade.",
]

fake_news_samples = [
    "SHOCKING: Government secretly putting mind-control chemicals in tap water to control the population! Scientists who found out have gone missing!",
    "BREAKING: Famous celebrity confirms they are actually a reptilian alien sent to infiltrate human society! Exclusive photos inside!",
    "The moon landing was FAKED in a Hollywood studio! New leaked documents prove NASA has been lying for 50 years!",
    "Bill Gates ADMITS in leaked video that COVID vaccines contain microchips to track every citizen worldwide!",
    "URGENT: 5G towers are secretly spreading a new deadly virus! Government covering up thousands of deaths!",
    "You will not believe this MIRACLE CURE that doctors don't want you to know about! Cures cancer in 3 days!",
    "EXPOSED: The earth is actually flat and NASA has been hiding this truth with fake satellite images for decades!",
    "Scientists BRIBED to hide the truth about chemtrails! Planes are spraying mind-altering substances on us daily!",
    "BOMBSHELL: Secret elite society controls all world governments and is planning to reduce population by 90 percent!",
    "This common household item is actually a secret government spy device recording your every conversation! SHARE NOW!",
    "BREAKING: Mainstream media is owned by the deep state and everything they report is completely fabricated lies!",
    "Shocking revelation: Famous politician was secretly replaced by a clone three years ago, insiders reveal!",
    "WARNING: New vaccine causes magnetism in humans! People are sticking forks to their arms as proof!",
    "CONFIRMED: Aliens have landed in Nevada and the army is covering it up! Eye witness videos going viral!",
    "The world will end next Tuesday according to ancient prophecy discovered in hidden manuscript! Prepare now!",
    "Secret society of bankers controls the weather using HAARP technology to destroy crops and cause food shortages!",
    "EXPOSED: Major tech companies are reading your thoughts through your smartphone's camera using AI mind reading!",
    "MIRACLE: Man cures stage 4 cancer by drinking bleach solution! Big Pharma trying to silence him!",
    "Ancient pyramids were built by aliens thousands of years before humans existed, newly decoded hieroglyphs reveal!",
    "WARNING: New law will allow government to enter your home without warrant and seize all electronics starting Monday!",
    "Famous actor found to be secret government agent spying on Hollywood elites for decades, insider claims!",
    "BREAKING: Doctors confirm that eating sugar and salt cures all known viruses, but pharmaceutical lobby hides truth!",
    "Leaked documents prove the moon is actually a hollow spacecraft used as a military base by world governments!",
    "Hundreds die from new experimental drug but government orders hospitals to report deaths as natural causes!",
    "SHOCKING: Birds are not real animals but government surveillance drones that have been deployed since the 1970s!",
]

# Build dataframe
texts  = real_news_samples + fake_news_samples
labels = ['REAL'] * len(real_news_samples) + ['FAKE'] * len(fake_news_samples)

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

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
