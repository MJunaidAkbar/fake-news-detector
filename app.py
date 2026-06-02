"""
Fake News Detection - Flask Web Application
============================================
Run AFTER training: python app.py
Then open: http://localhost:5000
"""

import os
import re
import pickle
import string
import nltk
from flask import Flask, render_template, request, jsonify

# ─── NLTK setup ───────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

app        = Flask(__name__)
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

# ─── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join('model', 'fake_news_pipeline.pkl')
pipeline   = None

def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        print("✅  Model loaded successfully.")
    else:
        print("⚠️   Model not found. Please run: python train_model.py")

# ─── Preprocessing (must match training) ─────────────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text   = text.lower()
    text   = re.sub(r'http\S+|www\S+', '', text)
    text   = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.',
            'status': 'error'
        }), 500

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided.', 'status': 'error'}), 400

    news_text = data['text'].strip()
    if len(news_text) < 20:
        return jsonify({'error': 'Please enter a longer news article (at least 20 characters).', 'status': 'error'}), 400

    # Preprocess & predict
    clean     = preprocess_text(news_text)
    prediction = pipeline.predict([clean])[0]
    proba      = pipeline.predict_proba([clean])[0]

    # Map to readable output
    label_idx      = list(pipeline.classes_).index(prediction)
    confidence     = round(float(proba[label_idx]) * 100, 1)
    fake_prob      = round(float(proba[list(pipeline.classes_).index('FAKE')]) * 100, 1)
    real_prob      = round(float(proba[list(pipeline.classes_).index('REAL')]) * 100, 1)

    # Confidence level text
    if confidence >= 90:
        conf_level = "Very High"
    elif confidence >= 75:
        conf_level = "High"
    elif confidence >= 60:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    return jsonify({
        'status':       'success',
        'prediction':   prediction,                       # 'REAL' or 'FAKE'
        'confidence':   confidence,
        'conf_level':   conf_level,
        'real_prob':    real_prob,
        'fake_prob':    fake_prob,
        'word_count':   len(news_text.split()),
        'char_count':   len(news_text),
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': pipeline is not None})

# ─── Main ─────────────────────────────────────────────────────────────────────
load_model()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
