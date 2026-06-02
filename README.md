# TruthScan — Fake News Detection System
### Final Year Project | Islamia University of Bahawalpur
**Student:** Zainab Bibi (F22BINFT1M01069)  
**Supervisor:** Sir Faisal Shahzad

---

## Project Overview
An automated Fake News Detection System using Natural Language Processing (NLP) and Machine Learning. Users paste a news article and the system classifies it as **Real** or **Fake** with a confidence score.

## Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML / NLP | scikit-learn, NLTK |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Vanilla JS |
| Feature Extraction | TF-IDF (bigrams) |
| Classifier | Logistic Regression |

## Project Structure
```
fake_news_detector/
├── train_model.py      ← Step 1: Train and save the ML model
├── app.py              ← Step 2: Run the Flask web server
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Frontend UI
└── model/              ← Created automatically after training
    └── fake_news_pipeline.pkl
```

## Setup & Run

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model
```bash
python train_model.py
```
This will:
- Preprocess the dataset (tokenization, stop-word removal, lemmatization)
- Extract TF-IDF features
- Train a Logistic Regression classifier
- Save the model to `model/fake_news_pipeline.pkl`
- Print accuracy and classification report

### Step 3 — Start the Web App
```bash
python app.py
```

### Step 4 — Open in Browser
```
http://localhost:5000
```

---

## NLP Pipeline
1. **Text Cleaning** — lowercase, remove URLs, punctuation, numbers
2. **Tokenization** — split text into tokens
3. **Stop-word Removal** — remove common words (the, is, at…)
4. **Lemmatization** — reduce words to root form (running → run)
5. **TF-IDF Vectorization** — convert text to numeric feature matrix
6. **Classification** — Logistic Regression predicts Real / Fake

## Using Real Dataset (Recommended)
For better accuracy, download the Kaggle "Fake and Real News Dataset":
1. Visit: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download `Fake.csv` and `True.csv`
3. Place them in the project folder
4. Modify `train_model.py` to load from CSV:
```python
fake = pd.read_csv('Fake.csv'); fake['label'] = 'FAKE'
real = pd.read_csv('True.csv');  real['label'] = 'REAL'
df = pd.concat([fake[['text','label']], real[['text','label']]]).sample(frac=1)
```

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

All printed automatically after `python train_model.py`.
